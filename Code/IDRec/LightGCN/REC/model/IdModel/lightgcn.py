import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from REC.model.layers import MLPLayers
from REC.utils import InputType
from REC.model.basemodel import BaseModel
from REC.model.layers import LightGCNConv
import os
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

###############################################################################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

##########################################################################################################################################
class CrossAttentionRefiner(nn.Module):
    def __init__(self, feature_dim, embedding_dim):
        super(CrossAttentionRefiner, self).__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

        # linear layer define
        self.feature_to_query = nn.Linear(self.feature_dim, self.embedding_dim)
        self.embedding_to_key = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.embedding_to_value = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, features, embeddings):
        # feature to query
        queries = self.feature_to_query(features)  # Shape: [19672, 512]
        keys = self.embedding_to_key(embeddings)  # Shape: [19672, 512]
        values = self.embedding_to_value(embeddings)  # Shape: [19672, 512]

        # 计算注意力权重
        attention_scores = torch.matmul(queries, keys.t())  # Shape: [19672, 19672]
        attention_scores = attention_scores / (self.embedding_dim ** 0.5)  #
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Softmax

        # 使用注意力权重加权值
        refined_embeddings = torch.matmul(attention_weights, values)  # Shape: [19672, 512]

        return refined_embeddings
####################################################################################

class LightGCN(BaseModel):

    input_type = InputType.PAIR
    
    def __init__(self, config, data):
        super(LightGCN, self).__init__()
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN

        self.device = config['device']
              
        self.user_num = data.user_num
        # print("user_num",self.user_num)
        self.item_num = data.item_num
        # print("item_num",self.item_num)
        
        self.edge_index, self.edge_weight = data.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)

        self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.latent_dim)
        
        self.device = config['device']
        path = os.path.join(config['data_path'], 'pop.npy')
        pop_prob_list = np.load(path)
        self.pop_prob_list = torch.FloatTensor(pop_prob_list).to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.store_ufeatures = None
        self.store_ifeatures = None

########################################################################################################################################################################################################################################
        self.title_features = torch.from_numpy(np.load(os.path.join(config['data_path'], 'MicroLens-100k_title_en_text_features_BgeM3.npy'))).cuda() # [19738, 1024]
        self.cover_features = torch.from_numpy(np.load(os.path.join(config['data_path'], 'MicroLens-100k_image_features_CLIPRN50.npy'))).cuda() # [19738, 1024]
        self.video_features = torch.from_numpy(np.load(os.path.join(config['data_path'], 'MicroLens-100k_video_features_VideoMAE.npy'))).cuda()# [19738, 768]
        self.item_status = torch.from_numpy(np.load(os.path.join(config['data_path'], 'item_status.npy'))).cuda() # [19738]
        self.item_feature = None
        self.item_map = nn.Linear(2816, 512)
        # self.batch_norm = nn.BatchNorm1d(512)
        # self.weight = nn.Parameter(torch.rand((2816, 512)))
        # self.cross_attention = CrossAttentionRefiner(512, 512)

        self.transformer = Transformer(
            width=512,
            layers=2,
            heads=4
        )
############################################################################################################################################################################################################################3

        self.apply(self._init_weights)
     
    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
#################################################################################################
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
###################################################################################################
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def computer(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            # print("edge_weight", self.edge_weight.shape,"edge_index",self.edge_index.shape)
            # edge_weight : tensor[956710]
            # edge_index : tensor[2, 956710]
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        # lightgcn_all_embeddings.shape: [119673,512] 100001(user) + 19672(item)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.user_num, self.item_num])
        return user_all_embeddings, item_all_embeddings

    def forward(self, input):
        user, item = input
        user_all_embeddings, item_all_embeddings = self.computer()
        embed_user = user_all_embeddings[user]
        embed_item = item_all_embeddings[item]
        
        logits = torch.matmul(embed_user, embed_item.t())  #[batch, batch]
        label = torch.arange(item.numel()).to(self.device)
        
        flatten_item_seq = item.view(-1)

        debias_logits = torch.log(self.pop_prob_list[flatten_item_seq])
        logits = logits - debias_logits

        history = flatten_item_seq.unsqueeze(-1).expand(-1, len(flatten_item_seq))
        history_item_mask = (history == flatten_item_seq)
        unused_item_mask = torch.scatter(history_item_mask, 1, label.view(-1, 1), False)
        logits[unused_item_mask] = -1e8
        loss = self.loss_func(logits, label)
        # new loss function
        # loss_contrastive = 0
        # if self.item_feature is not None:
        #     loss_contrastive = F.pairwise_distance(item_all_embeddings, self.item_feature)
        #     loss_contrastive = loss_contrastive[item]
        #     loss_contrastive = torch.sum(loss_contrastive)
        # # 总损失
        # loss = loss + loss_contrastive
        return loss
      
           
    @torch.no_grad()
    def predict(self, user,features_pad):    
        embed_user = self.store_ufeatures[user]       
        scores = torch.matmul(embed_user,self.store_ifeatures.t())
        return scores

    @torch.no_grad()   
    def compute_item_all(self):
############################################################################################################################
        item_features = torch.cat((self.title_features, self.cover_features, self.video_features), dim=1).cuda()
        item_features = item_features * self.item_status.unsqueeze(1)
        non_zero_indices = torch.nonzero(item_features.sum(dim=1) != 0).squeeze()
        item_features = item_features[non_zero_indices]
        random_tensor = torch.randn(1, 2816).cuda()
        self.item_feature = torch.cat((random_tensor, item_features), dim=0)
        self.item_feature = self.item_map(self.item_feature)
        self.item_feature = self.transformer(self.item_feature)
        # self.item_feature = self.item_map(self.item_feature)
        # self.item_feature = self.batch_norm(self.item_feature)
        self.item_feature = F.softmax(self.item_feature, dim=1)  #  Softmax
# [19672,512]
#############################################################################################################################

        self.store_ufeatures, self.store_ifeatures= self.computer()

######################################################################################################3
        # similarity :
        similarity = F.cosine_similarity(self.store_ifeatures, self.item_feature, dim=-1)  # 计算余弦相似度, tensor[19672]
        similarity_matrix = similarity.unsqueeze(1)  # 形状为 [19672, 1]
        self.store_ifeatures = self.store_ifeatures * similarity_matrix + self.item_feature * (1 - similarity_matrix)

        # # cross attention 方法：
        # self.store_ifeatures = self.cross_attention(self.item_feature, self.store_ifeatures)

        # 计算权重的方法
        # self.store_ifeatures = torch.matmul(self.item_feature, self.weight) + self.store_ifeatures

        # for debug
        # store_ifeatures_sums = torch.sum(self.item_feature, dim=1)
        # print(store_ifeatures_sums[:10])
        # store_ifeatures_sums = torch.sum(self.store_ifeatures, dim=1)
        # print(store_ifeatures_sums[:10])
#############################################################################################################
        # store_ufeatures : [100001,512] the first is "pad"
        # store_ifeatures : [19672,512]
#####################################################################################################################
        return None

