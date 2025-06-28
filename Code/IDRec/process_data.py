# This file serves the purpose of transforming raw interaction data into the data forms required to execute IDRec baselines.


import pandas as pd
import numpy as np
import os
import torch

SEQ_LEN = 10
file_l = ['MicroLens-100k_pairs.tsv']
data_l = ['ks']
for idx in range(len(file_l)):
    dat_seq = pd.read_csv(file_l[idx], sep='\t', header=None)
    dat_arr = np.array(dat_seq)
    inter = []
    for seq in dat_arr:
        uid = seq[0]
        iseq = seq[1].split()
        for i, item in enumerate(iseq):
            inter.append([item, uid, i])

    inter_df = np.array(inter)
    print("inter_df",inter_df.shape)
    dat = pd.DataFrame(inter_df)
    dat.columns = ['item_id', 'user_id', 'timestamp']
    dat['timestamp'] = dat['timestamp'].astype(int)
    dat.sort_values(by='timestamp', inplace=True, ascending=True)
    user_list = dat['user_id'].values
    item_list = dat['item_id'].values
    # item_ids = set(range(1, 19739))
    # # print("item_ids",item_ids)
    # item_list = dat['item_id'].values
    # # print("item_list",item_list)
    # print("len",len(item_list))
    present_ids = set(item_list)  # 将字符串转换为整数集合 
    present_ids = sorted(present_ids)
    print("present_ids",len(present_ids))
    # diff_set = item_ids.symmetric_difference(present_ids)
    # print("missing_ids", diff_set)
    # print("missing_ids", len(diff_set))
    

    index = {}
    for i, key in enumerate(user_list):
        if key not in index:
            index[key] = [i]
        else:
            index[key].append(i)

            indices = []

    for index in index.values():
        indices.extend(list(index)[-(SEQ_LEN + 3):])

    final_dat = dict()
    for k in dat:
        final_dat[k] = dat[k].values[indices]

    final_dat = pd.DataFrame(final_dat)
    print(final_dat)
    print(final_dat['user_id'].nunique(), final_dat['item_id'].nunique(), final_dat.shape[0])
    os.makedirs(f'./{data_l[idx]}/', exist_ok=True)
 # 获取final_dat中的所有物品ID  
    present_item_ids = set(final_dat['item_id'].astype(int).values)  
# 创建一个包含1到19738的所有物品ID的集合  
    all_item_ids = set(range(1,19739))  
# 计算缺失的物品ID  
    missing_item_ids = all_item_ids - present_item_ids  
# 打印缺失的物品ID  
    print("未出现的物品ID有:", sorted(missing_item_ids))  
    print("缺失物品ID的数量:", len(missing_item_ids))
# 创建一个长度为19738的数组，初始值为0  
    item_status = np.zeros(19738, dtype=int)  
# 将存在的物品ID对应的位置设置为1  
    for item_id in present_item_ids:  
        if 1 <= item_id <= 19738:  # 确保物品ID在1到19738之间  
            item_status[item_id - 1] = 1  # 将对应位置设置为1（索引从0开始）  
    print(np.sum(item_status))
    np.save('item_status.npy', item_status)  
    item_status = torch.from_numpy(item_status)
    print("item_status",item_status.shape)
    final_dat.to_csv(f'./{data_l[idx]}/{data_l[idx]}.inter', index=False)
# %%
# The following part generates the popularity count file (i.e. the pop.npy file needed in baseline code) of the dataset


SEQ_LEN = 10


class Data:
    def __init__(self, df):
        self.inter_feat = df
        self._data_processing()

    def _data_processing(self):

        self.id2token = {}
        self.token2id = {}
        remap_list = ['user_id', 'item_id']
        for feature in remap_list:
            feats = self.inter_feat[feature]
            new_ids_list, mp = pd.factorize(feats)
            mp = np.array(['[PAD]'] + list(mp))
            # mp = np.array(list(mp))
            print(mp)
            token_id = {t: i for i, t in enumerate(mp)}
            self.id2token[feature] = mp
            self.token2id[feature] = token_id
            self.inter_feat[feature] = new_ids_list + 1

        self.user_num = len(self.id2token['user_id'])
        self.item_num = len(self.id2token['item_id'])
        print(self.id2token['user_id'])
        print(self.user_num,self.item_num)
        self.inter_num = len(self.inter_feat)
        self.uid_field = 'user_id'
        self.iid_field = 'item_id'
        self.user_seq = None
        self.train_feat = None
        self.feat_name_list = ['inter_feat']

    def build(self):

        self.sort(by='timestamp')
        user_list = self.inter_feat['user_id'].values
        item_list = self.inter_feat['item_id'].values
        grouped_index = self._grouped_index(user_list)

        user_seq = {}
        for uid, index in grouped_index.items():
            user_seq[uid] = item_list[index]

        self.user_seq = user_seq
        train_feat = dict()
        test_feat = dict()
        valid_feat = dict()
        indices = []

        for index in grouped_index.values():
            indices.extend(list(index)[:-2])
        for k in self.inter_feat:
            train_feat[k] = self.inter_feat[k].values[indices]

        indices = []
        for index in grouped_index.values():
            indices.extend([index[-2]])
        for k in self.inter_feat:
            valid_feat[k] = self.inter_feat[k].values[indices]

        indices = []
        for index in grouped_index.values():
            indices.extend([index[-1]])
        for k in self.inter_feat:
            test_feat[k] = self.inter_feat[k].values[indices]

        self.train_feat = train_feat
        return train_feat, valid_feat, test_feat

    def _grouped_index(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index

    def _build_seq(self, train_feat):
        max_item_list_len = SEQ_LEN + 1
        uid_list, item_list_index = [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1)
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid:
                save = True
            if save:
                if i - seq_start > max_item_list_len:
                    offset = (i - seq_start) % max_item_list_len
                    seq_start += offset
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        item_list_index.append(slice(sub[0], sub[-1] + 1))


                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(seq_start, i))

                save = False
                last_uid = uid
                seq_start = i

        seq_train_feat = {}
        seq_train_feat['user_id'] = np.array(uid_list)
        seq_train_feat['item_seq'] = []
        seq_train_item = []
        for index in item_list_index:
            seq_train_feat['item_seq'].append(train_feat['item_id'][index])
            seq_train_item += list(train_feat['item_id'][index])

        self.seq_train_item = seq_train_item
        return seq_train_feat

    def sort(self, by, ascending=True):
        self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True)


data_list = ['ks', ]

for idx in range(len(data_list)):
    inter = pd.read_csv(f'./{data_list[idx]}/{data_list[idx]}.inter', delimiter=',',
                        dtype={'item_id': str, 'user_id': str, 'timestamp': int}, header=0,
                        names=['item_id', 'user_id', 'timestamp']
                        )

    item_num = inter['item_id'].nunique()
    D = Data(inter)
    train, valid, test = D.build()
    D._build_seq(train)
    train_items = D.seq_train_item
    # print("train_items",len(train_items))
    train_item_counts = [0] * (item_num + 1)
    for i in train_items:
        train_item_counts[i] += 1
    item_counts_powered = np.power(train_item_counts, 1.0)
    pop_prob_list = []

    for i in range(1, item_num + 1):
        pop_prob_list.append(item_counts_powered[i])
    pop_prob_list = pop_prob_list / sum(np.array(pop_prob_list))
    pop_prob_list = np.append([1], pop_prob_list)
    print(('prob max: {}, prob min: {}, prob mean: {}'. \
           format(max(pop_prob_list), min(pop_prob_list), np.mean(pop_prob_list))))

    np.save(f'./{data_list[idx]}/pop', pop_prob_list)
    # print(pop_prob_list)