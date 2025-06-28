# README for VideoGCN:happy::happy::happy:

We follow the work LightGCN([LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (acm.org)](https://dl.acm.org/doi/pdf/10.1145/3397271.3401063)) and MicroLens([westlake-repl/MicroLens: A Large Short-video Recommendation Dataset with Raw Text/Audio/Image/Videos (Talk Invited by DeepMind). (github.com)](https://github.com/westlake-repl/MicroLens/)). As MicroLens have already run LightGCN, so we download their code and do some modification for our VideoGCN.  

Most of our code work can be seen in ".\MicroLens-master\Code\IDRec". We modified some of the code in process_data.ipynb to catch up which videos do not appear in the user-item pairs and we save this information as a statues numpy array file. The modified code can be seen in process_data.py.

Our modification for the model can be seen in ".\MicroLens-master\Code\IDRec\LightGCN\REC\model\IdModel\lightgcn.py". We add transformer encoder layers(unfortunately, we don't have time to add those hyperparameters into a config file) and do cosine similarity to refine item-embeddings. We use markers like"############################" to mark our modification. Some of the lines is commented out because these methods don't enhance the performance.

The method details and training details can be seen in our report.

The environment we use : we first reproduced the result of LightGCN, using its formal code(not the implementation of MicroLens one) , then we reproduce the MicroLens's LightGCN version. New packages are installed if there is some missing.

 :joy_cat::joy_cat::joy_cat: