import os
import torch
import config
from tap import Tap
from typing import List


class DataParser(Tap):
    if config.ModelDatasetSel == 1 :
       dataset_name: str = 'BA_Shapes' # modified from 'bbbp'
       data_split_ratio: List = [0.8, 0.1, 0.1]   # the ratio of training, validation and testing set for random split
       random_split: bool =  True  #  changed from True
       seed: int = 1 # changed from 1
    elif config.ModelDatasetSel == 2 :
       dataset_name: str = 'mutag' 
       data_split_ratio: List = [0.9, 0.0, 0.1]   # the ratio of training, validation and testing set for random split
       random_split: bool = False  #  changed from True
       seed: int = 1 # changed from 1 ***
    elif config.ModelDatasetSel == 3 :
       dataset_name: str = 'mutag' 
       data_split_ratio: List = [0.9, 0.0, 0.1]   # the ratio of training, validation and testing set for random split
       random_split: bool = False   #  changed from True
       seed: int = 1 # changed from 1 ***
    elif config.ModelDatasetSel == 4 :
       dataset_name: str = 'bbbp' 
       data_split_ratio: List = [0.8, 0.1, 0.1]   # the ratio of training, validation and testing set for random split
       random_split: bool = True
       seed: int = 1 # changed from 1 ***
    elif config.ModelDatasetSel == 5 :
       dataset_name: str = 'Graph_SST2' 
       data_split_ratio: List = [0.9, 0.0, 0.1] # i.e. 67,349, 872 and 1821 (0.96, 0.13, 0.26) # the ratio of training, validation and testing set for random split
       random_split: bool = False # change to False from True
       seed: int = 1 # changed from 1
    elif config.ModelDatasetSel == 6 :
       dataset_name: str = 'ba_2motifs' # modified from 'bbbp'
       data_split_ratio: List = [0.8, 0.1, 0.1]   # the ratio of training, validation and testing set for random split
       random_split: bool =  True  #  changed from True
       seed: int = 1 # changed from 1




   #  dataset_dir: str = '/home/pcshark/SubgraphX/datasets'  # Modified from '../datasets'
    dataset_dir: str = '/content/drive/MyDrive/SubgraphX/datasets'  # Modified from '../datasets'
    


class GATParser(Tap):           # hyper-parameter for gat model
    gat_dropout: float = 0.6    # dropout in gat layer
    gat_heads: int = 10         # multi-head
    gat_hidden: int = 10        # the hidden units for each head
    gat_concate: bool = True    # the concatenation of the multi-head feature
    num_gat_layer: int = 3


class ModelParser(GATParser):
    device_id: int = 0  
    if config.ModelDatasetSel == 1 :
       # BA_shapes (GCN)
       model_name: str = 'gcn'
       latent_dim: List[int] = [20, 20, 20] # the hidden units for each gnn layer
       concate: bool = True   # whether to concate the gnn features before mlp
       emb_normlize: bool = True    # the l2 normalization after gnn layer
       readout: 'str' = 'max'      # readout change from max              # the graph pooling method
       dropout: float = 0.5                      # the dropout after mlp layers
    elif config.ModelDatasetSel == 2 :
       # MUTAG (GIN)
       model_name: str = 'gin'
       latent_dim: List[int] = [128, 128, 128] # the hidden units for each gnn layer
       concate: bool = False                     # whether to concate the gnn features before mlp
       emb_normlize: bool = False                # the l2 normalization after gnn layer
       readout: 'str' = 'max'                    # the graph pooling method
       dropout: float = 0.5                      # the dropout after mlp layers
    elif config.ModelDatasetSel == 3 :
       # MUTAG (GCN)
       model_name: str = 'gcn'
       latent_dim: List[int] = [128, 128, 128] # the hidden units for each gnn layer
       concate: bool = False                     # whether to concate the gnn features before mlp
       emb_normlize: bool = False                # the l2 normalization after gnn layer
       readout: 'str' = 'max'                    # the graph pooling method
       dropout: float = 0.5                      # the dropout after mlp layers
    elif config.ModelDatasetSel == 4 :
       # BBBP (GCN)
       model_name: str = 'gcn'
       latent_dim: List[int] = [128, 128, 128] # the hidden units for each gnn layer
       concate: bool = False                     # whether to concate the gnn features before mlp
       emb_normlize: bool = False                # the l2 normalization after gnn layer
       readout: 'str' = 'max'                    # the graph pooling method
       dropout: float = 0.5                      # the dropout after mlp layers
    elif config.ModelDatasetSel == 5 :
       # Graph-sst2 (GAT)
       model_name: str = 'gat'
       latent_dim: List[int] = [10, 10, 10] # the hidden units for each gnn layer
       concate: bool = True                     # whether to concate the gnn features before mlp
       emb_normlize: bool = False                # the l2 normalization after gnn layer
       readout: 'str' = 'max'                    # the graph pooling method
       dropout: float = 0.6                     # the dropout after mlp layers
    elif config.ModelDatasetSel == 6 :
       #  ba_2motifs (GCN)
       model_name: str = 'gcn'
       latent_dim: List[int] = [20, 20, 20] # the hidden units for each gnn layer
       concate: bool = False                     # whether to concate the gnn features before mlp
       emb_normlize: bool = True                # the l2 normalization after gnn layer
       readout: 'str' = 'mean'                    # the graph pooling method
       dropout: float = 0.5                      # the dropout after mlp layers
    

   #  checkpoint: str = '/home/pcshark/SubgraphX/checkpoints' # modified from './checkpoint'
    checkpoint: str = '/content/drive/MyDrive/SubgraphX/checkpoints' # modified from './checkpoint'
    
    mlp_hidden: List[int] = []                # the hidden units for mlp classifier
    gnn_dropout: float = 0.0                  # the dropout after gnn layers
    
    adj_normlize: bool = True                 # the edge_weight normalization for gcn conv

   

    def process_args(self) -> None:
        self.device = torch.device('cpu')  # I reveresed comment this to allow for cpu usage 
        if torch.cuda.is_available():
            self.device = torch.device('cuda', self.device_id)
        else:
            pass


class MCTSParser(DataParser, ModelParser):
   #  rollout: int = 20                         # the rollout number   : This line was commented out


     # Conditional n_rollout setting based on config.DargumentX
    if config.DargumentX == 'MODIFIED' :
            # Modified
          
            if config.ExplanationLogic == 2 :
               rollout: int = 20 
            else :
               rollout: int = 1
           
    elif config.DargumentX == 'MCTS' :
            # orignal MCTS
            rollout: int = 20 

          
    
    if config.ModelDatasetSel == 1 :
       # BA_shapes (GCN)
       min_atoms: int = 5                       # for the synthetic dataset, change the minimal atoms to 5.
       high2low: bool = True                    # expand children with different node degree ranking method
       c_puct: float = 10                       # the exploration hyper-parameter
    elif config.ModelDatasetSel == 2 :
       # MUTAG (GIN)
       min_atoms: int = 3                        # for the synthetic dataset, change the minimal atoms to 5.                      
       high2low: bool = False                    # expand children with different node degree ranking method
       c_puct: float = 10                        # the exploration hyper-parameter  
    elif config.ModelDatasetSel == 3 :
       # MUTAG (GCN)
       min_atoms: int = 3                        # for the synthetic dataset, change the minimal atoms to 5.                        
       high2low: bool = False                    # expand children with different node degree ranking method
       c_puct: float = 10                         # the exploration hyper-parameter
    elif config.ModelDatasetSel == 4 :
        # BBBP (GCN)
       min_atoms: int = 3                        # for the synthetic dataset, change the minimal atoms to 5.                        
       high2low: bool = False                    # expand children with different node degree ranking method
       c_puct: float = 5                       # the exploration hyper-parameter     
    elif config.ModelDatasetSel == 5 :
       # Graph_SST2 (GAT)
       min_atoms: int = 3                        # for the synthetic dataset, change the minimal atoms to 5.                        
       high2low: bool = False                    # expand children with different node degree ranking method
       c_puct: float = 5                         # the exploration hyper-parameter 
                                   
    elif config.ModelDatasetSel == 6 :
       # ba_2motifs
       min_atoms: int = 5                       # for the synthetic dataset, change the minimal atoms to 5.                        
       high2low: bool = False                    # expand children with different node degree ranking method
       c_puct: float = 10                        # the exploration hyper-parameter     
     


    
    expand_atoms: int = 12                     #  of atoms to expand children

    def process_args(self) -> None:
      #   self.explain_model_path = os.path.join(self.checkpoint,
      #                                          self.dataset_name,
      #                                          f"{self.model_name}_best.pth")

         self.explain_model_path = os.path.join(self.checkpoint,
                                               self.dataset_name,
                                               f"{self.model_name}_latest.pth")
         
        


class RewardParser(Tap):
    reward_method: str = 'mc_l_shapley'                         # Liberal, gnn_score, mc_shapley, l_shapley， mc_l_shapley
    local_raduis: int = 4                                       # (n-1) hops neighbors for l_shapley
    subgraph_building_method: str = 'zero_filling'                  
    sample_num: int = 100                                       # sample time for monte carlo approximation


class TrainParser(Tap):
    learning_rate: float = 0.005
    batch_size: int = 64
    weight_decay: float = 0.0
    max_epochs: int = 800
    save_epoch: int = 10                                        
    early_stopping: int = 100                                  


data_args = DataParser().parse_args(known_only=True)
model_args = ModelParser().parse_args(known_only=True)
mcts_args = MCTSParser().parse_args(known_only=True)
reward_args = RewardParser().parse_args(known_only=True)
train_args = TrainParser().parse_args(known_only=True)

# import torch
# import random
# import numpy as np
# random_seed = 1234
# random.seed(random_seed)
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)
