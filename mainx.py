import sys
import time
import os
import os.path as osp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timeit
import config
# import CPUMonitor
# import PrintOutputToCSV
# import SampleGraphViewing
# from compatibility import compatible_state_dict,compatible_state_dict_new
# from torch_geometric.data import download_url, extract_zip
# from dig.xgraph.dataset import SynGraphDataset
# from models.GCN import GCNNet, GCNNet_NC
# from Configures import model_args
# from load_dataset import *


# Provide a switch here to select a prefered module to execute 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



#This line imports the subgraphx model and inturn calls the SubgraphX class
if  config.ExplanationLevel== 'Node':
 
  from fornode.subgraphx import pipeline
  
  fidelity_scores, sparsity_scores, duration_scores = pipeline(subgraph_max_nodes=10)
  print(f"fidelity score: {fidelity_scores.mean().item()}, sparsity score: {sparsity_scores.mean().item()}, duration score: {duration_scores.mean().item()} ")

else:

  from forgraph.subgraphx import pipeline
  fidelity_scores, sparsity_scores, duration_scores = pipeline(15)
  print(f"fidelity score: {fidelity_scores.mean().item()}, sparsity score: {sparsity_scores.mean().item()}, duration score: {duration_scores.mean().item()} ")







