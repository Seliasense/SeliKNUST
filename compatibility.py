
import re
import math 
import torch
import torch.nn as nn
from torch_geometric import __version__
from collections import OrderedDict

def compatible_state_dict_new(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        comp_key = key
        comp_value = value
   
        comp_key = re.sub(r'^model.', '', key)
        fin_comp_key = re.sub(r'gnn_layers(.[0-9]).weight', 'gnn_layers\g<1>.lin.weight', comp_key)
       
        new_state_dict[fin_comp_key] = comp_value
               
    return new_state_dict



def compatible_state_dict_JJ(state_dict):
    comp_state_dict = OrderedDict()
    for key, value in state_dict.items():
        comp_key = key
        comp_value = value
        if int(__version__[0]) >= 2:
            comp_key = re.sub(r'model.gnn_layers(.[0-9]).weight', 'model.gnn_layers\g<1>.lin.weight', key)
            if comp_key != key:
                comp_value = value.T
        
        if comp_key != key:
            # comp_state_dict[key] = value
            comp_state_dict[comp_key] = comp_value
        else:
            comp_state_dict[key] = value
            
    return comp_state_dict


def compatible_state_dict_graph_sst2(state_dict):
    comp_state_dict = OrderedDict()
    for key, value in state_dict.items():
        comp_key = key
        comp_value = value
        if int(__version__[0]) >= 2:

            comp_key = re.sub(r'model.gnn_layers(.[0-9]).lin_l.weight', 'model.gnn_layers\g<1>.lin.weight', key)
            comp_key_one = re.sub(r'model.gnn_layers(.[0-9]).att_l', 'model.gnn_layers\g<1>.att_src', comp_key)
            comp_key_two = re.sub(r'model.gnn_layers(.[0-9]).att_r', 'model.gnn_layers\g<1>.att_dst', comp_key_one)
     
            if comp_key_two != key:
                comp_value = value.mT
        
        if comp_key_two != key:
            comp_state_dict[comp_key_two] = comp_value.mT
        else:
            comp_state_dict[key] = value
            
    return comp_state_dict



# Original regular expression 

def compatible_state_dict(state_dict):
    comp_state_dict = OrderedDict()
    for key, value in state_dict.items():
        comp_key = key
        comp_value = value
        if int(__version__[0]) >= 2:
            comp_key = re.sub(r'conv(1|s.[0-9]).weight', 'conv\g<1>.lin.weight', key)
            if comp_key != key:
                comp_value = value.T
        if comp_key != key:
            comp_state_dict[key] = value
            comp_state_dict[comp_key] = comp_value
    return comp_state_dict

