# config.py
"""Configuration settings for the application"""

# Graph Index selecor 
GraphIndexZ = 0
NodeIndexz = None




ExplanationLogic= None


# Global trigger variable
CPU_MONITOR_TRIGGER = False
SelectedNodeCoalition =None
Gloret_list =0

# Mcpu_percent = 0
# Mmemory_percent = 0           
# Mcpu_freq = 0               
# Mload_avg = 0

# ModelAndDataset Selector
# All are 3-layer GNNs
# If 1 then BA_shapes GCN, 2 then MUTAG GIN , 3 then MUTAG GCN , 4 then BBBP GCN
# 5 then Graph-sst2 GAT, 6 then BA_2motifs GCN

ModelDatasetSel = 2
DargumentX = 'MCTS'
# DargumentX = 'MODIFIED'



if ModelDatasetSel == 1 :
    Gmin_atoms = 5
    strategy = DargumentX
    
    if  DargumentX == 'MCTS' :
        DataSel = 'BA_shapes'   # Strictly node level prediction dataset with GCN_2l
        DargumentX = 'MCTS' #  2 is when using original MCTS and 1 is for modified case
        ModelSel = 'GCN_3l'     # powerful with node-based task
        SubgraphAlgoSel =1  # this is the default logic value
        ExplanationLevel = 'Node'
    elif DargumentX == 'MODIFIED' :
        DataSel = 'BA_shapes'   # Strictly node level prediction dataset with GCN_2l
        DargumentX = 'MODIFIED' #  2 is when using original MCTS and 1 is for modified case
        ModelSel = 'GCN_3l'     # powerful with node-based task
        SubgraphAlgoSel =3  # this is the default logic value
        ExplanationLevel = 'Node'
        NumOfCombinationSets = 3 # For Sub-Algorithm Logic.option 1, combinations are not needed. Number of combinations to form basically 3 is good for GIN most cases , but 5 is GCN

elif ModelDatasetSel == 2 :
    Gmin_atoms = 3
    strategy = DargumentX
    
    if  DargumentX == 'MCTS' :
        DataSel = 'mutag'
        DargumentX = 'MCTS'
        ModelSel = 'GIN_3l' 
        SubgraphAlgoSel =1  # this is the default logic value
        ExplanationLevel = 'graph'
    elif DargumentX == 'MODIFIED' :
        DataSel = 'mutag'
        DargumentX = 'MODIFIED'
        ModelSel = 'GIN_3l' 
        SubgraphAlgoSel =3 # this is the default logic value
        ExplanationLevel = 'graph'
        NumOfCombinationSets = 6

elif ModelDatasetSel == 3 :
     Gmin_atoms = 3
     strategy = DargumentX
     
     if  DargumentX == 'MCTS' :
        DataSel = 'mutag'
        DargumentX = 'MCTS'
        ModelSel = 'GCN_3l' 
        SubgraphAlgoSel =1  # this is the default logic value
        ExplanationLevel = 'graph'
     elif DargumentX == 'MODIFIED' :
        DataSel = 'mutag'
        DargumentX = 'MODIFIED'
        ModelSel = 'GCN_3l' 
        SubgraphAlgoSel =3  # this is the default logic value
        ExplanationLevel = 'graph'
        NumOfCombinationSets = 6
elif ModelDatasetSel == 4 :
    Gmin_atoms = 3
    strategy = DargumentX
    
    if  DargumentX == 'MCTS' :
        DataSel = 'BBBP'
        DargumentX = 'MCTS'
        ModelSel = 'GCN_3l' 
        SubgraphAlgoSel =1  # this is the default logic value
        ExplanationLevel = 'graph'
    elif DargumentX == 'MODIFIED' :
        DataSel = 'BBBP'
        DargumentX = 'MCTS'
        ModelSel = 'GCN_3l' 
        SubgraphAlgoSel =3  # this is the default logic value
        ExplanationLevel = 'graph'
        NumOfCombinationSets = 6
elif ModelDatasetSel == 5 :
    Gmin_atoms = 3
    strategy = DargumentX

    if  DargumentX == 'MCTS' :
        DataSel = 'graph_sst2'
        DargumentX = 'MCTS'
        ModelSel = 'GAT_3l' 
        SubgraphAlgoSel =1  # this is the default logic value
        ExplanationLevel = 'graph'
    elif DargumentX == 'MODIFIED' :
        DataSel = 'graph_sst2'
        DargumentX = 'MCTS'
        ModelSel = 'GAT_3l' 
        SubgraphAlgoSel =3  # this is the default logic value
        ExplanationLevel = 'graph'
        NumOfCombinationSets = 6
elif ModelDatasetSel == 6 :
    Gmin_atoms = 5
    strategy = DargumentX

    if  DargumentX == 'MCTS' :
        DataSel = 'ba_2motifs'
        DargumentX = 'MCTS'
        ModelSel = 'GCN_3l' 
        SubgraphAlgoSel =1  # this is the default logic value
        ExplanationLevel = 'graph'
    elif DargumentX == 'MODIFIED' :
        DataSel = 'ba_2motifs'
        DargumentX = 'MCTS'
        ModelSel = 'GCN_3l' 
        SubgraphAlgoSel =3  # this is the default logic value
        ExplanationLevel = 'graph'
        NumOfCombinationSets = 6


# min_atoms
# if DataSel == 'BA_shapes' :
#     Gmin_atoms = 5
# elif DataSel == 'mutag' :
#     Gmin_atoms = 5
# elif DataSel == 'ba_2motifs' :
#     Gmin_atoms = 5
# elif DataSel == 'BBBP' :
#     Gmin_atoms = 5
# elif DataSel == 'graph_sst2' :
#     Gmin_atoms = 5


# Global variables
# Selectors in order for node level explanation [Dataset,Algorithm,Model]
# DataSel = 'BA_shapes'   # Strictly node level prediction dataset with GCN_2l
# DargumentX = 'MCTS' #  2 is when using original MCTS and 1 is for modified case
# ModelSel = 'GCN_2l'     # powerful with node-based task
# SubgraphAlgoSel =1  # this is the default logic value
# ExplanationLevel = 'Node'

# DataSel = 'BA_shapes'   # Strictly node level prediction dataset with GCN_2l
# DargumentX = 'MCTS' #  2 is when using original MCTS and 1 is for modified case
# ModelSel = 'GCN_3l'     # powerful with node-based task
# SubgraphAlgoSel =1  # this is the default logic value



# Selectors in order for graph level explanation [Dataset,Algorithm,Model]
# DataSel = 'mutag'
# DargumentX = 'MCTS'
# ModelSel = 'GCN_2l' 
# SubgraphAlgoSel =1  # this is the default logic value

# DataSel = 'mutag'
# DargumentX = 'MCTS'
# ModelSel = 'GIN_2l' 
# SubgraphAlgoSel =1  # this is the default logic value

# DataSel = 'mutag'
# DargumentX = 'MCTS'
# ModelSel = 'GCN_3l' 
# SubgraphAlgoSel =1  # this is the default logic value
# ExplanationLevel = 'graph'

# DataSel = 'mutag'
# DargumentX = 'MCTS'
# ModelSel = 'GIN_3l'
# SubgraphAlgoSel =1  # this is the default logic value


# DataSel = 'ba_2motifs'
# DargumentX = 'MCTS'
# ModelSel = 'GCN_2l' 
# SubgraphAlgoSel =1  # this is the default logic value


#  Run to this end so far

# DataSel = 'BBBP'
# DargumentX = 'MCTS'
# ModelSel = 'GCN_3l' 
# SubgraphAlgoSel =1  # this is the default logic value

# DataSel = 'BBBP'     # Not available yet
# DargumentX = 'MCTS'
# ModelSel = 'GIN_3l' 
# SubgraphAlgoSel =1  # this is the default logic value


# DataSel = 'graph_sst2'
# DargumentX = 'MCTS'
# ModelSel = 'GCN_3l' 
# SubgraphAlgoSel =1  # this is the default logic value



# Configurations for the modified algorithm


# Selectors in order for node level explanation [Dataset,Algorithm,Model,Sub-AlgorithmLogic]
# DataSel = 'BA_shapes'   # Strictly node level prediction dataset with GCN_2l
# DargumentX = 'MODIFIED'
# ModelSel = 'GCN_2l'
# SubgraphAlgoSel = 3 #  Sub-Algorithm Logic.option 1 is for node level pruning + ego-network and 3 is for pruning + ego-network + combinatorial . Graph level; option 2 selects the max subgraph, option 3 defines the subgraph always to the min limit
# NumOfCombinationSets = 3 # For Sub-Algorithm Logic.option 1, combinations are not needed. Number of combinations to form basically 3 is good for GIN most cases , but 5 is GCN



# Selectors in order for node level explanation [Dataset,Algorithm,Model,Sub-AlgorithmLogic]

# DataSel = 'mutag'
# DargumentX = 'MODIFIED'
# ModelSel = 'GCN_2l'
# SubgraphAlgoSel = 3 # option 2 has slightly better fidelity but 3 has better duration ( both have better fidelity than MCTS)
# NumOfCombinationSets = 6  

# DataSel = 'mutag'
# DargumentX = 'MODIFIED'
# ModelSel = 'GIN_2l'
# SubgraphAlgoSel = 3 # option 3 with 6 combinations seems better under GIN_2l
# NumOfCombinationSets = 6  


# DataSel = 'mutag'
# DargumentX = 'MODIFIED'
# ModelSel = 'GCN_3l'
# SubgraphAlgoSel = 3 
# NumOfCombinationSets = 6


# DataSel = 'mutag'
# DargumentX = 'MODIFIED'
# ModelSel = 'GIN_3l'
# SubgraphAlgoSel = 3 
# NumOfCombinationSets = 6


# DataSel = 'ba_2motifs'
# DargumentX = 'MODIFIED'
# ModelSel = 'GCN_2l'
# SubgraphAlgoSel = 3 # option 3 with 6 combinations seems better under GIN_2l
# NumOfCombinationSets = 6  


# DataSel = 'BBBP'
# DargumentX = 'MODIFIED'
# ModelSel = 'GCN_3l'
# SubgraphAlgoSel = 3 # option 3 with 6 combinations seems better under GIN_2l
# NumOfCombinationSets = 6  

# DataSel = 'BBBP'     # Not available yet
# DargumentX = 'MODIFIED'
# ModelSel = 'GIN_3l'
# SubgraphAlgoSel = 3 # option 3 with 6 combinations seems better under GIN_2l
# NumOfCombinationSets = 6 


# DataSel = 'graph_sst2'    
# DargumentX = 'MODIFIED'
# ModelSel = 'GCN_3l'
# SubgraphAlgoSel = 3 # option 3 with 6 combinations seems better under GIN_2l
# NumOfCombinationSets = 6 





# DataSel = 'ba_2motifs'  # Strictly grpah level prediction with GCN_2l
                          # The house motif is the distinguishing feature for one class (label 0). & The cycle motif (a 5-node cycle) is the distinguishing feature for the other class (label 1).
                          # GCN_2l, Use  Algorithm  2 & 3 ( Best is 3)

#DataSel = 'mutag' # class 0: non-mutagenic & class 1: mutagenic runs well on Algorithm 1 & 2 (GIN_2l with 2 especially)
#                   GCN_2l use Algorithm 2 & 3 ( best 3 )  and for GIN_2l use 2 & 3 (best 2) for mutag 
#
# DataSel = 'BBBP' #    Goes with GCN_3l




