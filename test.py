
import sys
print(sys.executable)



# # Step 1: Try loading with compatibility (Solution 3)
# data = load_compatible_pickle('ba_shapes.pkl')

# # Step 2: If data loads but you get PyG errors, convert it
# if not hasattr(data, '_store') and hasattr(data, '__dict__'):
#     from torch_geometric.data import Data
#     data = Data.from_dict(data.__dict__)

# # Step 3: Verify the data works with PyG 2.7.0
# print(f"✅ Data ready for PyG 2.7.0")
# print(f"Has _store attribute: {hasattr(data, '_store')}")
# print(f"Node features shape: {data.x.shape if hasattr(data, 'x') else 'N/A'}")
# print(f"Edge index shape: {data.edge_index.shape if hasattr(data, 'edge_index') else 'N/A'}")

# # Step 4: Save in new format for future use
# torch.save(data, 'ba_shapes_converted.pt')
# print("✅ Saved in new format for future loading")



# Protocol-Aware Loading with Fallbacks
# import pickle
# import torch
# import os
# from torch_geometric.data import Data

# def load_compatible_pickle(filepath):
#     """Load pickle file with compatibility handling for different Python versions"""
    
#     # First, check the protocol version
#     with open(filepath, 'rb') as f:
#         # Peek at the first few bytes to identify protocol
#         first_bytes = f.read(20)
#         f.seek(0)
        
#         # Protocol 2 (Python 2.3+) starts with b'\x80\x02'
#         # Protocol 3 (Python 3.0+) starts with b'\x80\x03'
#         # Protocol 4 (Python 3.4+) starts with b'\x80\x04'
#         # Protocol 5 (Python 3.8+) starts with b'\x80\x05'
        
#         if first_bytes.startswith(b'\x80\x02'):
#             print("📦 Detected pickle protocol 2 (Python 2.3-2.7)")
#         elif first_bytes.startswith(b'\x80\x03'):
#             print("📦 Detected pickle protocol 3 (Python 3.0-3.7)")
#         elif first_bytes.startswith(b'\x80\x04'):
#             print("📦 Detected pickle protocol 4 (Python 3.8-3.13)")
#         elif first_bytes.startswith(b'\x80\x05'):
#             print("📦 Detected pickle protocol 5 (Python 3.8+ with out-of-band data)")
    
#     # Try different loading strategies
#     strategies = [
#         # Strategy 1: Standard loading
#         lambda f: pickle.load(f),
#         # Strategy 2: With latin1 encoding (common for Python 2 data)
#         lambda f: pickle.load(f, encoding='latin1'),
#         # Strategy 3: With bytes encoding
#         lambda f: pickle.load(f, encoding='bytes'),
#         # Strategy 4: Force protocol 3 compatibility
#         lambda f: pickle.load(f, encoding='latin1', fix_imports=True)
#     ]
    
#     for i, strategy in enumerate(strategies, 1):
#         try:
#             with open(filepath, 'rb') as f:
#                 data = strategy(f)
#             print(f"✅ Success with strategy {i}")
#             # print(f" Print the loaded data with strategy 1 : {data}")
#             return data
#         except Exception as e:
#             print(f"⚠️ Strategy {i} failed: {type(e).__name__}: {e}")
#             continue
    
#     raise ValueError("❌ All loading strategies failed")

# # Use it
# try:
#     data = load_compatible_pickle('/home/pcshark/SubgraphX/datasets/BA_shapes/raw/BA_shapes.pkl')
    
#     print(f" Print data : {data}")
    
#     print(f" Print attr : {data[0].__dict__}")
#     # If it's a PyG data object from older version, convert it
     
#     if hasattr(data, '__dict__') and not hasattr(data, '_store'):
#         from torch_geometric.data import Data
#         data = Data.from_dict(data.__dict__)
#         print("✅ Converted to PyG 2.x format")
#         print(f" Print the converted data : {data}")
    
#     print(f"Final data type: {type(data)}")
#     if hasattr(data, 'keys'):
#         print(f"Available attributes: {data.keys()}")
        
# except Exception as e:
#     print(f"❌ Failed to load: {e}")



# if __name__ == '__main__':

#     filepath = '/home/pcshark/SubgraphX/datasets/BA_shapes/raw'

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Handle PyG Version Compatibility
# import pickle
# import torch
# import os
# import torch_geometric
# from torch_geometric.data import Data

# def convert_old_pyg_data(obj):
#     """Recursively convert old PyG data objects to new format"""
#     if hasattr(obj, '__dict__') and hasattr(obj, 'keys'):
#         # It's likely a PyG Data object
#         return Data.from_dict(obj.__dict__)
#     elif isinstance(obj, dict):
#         return {k: convert_old_pyg_data(v) for k, v in obj.items()}
#     elif isinstance(obj, (list, tuple)):
#         return type(obj)(convert_old_pyg_data(x) for x in obj)
#     else:
#         return obj

# # Load with encoding handling
# folder='/home/pcshark/SubgraphX/datasets/BA_shapes/raw'
# prefix='BA_shapes'
# with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
#     try:
#         raw_data = pickle.load(f)
#     except UnicodeDecodeError:
#         f.seek(0)
#         raw_data = pickle.load(f, encoding='latin1')

# # Convert if needed
# if hasattr(raw_data, '__dict__') and not hasattr(raw_data, '_store'):
#     print("🔄 Converting old-style PyG data to new format...")
#     data = convert_old_pyg_data(raw_data)
# else:
#     data = raw_data

# print(f"✅ Data loaded with the old format: {raw_data}")

# print(f"✅ Data loaded with the new format: {data}")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This loads  BA_shapes pickle data


# import pickle
# import torch
# import os

# from torch_geometric.data import Data

# try:
#     # Attempt 1: Standard loading
#     folder='/home/pcshark/SubgraphX/datasets/BA_shapes/raw'
#     prefix='BA_shapes'
#     with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
#         data = pickle.load(f)
#         print(f" Print the sliced pickle data : {data[0]}")
#         print(f" Print all loaded pickle data : {data}")
#     print("✅ Successfully loaded with standard pickle!")
    
# except UnicodeDecodeError:
#     # Attempt 2: Handle Python 2/3 encoding issues
#     with open('ba_shapes.pkl', 'rb') as f:
#         data = pickle.load(f, encoding='latin1')
#     print("✅ Loaded with latin1 encoding!")

# except Exception as e:
#     print(f"⚠️ Standard loading failed: {e}")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# x = lambda a : a + 10
# print(x(5)) 

# x = lambda a, b : a * b
# print(x(5, 6)) 



# import pickle
# print(pickle.DEFAULT_PROTOCOL)


# # In Python interactive shell or script
# try:
#     import pickle
#     print("✅ pickle module exists")
#     print(f"pickle version: {pickle.__doc__}")
# except ImportError:
#     print("❌ pickle module does not exist")



# import os
# import torch
# import shutil
# import numpy as np
# from torch_geometric.data import Data, InMemoryDataset
# from torch_geometric.utils import barabasi_albert_graph

# def create_syngraph_dataset_compatible():
#     """Create dataset in the exact format expected by SynGraphDataset"""
    
#     root_dir = '/home/pcshark/SubgraphX/datasets/BA_shapes'
#     processed_dir = os.path.join(root_dir, 'processed')
    
#     # Backup and clean existing data
#     if os.path.exists(processed_dir):
#         backup_dir = processed_dir + '_backup_' + str(int(torch.ones(1).item() * 1000000000))
#         print(f"Backing up old data to {backup_dir}")
#         shutil.move(processed_dir, backup_dir)
    
#     # Create fresh directory
#     os.makedirs(processed_dir, exist_ok=True)
#     print(f"Created directory: {processed_dir}")
    
#     # Generate synthetic graph data
#     print("Generating BA Shapes dataset...")
    
#     # Parameters for BA Shapes
#     num_graphs = 1  # Usually BA Shapes has 1 graph
#     num_nodes = 300
#     num_motifs = 5
#     motif_size = 5  # House motif has 5 nodes
    
#     # Create data list
#     data_list = []
    
#     for i in range(num_graphs):
#         # Create base BA graph
#         edge_index = barabasi_albert_graph(num_nodes, num_edges=3)
        
#         # Create node features (identity matrix for synthetic data)
#         x = torch.eye(num_nodes, num_nodes)
        
#         # Create labels (0 for base nodes, 1 for motif nodes)
#         y = torch.zeros(num_nodes, dtype=torch.long)
        
#         # Mark motif nodes (simplified - in reality these would be house structures)
#         # You'll need to adjust this based on your actual motif placement
#         motif_indices = torch.randperm(num_nodes)[:num_motifs * motif_size]
#         y[motif_indices] = 1
        
#         # Create node mask for motif nodes
#         node_mask = torch.zeros(num_nodes, dtype=torch.bool)
#         node_mask[motif_indices] = True
        
#         # Create edge mask (if needed)
#         edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        
#         # Create Data object
#         data = Data(
#             x=x,
#             edge_index=edge_index,
#             y=y,
#             node_mask=node_mask,
#             edge_mask=edge_mask,
#             num_nodes=num_nodes
#         )
        
#         data_list.append(data)
    
#     # Collate the data list into the format expected by SynGraphDataset
#     # This creates a tuple (data, slices) where:
#     # - data is a concatenated Data object
#     # - slices is a dictionary with slice indices for each attribute
    
#     if len(data_list) == 1:
#         # For single graph, create appropriate slices
#         data = data_list[0]
#         slices = {
#             'x': torch.tensor([0, data.num_nodes]),
#             'edge_index': torch.tensor([0, data.edge_index.shape[1]]),
#             'y': torch.tensor([0, data.num_nodes]),
#             'node_mask': torch.tensor([0, data.num_nodes]),
#             'edge_mask': torch.tensor([0, data.edge_index.shape[1]]),
#         }
#         collated_data = (data, slices)
#     else:
#         # For multiple graphs, use PyG's collate function
#         from torch_geometric.data import InMemoryDataset
#         collated_data = InMemoryDataset.collate(data_list)
    
#     # Save in the exact format expected by SynGraphDataset
#     save_path = os.path.join(processed_dir, 'data.pt')
#     torch.save(collated_data, save_path)
#     print(f"✅ Saved dataset to: {save_path}")
#     print(f"   Format: {type(collated_data)} with length {len(collated_data)}")
    
#     # Save pre_filter and pre_transform as required
#     torch.save("lambda data: True", os.path.join(processed_dir, 'pre_filter.pt'))
#     torch.save("None", os.path.join(processed_dir, 'pre_transform.pt'))
#     print(f"✅ Saved pre_filter.pt and pre_transform.pt")
    
#     # Verify the saved file
#     print("\n🔍 Verifying saved file...")
#     try:
#         loaded_data, loaded_slices = torch.load(save_path, weights_only=False)
#         print(f"✅ Successfully loaded as (data, slices)")
#         print(f"   Data type: {type(loaded_data)}")
#         print(f"   Slices keys: {loaded_slices.keys()}")
#         print(f"   Number of nodes: {loaded_data.num_nodes}")
#         print(f"   Edge index shape: {loaded_data.edge_index.shape}")
#     except Exception as e:
#         print(f"❌ Verification failed: {e}")
#         # Try loading without unpacking
#         loaded = torch.load(save_path, weights_only=False)
#         print(f"   Actual format: {type(loaded)}")
#         if isinstance(loaded, tuple):
#             print(f"   Tuple length: {len(loaded)}")
    
#     return collated_data

# if __name__ == "__main__":
#     create_syngraph_dataset_compatible()



# import torch
# import os
# from torch_geometric.data import Data

# def convert_old_data(file_path):
#     """Convert old PyG data format to current version"""
#     try:
#         # Load with weights_only=False for older formats
#         old_data = torch.load(file_path, weights_only=False)
        
#         print(f"\n{'='*60}")
#         print(f"Loading file: {file_path}")
#         print(f"{'='*60}")
#         print(f"Data type: {type(old_data)}")
        
#         # Better inspection of the loaded data
#         if isinstance(old_data, list):
#             print(f"Data is a LIST with {len(old_data)} items")
#             if len(old_data) > 0:
#                 print(f"First item type: {type(old_data[0])}")
#                 if hasattr(old_data[0], 'keys'):
#                     print(f"First item keys: {old_data[0].keys}")
#         elif hasattr(old_data, 'keys'):
#             print(f"Data keys: {old_data.keys}")
#         else:
#             print(f"Data dir: {dir(old_data)[:20]}...")  # Show first 20 attributes
        
#         # Try to access actual data
#         if isinstance(old_data, list):
#             # Handle list of graphs
#             new_data_list = []
#             for i, item in enumerate(old_data[:3]):  # Check first 3 items
#                 print(f"\nItem {i}:")
#                 if hasattr(item, 'x'):
#                     print(f"  x shape: {item.x.shape if item.x is not None else 'None'}")
#                 if hasattr(item, 'edge_index'):
#                     print(f"  edge_index shape: {item.edge_index.shape if item.edge_index is not None else 'None'}")
#                 if hasattr(item, 'y'):
#                     print(f"  y: {item.y}")
                
#                 # Convert each item to proper Data object
#                 if not isinstance(item, Data):
#                     new_item = Data()
#                     for key in item.keys:
#                         setattr(new_item, key, item[key])
#                     new_data_list.append(new_item)
#                 else:
#                     new_data_list.append(item)
            
#             # Save the converted list
#             output_path = file_path + '.converted'
#             torch.save(new_data_list, output_path)
#             print(f"\n✓ Converted list of {len(new_data_list)} graphs -> {output_path}")
            
#         elif hasattr(old_data, 'keys'):
#             # Handle single Data object
#             new_data = Data()
#             for key in old_data.keys:
#                 setattr(new_data, key, old_data[key])
            
#             output_path = file_path + '.converted'
#             torch.save(new_data, output_path)
#             print(f"\n✓ Converted single graph -> {output_path}")
#         else:
#             print(f"Unknown data format: {type(old_data)}")
            
#     except Exception as e:
#         print(f"✗ Error processing {file_path}: {e}")
#         import traceback
#         traceback.print_exc()

# # Convert all .pt files
# processed_dir = '/home/pcshark/SubgraphX/datasets/BA_shapes/processed/'

# if os.path.exists(processed_dir):
#     pt_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
#     print(f"Found {len(pt_files)} .pt files to convert")
    
#     for filename in pt_files:
#         full_path = os.path.join(processed_dir, filename)
#         convert_old_data(full_path)
# else:
#     print(f"Directory not found: {processed_dir}")


# import rdkit
# print(rdkit.__version__)
# import numpy
# print(numpy.__version__)





# import torch
# import torch_geometric

# print(f"PyTorch version: {torch.__version__}")
# print(f"PyG version: {torch_geometric.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")  # Should be False for CPU-only


# import torch
# # Check if CUDA is NOT available (this should be False for CPU-only)
# print(f"CUDA Available: {torch.cuda.is_available()}")

# # Try importing a PyG module and check a tensor's device
# from torch_scatter import scatter_sum
# import torch

# x = torch.tensor([[1], [2], [3], [4]])
# index = torch.tensor([0, 0, 1, 2])
# out = scatter_sum(x, index, dim=0)
# print(f"Result device: {out.device}") # Should be 'cpu'


# import os

# # Check if the file exists
# dataset_path = '/home/pcshark/SubgraphX/checkpoints/BA_Shapes/gcn_best.pth'
# if os.path.exists(dataset_path):
#     print(f"File exists at {dataset_path}")
# else:
#     print(f"File not found at {dataset_path}")
#     print(f"Current working directory: {os.getcwd()}")




# import os

# # Check if the file exists
# dataset_path = '../datasets/BA_shapes/raw/BA_shapes.pkl'
# if os.path.exists(dataset_path):
#     print(f"File exists at {dataset_path}")
# else:
#     print(f"File not found at {dataset_path}")
#     print(f"Current working directory: {os.getcwd()}")
    
#     # # Create directories if they don't exist
#     # os.makedirs('../datasets/BA_shapes/raw', exist_ok=True)
#     # print(f"Created directories: ../datasets/BA_shapes/raw/")




# import os
# import glob
# from pathlib import Path

# def locate_ba_shapes_pkl():
#     """
#     Comprehensive function to locate BA_shapes.pkl and return its directory
#     """
    
#     def check_path(path):
#         """Check if file exists at given path"""
#         full_path = os.path.join(path, 'BA_shapes.pkl')
#         if os.path.exists(full_path):
#             return path
#         return None
    
#     # Strategy 1: Check current and parent directories
#     current = Path.cwd()
#     for i in range(5):  # Check up to 5 levels up
#         test_path = current / 'datasets' / 'BA_shapes' / 'raw'
#         result = check_path(str(test_path))
#         if result:
#             return result
        
#         test_path = current / 'BA_shapes' / 'raw'
#         result = check_path(str(test_path))
#         if result:
#             return result
        
#         current = current.parent
    
#     # Strategy 2: Use glob to search
#     search_patterns = [
#         '**/BA_shapes/raw/BA_shapes.pkl',
#         '**/BA_shapes.pkl',
#         '../**/BA_shapes/raw/BA_shapes.pkl',
#         '~/datasets/**/BA_shapes.pkl',
#     ]
    
#     for pattern in search_patterns:
#         matches = glob.glob(os.path.expanduser(pattern), recursive=True)
#         if matches:
#             file_path = matches[0]
#             directory = os.path.dirname(file_path)
#             print(f"Found via glob: {file_path}")
#             return directory
    
#     # Strategy 3: Check torch_geometric cache
#     torch_geo_paths = [
#         os.path.expanduser('~/.cache/torch_geometric/datasets/BA_shapes/raw'),
#         os.path.expanduser('~/torch_geometric_data/BA_shapes/raw'),
#         os.path.expanduser('~/data/torch_geometric/BA_shapes/raw'),
#     ]
    
#     for path in torch_geo_paths:
#         result = check_path(path)
#         if result:
#             return result
    
#     return None

# # Usage
# dir_path = locate_ba_shapes_pkl()
# if dir_path:
#     print(f"Found! Directory: {dir_path}")
#     print(f"Full file path: {os.path.join(dir_path, 'BA_shapes.pkl')}")
# else:
#     print("BA_shapes.pkl not found")
    
#     # Option to generate the dataset
#     response = input("Would you like to generate the dataset? (y/n): ")
#     if response.lower() == 'y':
#         from torch_geometric.datasets import BA_shapes
#         import pickle
        
#         # Create in current directory
#         save_dir = './datasets/BA_shapes/raw'
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Download dataset
#         dataset = BA_shapes(root='./datasets/BA_shapes')
        
#         # Convert and save
#         data_list = []
#         for i in range(len(dataset)):
#             data = dataset[i]
#             data_list.append({
#                 'x': data.x.numpy(),
#                 'edge_index': data.edge_index.numpy(),
#                 'y': data.y.numpy(),
#                 'node_label': data.y.numpy()
#             })
        
#         save_path = os.path.join(save_dir, 'BA_shapes.pkl')
#         with open(save_path, 'wb') as f:
#             pickle.dump(data_list, f)
        
#         print(f"Dataset generated and saved to {save_path}")
#         dir_path = save_dir

# print(f"\nFinal directory path: {dir_path}")