import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from sklearn.decomposition import PCA

def NTSC(edge_index, args):
    degrees = calculate_degrees(edge_index)
    sum_degrees = to_dense_adj(edge_index)[0].fill_diagonal_(0).int() @ degrees.int()
    sum_degrees_log_norm = torch.log1p(sum_degrees).float()  
    max_val = sum_degrees_log_norm.max()
    final_weights = max_val - sum_degrees_log_norm  
    return final_weights   

def LGTC(x, edge_index, n_components = 8):
    pca = PCA(n_components=8)
    x = pca.fit_transform(x)
    _, neighbor_averages = sim_average(torch.tensor(x), edge_index)
    sim_all_x = similarity(torch.tensor(x))
    sim_all_x = sim_all_x.mean(dim = 1)
    diff_neighbor_and_all = (neighbor_averages - sim_all_x + 1) / 2  # Min-Max Normalization
    max_val = diff_neighbor_and_all.max()
    final_weights = max_val - diff_neighbor_and_all  
    return final_weights 
 
def calculate_degrees(edge_index):
    node_count = edge_index.max() + 1
    degree = torch.zeros(node_count, dtype=torch.long)
    for node in range(node_count):
        degree[node] = (edge_index[0] == node).sum() + (edge_index[1] == node).sum()
    return degree

def similarity(x):
    return F.normalize(x) @ F.normalize(x).T

def sim_average(feature, edge_index, transform = False):
    sim_x = similarity(feature)
    if transform == True:
        sim_neighbor_x = sim_x * edge_index
    else:
        sim_neighbor_x = sim_x * to_dense_adj(edge_index)[0]
    
    neighbor_averages = torch.zeros(sim_x.size(0))
    for i in range(sim_x.size(0)):
        temp_x = sim_neighbor_x[i]
        non_zero = temp_x[temp_x.nonzero(as_tuple=True)]
        if len(non_zero) > 0:
            neighbor_averages[i] = non_zero.mean()
    return sim_neighbor_x, neighbor_averages
