import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.base_model = ({'GCNConv': GCNConv})[args.base_model]
        self.k = args.num_layers
        self.conv = [self.base_model(args.input_dim, args.num_hidden)]
        for _ in range(self.k - 1):
            self.conv.append(self.base_model(args.num_hidden, args.num_hidden))
        self.conv = nn.ModuleList(self.conv)

        self.activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[args.activation]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x
    
class Projection(nn.Module):
    def __init__(self, args):
        super(Projection, self).__init__()
        self.fc1 = torch.nn.Linear(args.num_hidden, args.num_proj_hidden)
        self.fc2 = torch.nn.Linear(args.num_proj_hidden, args.num_hidden)
        
    def forward(self, z: torch.Tensor):
        z = F.elu(self.fc1(z))
        return self.fc2(z)