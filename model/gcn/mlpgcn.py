import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodules import GCN, Projection
    
class MLPGCN_Model(torch.nn.Module):
    def __init__(self, args, alpha, beta):
        super(MLPGCN_Model, self).__init__()
        self.encoder = GCN(args)
        self.tau: float = args.tau
        self.projector = Projection(args)
        
        self.fc1 = torch.nn.Linear(args.recon_dim, args.mlp_hidden_dim)
        self.fc2 = torch.nn.Linear(args.mlp_hidden_dim, args.num_hidden)

        self.alpha = torch.nn.Parameter(alpha.float())
        self.beta = torch.nn.Parameter(beta.float())

        self.mlp = nn.Sequential(
            nn.Linear(2, args.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.mlp_hidden_dim, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
    def forward(self, view) -> torch.Tensor:
        z1 = self.encoder(view.feature, view.edge_index)
        z2 = self.fc2(F.elu(self.fc1(view.feature)))

        weights = self.mlp(torch.stack([self.alpha, self.beta], dim = 1))

        z2 = z2 * weights
        
        return z1, z2
    
    
    
    
    
    
    
    
    
    
    
    
    