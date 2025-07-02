import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_cross(distribution_a, distribution_b):
    mean_a = torch.mean(distribution_a, dim=0)
    mean_b = torch.mean(distribution_b, dim=0)
    
    cov_a = torch.cov(distribution_a.t())
    cov_b = torch.cov(distribution_b.t())
    
    mean_loss = F.mse_loss(mean_a, mean_b)
    cov_loss = F.mse_loss(cov_a, cov_b)
    
    return mean_loss + cov_loss

def loss_rule(z, tau):
    f = lambda x: torch.exp(x / tau)
    z = F.normalize(z, dim=1)
    sim = f(z @ z.T)
    return -torch.log(
         sim.diag()
         / (sim.sum(1)) ).mean()

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor, args):
    f = lambda x: torch.exp(x / args.tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def loss(h1: torch.Tensor, h2: torch.Tensor, args, 
         mean: bool = True, batch_size: int = 0):

    l1 = semi_loss(h1, h2, args)
    l2 = semi_loss(h2, h1, args)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret