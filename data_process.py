import torch
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor, WikiCS, ppi
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork
from torch_geometric.utils import dropout_edge
from utils import drop_feature
from rules import NTSC, LGTC

class GraphData():
    def __init__(self, x, edge_index):
        self.x = x
        self.edge_index = edge_index

class View():
    def __init__(self, feature, edge_index):
        self.feature = feature
        self.edge_index = edge_index
        
    def to(self, device):
        self.feature = self.feature.to(device)
        self.edge_index = self.edge_index.to(device)
        return self
        
class DataRepo():
    def __init__(self, view_0, view_1, view_2, y, alpha, beta):
        self.raw = view_0
        self.aug_1 = view_1
        self.aug_2 = view_2
        self.y = y
        self.alpha = alpha
        self.beta = beta
        
    def to(self, device):
        self.raw = self.raw.to(device)
        self.aug_1 = self.aug_1.to(device)
        self.aug_2 = self.aug_2.to(device)
        self.y = self.y.to(device)
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)

def data_loader(args):
    datasets = load_dataset(args.dataset, args.path)
    data = datasets[0]

    alpha = NTSC(data.edge_index, args)
    beta = LGTC(data.x, data.edge_index, args.n_components)
    
    x_1 = drop_feature(data.x, args.drop_feature_rate_1)
    x_2 = drop_feature(data.x, args.drop_feature_rate_2)
    edge_index_1 = dropout_edge(data.edge_index, p = args.drop_edge_rate_1)[0]
    edge_index_2 = dropout_edge(data.edge_index, p = args.drop_edge_rate_2)[0]
    
    raw_data = View(data.x, data.edge_index)
    aug_data_1 = View(x_1, edge_index_1)
    aug_data_2 = View(x_2, edge_index_2)
    
    data_repo = DataRepo(raw_data, aug_data_1, aug_data_2, data.y, alpha, beta)
    return data_repo

def update(data: DataRepo, args):
    data.aug_1.feature = drop_feature(data.raw.feature, args.drop_feature_rate_1)
    data.aug_2.feature = drop_feature(data.raw.feature, args.drop_feature_rate_2)
    data.aug_1.edge_index = dropout_edge(data.raw.edge_index, p = args.drop_edge_rate_1)[0]
    data.aug_2.edge_index = dropout_edge(data.raw.edge_index, p = args.drop_edge_rate_2)[0]
    return data

def load_dataset(dataset_name, dataset_dir):

    print('Dataloader: Loading Dataset', dataset_name)
    assert dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'dblp', 'Photo','Computers', 'CS','Physics', 
                'ogbn-products', 'ogbn-arxiv', 'Wiki','ppi',
                'Cornell', 'Texas', 'Wisconsin',
                'chameleon', 'crocodile', 'squirrel']
    
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name, 
                transform=T.NormalizeFeatures())
        
    elif dataset_name == 'dblp':
        dataset = CitationFull(dataset_dir, name=dataset_name, 
                transform=T.NormalizeFeatures())
        
    elif dataset_name in ['Photo','Computers']:
        dataset = Amazon(dataset_dir, name=dataset_name, 
                transform=T.NormalizeFeatures())
        
    elif dataset_name in ['CS','Physics']:
        dataset = Coauthor(dataset_dir, name=dataset_name, 
                transform=T.NormalizeFeatures())
        
    elif dataset_name in ['Wiki']:
        dataset = WikiCS(dataset_dir,
                transform=T.NormalizeFeatures())
    elif dataset_name in ['ppi']:
        train = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'train')
        val = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'val')
        test = ppi.PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'test')
        dataset = [train, val, test]   
        
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
            return WebKB(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())
    elif dataset_name in ['chameleon', 'crocodile', 'squirrel']:
            return WikipediaNetwork(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())
    
    print('Dataloader: Loading success.')
    print(dataset[0])
    
    return dataset