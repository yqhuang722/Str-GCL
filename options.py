import argparse

def parse_commom_args(parser):
    parser.add_argument('--dataset', type = str, default = 'Cora')
    parser.add_argument('--gpu_id', type = int, default = 0)
    parser.add_argument('--path', type = str, default = './datasets')
    parser.add_argument('--log_dir', type = str, default = './log/Cora.txt')
    parser.add_argument('--seed', type = int, default = 39788)
    return parser

def parser_train_args(parser):
    parser.add_argument('--tau', type = float, default = 0.5, help = "Temperature for InfoNCE-Base.")
    parser.add_argument('--tau_rule', type = float, default = 0.7, help = "Temperature for InfoNCE-Rule.")
    parser.add_argument('--num_epochs', type = int, default = 500, help = "Number of epochs to train the model.")
    parser.add_argument('--num_hidden', type = int, default = 256, help = "Dimension of the embedding layer in the GCN/MLP/Proj.")
    parser.add_argument('--num_proj_hidden', type = int, default = 256, help = "Dimension of the hidden layer in the GCN/Proj.")
    parser.add_argument('--mlp_hidden_dim', type = int, default = 256, help = "Dimension of the hidden layer in the MLP.")  
    parser.add_argument('--learning_rate', type = float, default = 0.0005, help = "Learning rate for the optimizer.")
    parser.add_argument('--weight_decay', type = float, default = 0.00001, help = "Weight decay for L2 regularization.")
    parser.add_argument('--activation', type = str, default = 'relu', help = "Activation function used in the network layers.")
    
    parser.add_argument('--base_model', type = str, default = 'GCNConv', help = "Base GNN architecture.")
    parser.add_argument('--model_type', type = str, default = 'MLPGCN', help = "Str-GCL architecture.")
    parser.add_argument('--num_layers', type = int, default = 1, help = "Number of layers in the GNN model.")

    parser.add_argument('--drop_edge_rate_1', type = float, default = 0.2, help = "Drop edge rate for the first view in augmentation.")
    parser.add_argument('--drop_edge_rate_2', type = float, default = 0.4, help = "Drop edge rate for the second view in augmentation.")
    parser.add_argument('--drop_feature_rate_1', type = float, default = 0.3, help = "Drop feature rate for the first view in augmentation.")
    parser.add_argument('--drop_feature_rate_2', type = float, default = 0.4, help = "Drop feature rate for the second view in augmentation.")
    parser.add_argument('--n_components', type = int, default = 64, help = "Dimension of the PCA processing.")
    parser.add_argument('--loss_base', type = int, default = 1, help = "Weight of the Base-InfoNCE loss.") 
    parser.add_argument('--loss_rule', type = int, default = 1, help = "Weight of the rule loss.") 
    parser.add_argument('--loss_cross', type = int, default = 1, help = "Weight of the cross loss.") 
    return parser

def parser_data_args(parser):
    parser.add_argument('--input_dim', type = int, default = 0, help = "Dimension of the input features.")
    parser.add_argument('--recon_dim', type = int, default = 0, help = "Dimension of the features after PCA processing.")
    return parser

def add_data_args(args, data):
    args.input_dim = data.raw.feature.size(1)
    args.recon_dim = data.raw.feature.size(1)
    return args

def prepare_args():
    parser = argparse.ArgumentParser()
    parser = parse_commom_args(parser)
    parser = parser_train_args(parser)
    args = parser.parse_args()
    return args
