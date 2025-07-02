from model.gcn.mlpgcn import MLPGCN_Model

def select_model(args, alpha, beta):
    selected_model = {
        'MLPGCN': MLPGCN_Model(args, alpha, beta)
    }
    model = selected_model[args.model_type]
    return model