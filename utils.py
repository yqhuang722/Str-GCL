import torch
import torch.nn as nn
import numpy as np
import functools
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from torch_geometric.utils import to_dense_adj

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def recons_feature(x, edge_index):
    x_num = x.size(0)
    rec_feature = torch.zeros(x.size(0), dtype = torch.float).unsqueeze(1)
    rules = [cal_degrees]
    for rule in rules:
        new_feature = rule(x, edge_index)
        rec_feature = torch.cat((rec_feature, new_feature), dim = 1)
    rec_feature = rec_feature[:,1:]    
    return rec_feature

def cal_degrees(x, edge_index):
    degrees = calculate_degrees(edge_index)
    cal_degrees = degrees.clone()
    sum_degrees = to_dense_adj(edge_index)[0].fill_diagonal_(0).int() @ degrees.int()
    degrees[degrees == 0] = 1
    average_sum_degrees = sum_degrees / degrees
    rec_x = torch.concat((cal_degrees.unsqueeze(1), sum_degrees.unsqueeze(1), average_sum_degrees.unsqueeze(1)), 1)
    return rec_x

def calculate_degrees(edge_index):
    node_count = edge_index.max() + 1
    degree = torch.zeros(node_count, dtype=torch.long)
    for node in range(node_count):
        degree[node] = (edge_index[0] == node).sum() + (edge_index[1] == node).sum()
    return degree

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()

@repeat(10)
def label_classification(embeddings, y, ratio):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)
    X = normalize(X, norm='l2')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 - ratio, shuffle=True)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                 param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                 verbose=0)
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }