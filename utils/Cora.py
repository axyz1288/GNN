import numpy as np
import torch

def encode_onehot(classes, x):
    class_dict = dict(zip(classes, range(len(classes)))) 
    return torch.tensor([class_dict[i] for i in x], dtype=torch.long)

def norm(x):
    rowsum = x.sum(dim=1)
    r_inv = rowsum.pow(-1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    x = torch.mm(r_mat_inv.transpose(0, 1), x)
    return x

def load_data():
    data = np.genfromtxt('./data/cora/cora.content', dtype=np.dtype(str))
    id = data[:, 0].astype(np.int32)
    id_dict = dict(zip(id, range(id.shape[0])))
    
    # features of node
    features = torch.tensor(data[:, 1:-1].astype(np.float32))
    
    # classification of node
    labels = data[:, -1]
    labels = encode_onehot(np.unique(labels), labels)
    
    # edges
    edges = np.genfromtxt('./data/cora/cora.cites', dtype=np.int)
    edges = np.array([[id_dict[i] for i in j] for j in edges], dtype=np.long)
    edges = torch.tensor(edges).permute(1, 0)
    adj = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), 
                                size=(labels.shape[0], labels.shape[0]), 
                                dtype=torch.float)
    adj = torch.eye(adj.shape[0]) + adj + adj.t()

    features = norm(features).unsqueeze(0)
    adj = norm(adj).unsqueeze(0)
    labels = labels.unsqueeze(0)
    
    idx_train = torch.LongTensor(range(140))
    idx_val = torch.LongTensor(range(200, 500))
    idx_test = torch.LongTensor(range(500, 1500))
    return adj, features, labels, idx_train, idx_val, idx_test
