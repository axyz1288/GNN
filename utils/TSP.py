import numpy as np
import torch
from torch.utils.data import Dataset

class TSP(Dataset):
    def __init__(self, num_nodes, file_name):
        self.num_nodes = num_nodes
        self.coordinates = np.genfromtxt('./data/tsp/'+ file_name + '_coordinate.txt')
        self.tours = np.genfromtxt('./data/tsp/' + file_name + '_tour.txt')
        
    def __len__(self):
        return self.tours.shape[0]
        
    def __getitem__(self, idx):
        features = torch.tensor(self.coordinates[idx * self.num_nodes:(idx + 1) * self.num_nodes])
        tour = torch.tensor(self.tours[idx])
        labels = torch.cat([tour[1::], tour[0].unsqueeze(0)])
        idx = torch.cat([tour.unsqueeze(0), labels.unsqueeze(0)])
        adj = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1])).to_dense()
        return adj, labels, features
