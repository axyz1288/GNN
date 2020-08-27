import torch.nn as nn
import torch
from .layer import GCNConv

class GCN(nn.Module):
    def __init__(self, nemb, nhid, nclass, dropout=0.3, bias=True):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nemb, nhid)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GCNConv(nhid, nclass)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x, adj)
        return x