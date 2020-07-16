import torch.nn as nn
import torch
from .layer import GCNConv

class GCN(nn.Module):
    def __init__(self, ninp, nfeat, nhid, nclass, dropout=0.3, bias=True):
        super(GCN, self).__init__()
        # self.emb = nn.Linear(ninp, ninp)
        # self.act2 = nn.ReLU()
        self.conv1 = GCNConv(nfeat, nhid)
        self.act1 = nn.ReLU()
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        self.conv2 = GCNConv(nhid, nclass)

    def forward(self, x, adj):
        # adj_ = self.act2(self.emb(adj))
        x = self.conv1(x, adj)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x, adj)
        return x