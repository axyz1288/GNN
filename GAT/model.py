import torch.nn as nn
from .layer import GATConv

class GAT(nn.Module):
    def __init__(self, nemb, nhid, nnode, nclass, dropout=0.3):
        super(GAT, self).__init__()
        self.conv1 = GATConv(nemb, nhid, nnode)
        self.act1 = nn.ELU()
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        self.conv2 = GATConv(nhid, nclass, nnode)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x, adj)
        return x