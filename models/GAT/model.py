import torch.nn as nn
from .layer import GATConv

class GAT(nn.Module):
    def __init__(self, nemb, nhid, nclass, dropout=0.3, bias=True):
        super(GAT, self).__init__()
        self.conv1 = GATConv(nemb, nhid, dropout=dropout, bias=bias)
        self.conv2 = GATConv(nhid, nclass, dropout=dropout, bias=bias)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        return x