import torch
import torch.nn as nn

class GATConv(nn.Module):
    def __init__(self, nemb, nhid, nnode, dropout=0.3):
        super(GATConv, self).__init__()
        self.nemb = nemb
        self.nhid = nhid
        self.nnode = nnode

        self.w = nn.Linear(nemb, nhid)
        self.a = nn.Linear(2 * nhid, 1)
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.Softmax(dim=1)
        self.act3 = nn.ELU()
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.w.weight)
        torch.nn.init.xavier_uniform_(self.a.weight)

    def forward(self, x, adj):
        h = self.w(x)
        n_node, n_feature = h.shape
        a = torch.cat(
            [h.repeat(1, n_node).view(n_node * n_node, -1), h.repeat(n_node, 1)], 
            dim=1).view(n_node, -1, 2 * n_feature)
        
        # replace nonzero to e    
        e = self.act1(self.a(a)).squeeze(2)
        zero_vec = -9e15*torch.ones_like(e)
        attention = zero_vec.masked_scatter_(adj > 0, e)
        # attention x h
        attention = self.act2(attention)
        if self.dropout != None:
            attention = self.dropout(attention)
        out = self.act3(torch.mm(attention, h))
        return out
