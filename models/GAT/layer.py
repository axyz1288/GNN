import torch
import torch.nn as nn

class GATConv(nn.Module):
    def __init__(self, nemb, nhid, dropout=0.3, bias=True):
        super(GATConv, self).__init__()
        self.nemb = nemb
        self.nhid = nhid

        self.w = nn.Linear(nemb, nhid, bias)
        self.a = nn.Linear(2 * nhid, 1, bias)
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.Softmax(dim=2)
        self.act3 = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.w.weight)
        torch.nn.init.xavier_uniform_(self.a.weight)

    def forward(self, x, adj):
        h = self.w(x)
        
        # create adjacent matrix
        n_node, n_feature = (h.shape[1], h.shape[2])
        a = h.repeat(1, n_node, 1).view(-1, n_node, n_node, n_feature)
        a = torch.cat([a.transpose(1, 2), a], dim=-1)
        
        # replace nonzero to e    
        e = self.act1(self.a(a)).squeeze(-1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = zero_vec.masked_scatter_(adj > 0, e)
        
        # attention x h
        attention = self.act2(attention)
        attention = self.dropout(attention)
        out = self.act3(torch.matmul(attention, h))
        return out
