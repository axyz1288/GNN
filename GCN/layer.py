import torch
import torch.nn as nn

class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = None
        if bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.init_weights()
    
    def init_weights(self):
        y = 1.0 / (self.out_features ** 0.5)
        self.weight.data.uniform_(-y, y)
        if self.bias is not None:
            self.bias.data.uniform_(-y, y)

    def forward(self, x, adj):
        x = torch.mm(x, self.weight)
        if self.bias is not None:
            x += self.bias
        x = torch.spmm(adj, x)
        return x