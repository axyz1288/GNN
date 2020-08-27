import torch
import torch.nn as nn

class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features, bias)
        self.norm = nn.LayerNorm(out_features)
        self.apply(self.weights_init)
        
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias)

    def forward(self, x, adj):
        x = self.weight(x)
        x = self.norm(x)
        x = torch.matmul(adj, x)
        return x