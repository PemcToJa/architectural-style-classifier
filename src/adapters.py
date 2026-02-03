import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, in_features, bottleneck_dim=64, dropout=0.0):
        super().__init__()
        self.down_proj = nn.Linear(in_features, bottleneck_dim)
        self.non_linearity = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, in_features)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.down_proj.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        return x + self.up_proj(self.non_linearity(self.dropout(self.down_proj(x))))