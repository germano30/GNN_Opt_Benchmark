import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        def make_mlp(in_ch, out_ch):
            return nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Linear(out_ch, out_ch),
            )

        # First layer
        self.convs.append(GINConv(make_mlp(in_channels, hidden_channels)))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(make_mlp(hidden_channels, hidden_channels)))

        # Last layer
        self.convs.append(GINConv(make_mlp(hidden_channels, out_channels)))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x