import torch 
from torch_geometric.nn import GINConv
import torch.nn.functional as F


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(hidden_channels, hidden_channels))
        self.convs.append(GINConv(hidden_channels, out_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x