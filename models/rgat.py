import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv
from torch_scatter import scatter_add


class RGAT(nn.Module):
    """Relational GCN with relation-conditioned additive aggregation and options.

    This keeps the implementation simple but adds common conveniences:
    - optional BatchNorm
    - residual connections
    - relation embeddings projected and aggregated per target node
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_relations, rel_emb_dim=None, num_bases=None,
                 use_bn=False, residual=False):
        super().__init__()

        assert num_layers >= 2
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.residual = residual

        self.rel_emb_dim = rel_emb_dim if rel_emb_dim is not None else hidden_channels
        self.rel_emb = nn.Embedding(num_relations, self.rel_emb_dim)
        self.rel_proj = nn.Linear(self.rel_emb_dim, hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None

        self.convs.append(RGATConv(in_channels, hidden_channels, num_relations, num_bases=num_bases))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(RGATConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(RGATConv(hidden_channels, out_channels, num_relations, num_bases=num_bases))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()
        if hasattr(self.rel_emb, 'reset_parameters'):
            try:
                self.rel_emb.reset_parameters()
            except Exception:
                pass

    def forward(self, x, edge_index, edge_type):
        # x: [N, F]
        N = x.size(0)
        h = x

        for i, conv in enumerate(self.convs[:-1]):
            out = conv(h, edge_index, edge_type)

            # relation-conditioned message: project relation embeddings and
            # aggregate to target nodes (additive conditioning)
            rel_msg = self.rel_emb(edge_type)
            rel_msg = self.rel_proj(rel_msg)  # [E, hidden]
            target = edge_index[1]
            agg = scatter_add(rel_msg, target, dim=0, dim_size=N)

            out = out + agg

            if self.bns is not None:
                out = self.bns[i](out)

            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)

            if self.residual and out.shape == h.shape:
                h = h + out
            else:
                h = out

        h = self.convs[-1](h, edge_index, edge_type)
        return h

