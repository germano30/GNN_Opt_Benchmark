import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
	"""Relational GCN with optional batchnorm and residuals.

	Simple and complete building block compatible with the project's `GCN`/`GAT` style.
	"""

	def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
				 dropout, num_relations, num_bases=None, use_bn=False, residual=False):
		super().__init__()

		assert num_layers >= 2, 'num_layers must be >= 2'

		self.num_layers = num_layers
		self.dropout = float(dropout)
		self.residual = residual

		self.convs = nn.ModuleList()
		self.bns = nn.ModuleList() if use_bn else None

		# input layer
		self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases))
		if use_bn:
			self.bns.append(nn.BatchNorm1d(hidden_channels))

		# hidden layers
		for _ in range(num_layers - 2):
			self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))
			if use_bn:
				self.bns.append(nn.BatchNorm1d(hidden_channels))

		# output layer
		self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations, num_bases=num_bases))

		self.reset_parameters()

	def reset_parameters(self):
		for conv in self.convs:
			if hasattr(conv, 'reset_parameters'):
				conv.reset_parameters()
		if self.bns is not None:
			for bn in self.bns:
				bn.reset_parameters()

	def forward(self, x, edge_index, edge_type):
		# x: [N, F]
		h = x
		for i, conv in enumerate(self.convs[:-1]):
			out = conv(h, edge_index, edge_type)
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


