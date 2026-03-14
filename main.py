import argparse

from models.gat import GAT
from models.gcn import GCN
from models.rgat import RGAT
from models.rgcn import RGCN
from models.gin import GIN
import torch 
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


from muon import MuonWithAuxAdam

def main():
    
    # Load dataset
    if args.dataset == 'ogbl-collab':
        dataset = PygLinkPropPredDataset(name=args.dataset)
        evaluator = Evaluator(name=args.dataset)
        data = dataset[0]
        split_edge = dataset.get_edge_split()
        edge_index = data.edge_index
        edge_type = data.edge_attr.view(-1)  # Assuming edge_attr contains relation types
        x = data.x
        y = data.y
    elif args.dataset == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=args.dataset)
        evaluator = Evaluator(name=args.dataset)
        data = dataset[0]
        edge_index = data.edge_index
        edge_type = data.edge_attr.view(-1)  # Assuming edge_attr contains relation types
        x = data.x
        y = data.y
    elif args.dataset == '
