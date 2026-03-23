import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid, WordNet18RR
from torch_geometric.utils import negative_sampling

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator as LinkEvaluator
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator as NodeEvaluator
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator as GraphEvaluator

class WordNet18RREvaluator:
    """Evaluator that calculates MRR for Knowledge Graph datasets (WordNet18RR)."""
    def eval(self, dict_res):
        import torch
        p = dict_res['y_pred_pos']
        n = dict_res['y_pred_neg']
        if n.dim() == 1:
            n = n.view(p.shape[0], -1)
        p = p.view(-1, 1)
        ranks = (n >= p).sum(dim=-1) + 1.0
        mrr = (1.0 / ranks).mean()
        return {'mrr_list': torch.tensor([mrr])}

DATASETS = {
    'ogbl-collab': {'type': 'link', 'graph_type': 'homogeneous'},
    'ogbn-proteins': {'type': 'node', 'graph_type': 'homogeneous'},
    'ogbg-ppa': {'type': 'graph', 'graph_type': 'homogeneous'},
    'Cora': {'type': 'node', 'graph_type': 'homogeneous'},
    'WordNet18RR': {'type': 'link', 'graph_type': 'heterogeneous'},
    'ogbl-biokg': {'type': 'link', 'graph_type': 'heterogeneous'},
}

MODELS_HOMO = ['GCN', 'GAT', 'GIN']
MODELS_HETERO = ['RGCN', 'RGAT']
OPTIMIZERS = ['AdamW', 'SGD', 'Muon', 'Shampoo', 'SOAP']

def load_dataset(name):
    import os
    import torch
    path = f'dataset/processed_{name}.pt'
    if not os.path.exists(path):
        raise RuntimeError(f"Dataset {name} not prepared. Please run `python prepare_datasets.py --dataset {name}` first.")
    
    data, split_idx, task = torch.load(path, weights_only=False)
    
    if name == 'WordNet18RR':
        evaluator = WordNet18RREvaluator()
    elif task == 'link':
        from ogb.linkproppred import Evaluator as LinkEvaluator
        evaluator = LinkEvaluator(name=name)
    elif task == 'node':
        from ogb.nodeproppred import Evaluator as NodeEvaluator
        evaluator = NodeEvaluator(name=name) if name != 'Cora' else None
    elif task == 'graph':
        from ogb.graphproppred import Evaluator as GraphEvaluator
        evaluator = GraphEvaluator(name=name)
        
    return data, split_idx, evaluator, task
