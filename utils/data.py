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
    if name == 'Cora':
        dataset = Planetoid(root='dataset/Cora', name='Cora')
        data = dataset[0]
        split_idx = {'train': data.train_mask, 'valid': data.val_mask, 'test': data.test_mask}
        evaluator = None
        return data, split_idx, evaluator, 'node'
    elif DATASETS[name]['type'] == 'link':
        if name == 'WordNet18RR':
            dataset = WordNet18RR(root='dataset/WordNet18RR')
            data = dataset[0]
            val_edge_index = data.edge_index[:, data.val_mask]
            test_edge_index = data.edge_index[:, data.test_mask]
            
            val_neg_edge_index = negative_sampling(val_edge_index, num_nodes=data.num_nodes, num_neg_samples=val_edge_index.size(1))
            test_neg_edge_index = negative_sampling(test_edge_index, num_nodes=data.num_nodes, num_neg_samples=test_edge_index.size(1))
            
            split_idx = {
                'train': {'edge': data.edge_index[:, data.train_mask].t()},
                'valid': {'edge': val_edge_index.t(), 'edge_neg': val_neg_edge_index.t()},
                'test': {'edge': test_edge_index.t(), 'edge_neg': test_neg_edge_index.t()}
            }
            evaluator = WordNet18RREvaluator()
            return data, split_idx, evaluator, 'link'
        else:
            dataset = PygLinkPropPredDataset(name=name, root='dataset')
            evaluator = LinkEvaluator(name=name)
            data = dataset[0]
            split_idx = dataset.get_edge_split()
            return data, split_idx, evaluator, 'link'
    elif DATASETS[name]['type'] == 'node':
        dataset = PygNodePropPredDataset(name=name, root='dataset')
        evaluator = NodeEvaluator(name=name)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        return data, split_idx, evaluator, 'node'
    elif DATASETS[name]['type'] == 'graph':
        dataset = PygGraphPropPredDataset(name=name, root='dataset')
        evaluator = GraphEvaluator(name=name)
        split_idx = dataset.get_idx_split()
        return dataset, split_idx, evaluator, 'graph'
