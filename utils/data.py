from torch_geometric.datasets import Planetoid
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator as LinkEvaluator
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator as NodeEvaluator
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator as GraphEvaluator

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
