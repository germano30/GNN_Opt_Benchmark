import argparse
import time
import torch

_original_load = torch.load

def custom_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)

torch.load = custom_load

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated"
)

from models.gat import GAT
from models.gcn import GCN
from models.rgat import RGAT
from models.rgcn import RGCN
from models.gin import GIN

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator as LinkEvaluator
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator as NodeEvaluator
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator as GraphEvaluator

from muon import MuonWithAuxAdam
import torch.distributed as dist
dist.get_world_size = lambda *args, **kwargs: 1
dist.get_rank = lambda *args, **kwargs: 0
dist.all_gather = lambda *args, **kwargs: None
try:
    from preconditioned_stochastic_gradient_descent import PSGD # usually SOAP is implemented here or via another package
except ImportError:
    pass

try:
    import torch_optimizer as optim
except ImportError:
    pass

from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage

try:
    torch.serialization.safe_globals([
        Data,
        DataEdgeAttr,
        DataTensorAttr,
        GlobalStorage,
        NodeStorage,
        EdgeStorage
    ])
except AttributeError:
    pass

import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class GraphPredictor(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout):
        super(GraphPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class NodePredictor(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout):
        super(NodePredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

# Placeholder for optimizers
def get_optimizer(name, params, lr):
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr)
    elif name == 'SGD':
        return torch.optim.SGD(params, lr=lr)
    elif name == 'Muon':            
        import math
        hidden_weights = [p for p in params if p.ndim >= 2]
        hidden_gains_biases = [p for p in params if p.ndim < 2]

        param_groups = []
        
        # Simula o comportamento do adjust_lr_fn="match_rms_adamw" da implementacao nativa
        # O PyTorch calcula: ratio = 0.2 * sqrt(max(A, B))
        for p in hidden_weights:
            A, B = p.shape[:2]
            adjusted_ratio = 0.2 * math.sqrt(max(A, B))
            
            param_groups.append(dict(
                params=[p],
                use_muon=True,
                lr=lr * adjusted_ratio,
                weight_decay=5e-4 / adjusted_ratio
            ))
            
        param_groups.append(
            dict(
                params=hidden_gains_biases,
                use_muon=False,
                lr=lr, # AdamW passa a usar a mesma taxa do Muon
                betas=(0.9, 0.95),
                weight_decay=5e-4
            )
        )
        return MuonWithAuxAdam(param_groups)
    elif name == 'Shampoo':
        try:
            import torch_optimizer as optim
            return optim.Shampoo(params, lr=lr)
        except ImportError:
            raise ImportError("Please install torch_optimizer for Shampoo: pip install torch-optimizer")
    elif name == 'SOAP':
        try:
            from soap import SOAP
            return SOAP(params, lr=lr)
        except ImportError:
            # Another common source is via optimizer package or bitsandbytes, we'll assume a local 'soap.py' or package 'soap'
            raise ImportError("Please ensure SOAP optimizer is available (e.g. from soap import SOAP)")
    else:
        raise ValueError(f"Unknown optimizer {name}")

# Dataset info
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
OPTIMIZERS = ['AdamW', 'SGD', 'Muon']

def load_dataset(name):
    if name == 'Cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
        split_idx = {'train': data.train_mask, 'valid': data.val_mask, 'test': data.test_mask}
        evaluator = None
        return data, split_idx, evaluator, 'node'
    elif DATASETS[name]['type'] == 'link':
        dataset = PygLinkPropPredDataset(name=name)
        evaluator = LinkEvaluator(name=name)
        data = dataset[0]
        split_idx = dataset.get_edge_split()
        return data, split_idx, evaluator, 'link'
    elif DATASETS[name]['type'] == 'node':
        dataset = PygNodePropPredDataset(name=name)
        evaluator = NodeEvaluator(name=name)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        return data, split_idx, evaluator, 'node'
    elif DATASETS[name]['type'] == 'graph':
        dataset = PygGraphPropPredDataset(name=name)
        evaluator = GraphEvaluator(name=name)
        split_idx = dataset.get_idx_split()
        return dataset, split_idx, evaluator, 'graph'

def get_model(name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, num_relations=None):
    if graph_type == 'homogeneous':
        if name == 'GCN':
            return GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        elif name == 'GAT':
            return GAT(in_channels, hidden_channels, out_channels, num_layers, dropout)
        elif name == 'GIN':
            return GIN(in_channels, hidden_channels, out_channels, num_layers, dropout)
    else:
        if name == 'RGCN':
            return RGCN(in_channels, hidden_channels, out_channels, num_layers, dropout, num_relations)
        elif name == 'RGAT':
            return RGAT(in_channels, hidden_channels, out_channels, num_layers, dropout, num_relations)

def train_link_prediction(gnn, predictor, data, split_edge, optimizer, device, graph_type, batch_size=1024):
    gnn.train()
    predictor.train()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if graph_type == 'heterogeneous':
        edge_type = data.edge_attr.to(device)
    else:
        edge_type = None
    
    pos_train_edge = split_edge['train']['edge']
    
    if 'edge_neg' in split_edge['train']:
        neg_train_edge = split_edge['train']['edge_neg']
    else:
        # Dynamically generate negative edges
        neg_edge_index = negative_sampling(
            edge_index=pos_train_edge.t(), # (2, num_edges)
            num_nodes=data.num_nodes,
            num_neg_samples=pos_train_edge.size(0)
        )
        neg_train_edge = neg_edge_index.t()
    
    all_edges = torch.cat([pos_train_edge, neg_train_edge], dim=0)
    labels = torch.cat([torch.ones(pos_train_edge.size(0)), torch.zeros(neg_train_edge.size(0))], dim=0)
    
    dataset = torch.utils.data.TensorDataset(all_edges, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_loss = 0.0
    for batch_edges, batch_labels in dataloader:
        optimizer.zero_grad()
        batch_edges = batch_edges.to(device)
        batch_labels = batch_labels.to(device)
        
        if graph_type == 'heterogeneous':
            h = gnn(x, edge_index, edge_type)
        else:
            h = gnn(x, edge_index)
        
        out = predictor(h[batch_edges[:, 0]], h[batch_edges[:, 1]])
        loss = F.binary_cross_entropy(out.squeeze(-1), batch_labels.float())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def eval_link_prediction(gnn, predictor, data, split_edge, evaluator, device, graph_type):
    gnn.eval()
    predictor.eval()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if graph_type == 'heterogeneous':
        edge_type = data.edge_attr.to(device)
        h = gnn(x, edge_index, edge_type)
    else:
        h = gnn(x, edge_index)
    
    pos_test_edge = split_edge['test']['edge'].to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(device)
    
    pos_out = predictor(h[pos_test_edge[:, 0]], h[pos_test_edge[:, 1]]).squeeze(-1)
    neg_out = predictor(h[neg_test_edge[:, 0]], h[neg_test_edge[:, 1]]).squeeze(-1)
    
    result = evaluator.eval({'y_pred_pos': pos_out, 'y_pred_neg': neg_out})
    if 'hits@50' in result:
        return result['hits@50']
    elif 'hits@20' in result:
        return result['hits@20']
    elif 'mrr_list' in result:
        return result['mrr_list'].mean().item()
    else:
        return list(result.values())[0]

def train_node_classification(gnn, predictor, data, split_idx, optimizer, device, graph_type, batch_size=1024):
    gnn.train()
    predictor.train()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    
    if graph_type == 'heterogeneous':
        edge_type = data.edge_attr.to(device)
    else:
        edge_type = None
        
    train_indices = split_idx['train']
    if train_indices.dtype == torch.bool:
        train_indices = train_indices.nonzero(as_tuple=True)[0]
        
    dataset = torch.utils.data.TensorDataset(train_indices, y[train_indices])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_loss = 0.0
    for batch_indices, batch_labels in dataloader:
        optimizer.zero_grad()
        batch_indices = batch_indices.to(device)
        batch_labels = batch_labels.to(device)
        
        if graph_type == 'heterogeneous':
            h = gnn(x, edge_index, edge_type)
        else:
            h = gnn(x, edge_index)
            
        batch_out = predictor(h[batch_indices])
        if batch_labels.dim() > 1 and batch_labels.shape[-1] > 1:
            loss = F.binary_cross_entropy_with_logits(batch_out, batch_labels.float())
        else:
            loss = F.cross_entropy(batch_out, batch_labels.squeeze().long())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def eval_node_classification(gnn, predictor, data, split_idx, evaluator, device, graph_type):
    gnn.eval()
    predictor.eval()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    
    if graph_type == 'heterogeneous':
        edge_type = data.edge_attr.to(device)
        h = gnn(x, edge_index, edge_type)
    else:
        h = gnn(x, edge_index)
        
    out = predictor(h)
    
    test_indices = split_idx['test']
    if test_indices.dtype == torch.bool:
        test_indices = test_indices.nonzero(as_tuple=True)[0]
        
    if evaluator:
        y_true = y[test_indices]
        y_pred = out[test_indices]
        if y_true.dim() == 1 or y_true.shape[-1] == 1:
            y_pred = y_pred.argmax(dim=-1, keepdim=True)
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(-1)
        result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
        return list(result.values())[0]
    else:
        pred = out[test_indices].argmax(dim=-1)
        y_true = y[test_indices]
        correct = (pred == (y_true.squeeze() if y_true.dim() > 1 else y_true)).sum().item()
        return correct / test_indices.numel()

# Graph classification
def train_graph_classification(gnn, predictor, dataset, split_idx, optimizer, device, graph_type, batch_size=32):
    gnn.train()
    predictor.train()
    
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=batch_size, shuffle=True)
    
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        
        if graph_type == 'heterogeneous':
            h = gnn(batch.x, batch.edge_index, batch.edge_attr)
        else:
            h = gnn(batch.x, batch.edge_index)
            
        h_graph = global_mean_pool(h, batch.batch)
        out = predictor(h_graph)
        
        is_labeled = batch.y == batch.y
        if batch.y.dim() > 1 and batch.y.shape[-1] > 1:
            loss = F.binary_cross_entropy_with_logits(out[is_labeled], batch.y[is_labeled].float())
        else:
            loss = F.cross_entropy(out[is_labeled], batch.y.view(-1)[is_labeled].long())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def eval_graph_classification(gnn, predictor, dataset, split_idx, evaluator, device, graph_type, batch_size=32):
    gnn.eval()
    predictor.eval()
    
    y_true = []
    y_pred = []
    
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            if graph_type == 'heterogeneous':
                h = gnn(batch.x, batch.edge_index, batch.edge_attr)
            else:
                h = gnn(batch.x, batch.edge_index)
                
            h_graph = global_mean_pool(h, batch.batch)
            out = predictor(h_graph)
            
            if batch.y.dim() > 1 and batch.y.shape[-1] > 1:
                y_pred.append(out.cpu())
                y_true.append(batch.y.cpu())
            else:
                y_pred.append(out.argmax(dim=-1).view(-1, 1).cpu())
                y_true.append(batch.y.view(-1, 1).cpu())
            
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    
    if evaluator:
        result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
        if 'acc' in result:
            return result['acc']
        return list(result.values())[0]
    else:
        correct = (y_pred == y_true).sum().item()
        return correct / y_true.size(0)

def run_experiment(dataset_name, model_name, optimizer_name, epochs=10, lr=0.01, hidden_channels=256, num_layers=3, dropout=0.5, batch_size=1024):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_info = load_dataset(dataset_name)
    task = DATASETS[dataset_name]['type']
    graph_type = DATASETS[dataset_name]['graph_type']
    
    if task == 'link':
        data, split_edge, evaluator, _ = data_info
        num_relations = data.edge_attr.max().item() + 1 if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1
        in_channels = data.x.size(1)
        out_channels = hidden_channels
        gnn = get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, num_relations)
        predictor = LinkPredictor(hidden_channels, hidden_channels, 1, 2, dropout)
        gnn.to(device)
        predictor.to(device)
        params = list(gnn.parameters()) + list(predictor.parameters())
    elif task == 'node':
        data, split_idx, evaluator, _ = data_info
        num_relations = data.edge_attr.max().item() + 1 if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1
        in_channels = data.x.size(1)
        out_channels = hidden_channels
        gnn = get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, num_relations)
        num_classes = data.y.max().item() + 1 if dataset_name != 'Cora' else 7  # Cora has 7 classes
        predictor = NodePredictor(hidden_channels, num_classes, 2, dropout)
        gnn.to(device)
        predictor.to(device)
        params = list(gnn.parameters()) + list(predictor.parameters())
    else:
        # Graph
        dataset, split_idx, evaluator, _ = data_info
        in_channels = dataset.num_node_features
        out_channels = hidden_channels
        gnn = get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, 1)
        # OGBG PPA has 37 classes
        num_classes = dataset.num_classes
        predictor = GraphPredictor(hidden_channels, num_classes, 2, dropout)
        gnn.to(device)
        predictor.to(device)
        params = list(gnn.parameters()) + list(predictor.parameters())
    
    optimizer = get_optimizer(optimizer_name, params, lr)
    
    start_time = time.time()
    for epoch in range(epochs):
        if task == 'link':
            loss = train_link_prediction(gnn, predictor, data, split_edge, optimizer, device, graph_type, batch_size)
        elif task == 'node':
            loss = train_node_classification(gnn, predictor, data, split_idx, optimizer, device, graph_type, batch_size)
        elif task == 'graph':
            loss = train_graph_classification(gnn, predictor, dataset, split_idx, optimizer, device, graph_type, batch_size=32)
        print(f'Epoch {epoch}: Loss {loss:.4f}')
    training_time = time.time() - start_time
    
    # Evaluate
    if task == 'link':
        score = eval_link_prediction(gnn, predictor, data, split_edge, evaluator, device, graph_type)
    elif task == 'node':
        score = eval_node_classification(gnn, predictor, data, split_idx, evaluator, device, graph_type)
    elif task == 'graph':
        score = eval_graph_classification(gnn, predictor, dataset, split_idx, evaluator, device, graph_type, batch_size=32)
    
    return score, training_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys())
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--optimizer', type=str, required=True, choices=OPTIMIZERS)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    
    # Check model
    if DATASETS[args.dataset]['graph_type'] == 'homogeneous' and args.model not in MODELS_HOMO:
        raise ValueError("Invalid model for homogeneous graph")
    if DATASETS[args.dataset]['graph_type'] == 'heterogeneous' and args.model not in MODELS_HETERO:
        raise ValueError("Invalid model for heterogeneous graph")
    
    score, time_taken = run_experiment(args.dataset, args.model, args.optimizer, args.epochs, args.lr, batch_size=args.batch_size)
    print(f'Dataset: {args.dataset}, Model: {args.model}, Optimizer: {args.optimizer}, Score: {score:.4f}, Time: {time_taken:.2f}s')