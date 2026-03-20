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

def get_optimizer(name, named_params, lr, weight_decay=None):
    params = [p for n, p in named_params] # Extrai apenas os tensores para os otimizadores padroes genericos
    
    if name == 'AdamW':
        wd = weight_decay if weight_decay is not None else 1e-2
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == 'SGD':
        wd = weight_decay if weight_decay is not None else 0
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    elif name == 'Muon':            
        import math
        # Filtra ativamente qualquer parametro de 'embedding' para longe do Muon, mesmo que seja 2D.
        hidden_weights = [p for n, p in named_params if p.ndim >= 2 and 'embed' not in n.lower()]
        hidden_gains_biases = [p for n, p in named_params if p.ndim < 2 or 'embed' in n.lower()]

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
                weight_decay=(weight_decay if weight_decay is not None else 5e-4) / adjusted_ratio
            ))
            
        param_groups.append(
            dict(
                params=hidden_gains_biases,
                use_muon=False,
                lr=lr, # AdamW passa a usar a mesma taxa do Muon
                betas=(0.9, 0.95),
                weight_decay=(weight_decay if weight_decay is not None else 5e-4)
            )
        )
        return MuonWithAuxAdam(param_groups)
    elif name == 'Shampoo':
        try:
            import torch_optimizer as optim
            wd = weight_decay if weight_decay is not None else 0
            return optim.Shampoo(params, lr=lr, weight_decay=wd)
        except ImportError:
            raise ImportError("Please install torch_optimizer for Shampoo: pip install torch-optimizer")
    elif name == 'SOAP':
        try:
            from utils.soap import SOAP
            wd = weight_decay if weight_decay is not None else 0.01
            return SOAP(params, lr=lr, weight_decay=wd)
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
OPTIMIZERS = ['AdamW', 'SGD', 'Muon','Shampoo','SOAP']

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
        return result['hits@50'], 'Hits@50'
    elif 'hits@20' in result:
        return result['hits@20'], 'Hits@20'
    elif 'mrr_list' in result:
        return result['mrr_list'].mean().item(), 'MRR'
    else:
        return list(result.values())[0], list(result.keys())[0]

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
        return list(result.values())[0], list(result.keys())[0]
    else:
        pred = out[test_indices].argmax(dim=-1)
        y_true = y[test_indices]
        correct = (pred == (y_true.squeeze() if y_true.dim() > 1 else y_true)).sum().item()
        return correct / test_indices.numel(), 'Accuracy'

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
            return result['acc'], 'Accuracy'
        return list(result.values())[0], list(result.keys())[0]
    else:
        correct = (y_pred == y_true).sum().item()
        return correct / y_true.size(0), 'Accuracy'

def run_experiment(dataset_name, model_name, optimizer_name, seed=42, epochs=10, lr=0.01, 
                   weight_decay=None, patience=None,
                   hidden_channels=256, num_layers=3, dropout=0.5, batch_size=1024):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_info = load_dataset(dataset_name)
    task = DATASETS[dataset_name]['type']
    graph_type = DATASETS[dataset_name]['graph_type']
    
    if task == 'link':
        data, split_edge, evaluator, _ = data_info
        num_relations = data.edge_attr.max().item() + 1 if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1
        if hasattr(data, 'x') and data.x is not None:
            in_channels = data.x.size(1)
        else:
            in_channels = hidden_channels
            print(f"\n[AVISO] Dataset '{dataset_name}' (link) nao possui features 'data.x'. A GNN '{model_name}' deve construir um Embedding interno.\n")
        out_channels = hidden_channels
        gnn = get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, num_relations)
        predictor = LinkPredictor(hidden_channels, hidden_channels, 1, 2, dropout)
        gnn.to(device)
        predictor.to(device)
        named_params = list(gnn.named_parameters()) + list(predictor.named_parameters())
    elif task == 'node':
        data, split_idx, evaluator, _ = data_info
        num_relations = data.edge_attr.max().item() + 1 if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1
        if hasattr(data, 'x') and data.x is not None:
            in_channels = data.x.size(1)
        else:
            in_channels = hidden_channels
            print(f"\n[AVISO] Dataset '{dataset_name}' (node) nao possui features 'data.x'. A GNN '{model_name}' deve construir um Embedding interno.\n")
        out_channels = hidden_channels
        gnn = get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, num_relations)
        num_classes = data.y.max().item() + 1 if dataset_name != 'Cora' else 7  # Cora has 7 classes
        predictor = NodePredictor(hidden_channels, num_classes, 2, dropout)
        gnn.to(device)
        predictor.to(device)
        named_params = list(gnn.named_parameters()) + list(predictor.named_parameters())
    else:
        # Graph
        dataset, split_idx, evaluator, _ = data_info
        
        if hasattr(dataset, 'num_node_features') and dataset.num_node_features > 0:
            in_channels = dataset.num_node_features
        elif hasattr(dataset[0], 'x') and dataset[0].x is not None:
             in_channels = dataset[0].x.size(1)
        else:
            in_channels = hidden_channels
            
        out_channels = hidden_channels
        gnn = get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, 1)
        # OGBG PPA has 37 classes
        num_classes = dataset.num_classes
        predictor = GraphPredictor(hidden_channels, num_classes, 2, dropout)
        gnn.to(device)
        predictor.to(device)
        named_params = list(gnn.named_parameters()) + list(predictor.named_parameters())
    
    optimizer = get_optimizer(optimizer_name, named_params, lr, weight_decay)
    
    train_losses = []
    eval_scores = []
    metric_name = "Score"
    best_score = -1.0
    epochs_no_improve = 0
    
    start_time = time.time()
    for epoch in range(epochs):
        if task == 'link':
            loss = train_link_prediction(gnn, predictor, data, split_edge, optimizer, device, graph_type, batch_size)
            score, m_name = eval_link_prediction(gnn, predictor, data, split_edge, evaluator, device, graph_type)
        elif task == 'node':
            loss = train_node_classification(gnn, predictor, data, split_idx, optimizer, device, graph_type, batch_size)
            score, m_name = eval_node_classification(gnn, predictor, data, split_idx, evaluator, device, graph_type)
        elif task == 'graph':
            loss = train_graph_classification(gnn, predictor, dataset, split_idx, optimizer, device, graph_type, batch_size=32)
            score, m_name = eval_graph_classification(gnn, predictor, dataset, split_idx, evaluator, device, graph_type, batch_size=32)
            
        train_losses.append(loss)
        eval_scores.append(score)
        metric_name = m_name
        
        if score > best_score:
            best_score = score
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        print(f'Epoch {epoch+1:03d}/{epochs}: Train Loss {loss:.4f} | Val/Test {metric_name}: {score:.4f}')
        
        if patience is not None and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}!")
            rem_epochs = epochs - (epoch + 1)
            train_losses.extend([loss] * rem_epochs)
            eval_scores.extend([score] * rem_epochs)
            break
            
    training_time = time.time() - start_time
    return best_score, training_time, train_losses, eval_scores, metric_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys())
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--optimizer', type=str, required=True, choices=OPTIMIZERS + ['all'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--runs', type=int, default=1, help='Numero de execucoes com seeds diferentes')
    args = parser.parse_args()
    
    # Check model
    if DATASETS[args.dataset]['graph_type'] == 'homogeneous' and args.model not in MODELS_HOMO:
        raise ValueError("Invalid model for homogeneous graph")
    if DATASETS[args.dataset]['graph_type'] == 'heterogeneous' and args.model not in MODELS_HETERO:
        raise ValueError("Invalid model for heterogeneous graph")
    
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            
        runs_cfg = cfg.get('experiment', {}).get('runs', args.runs)
        epochs_cfg = cfg.get('experiment', {}).get('epochs', args.epochs)
        batch_size_cfg = cfg.get('experiment', {}).get('batch_size', args.batch_size)
        patience_cfg = cfg.get('experiment', {}).get('patience', None)
        
        lr_cfg = cfg.get('hyperparameters', {}).get('lr', args.lr)
        weight_decay_cfg = cfg.get('hyperparameters', {}).get('weight_decay', None)
        hidden_channels_cfg = cfg.get('hyperparameters', {}).get('hidden_channels', 256)
        num_layers_cfg = cfg.get('hyperparameters', {}).get('num_layers', 3)
        dropout_cfg = cfg.get('hyperparameters', {}).get('dropout', 0.5)
        
        datasets_list = cfg.get('targets', {}).get('datasets', [args.dataset])
        models_list = cfg.get('targets', {}).get('models', [args.model])
        optimizers_list = cfg.get('targets', {}).get('optimizers', [args.optimizer])
        if 'all' in optimizers_list:
            optimizers_list = OPTIMIZERS
    else:
        runs_cfg = args.runs
        epochs_cfg = args.epochs
        lr_cfg = args.lr
        batch_size_cfg = args.batch_size
        patience_cfg = None
        weight_decay_cfg = None
        hidden_channels_cfg = 256
        num_layers_cfg = 3
        dropout_cfg = 0.5
        
        datasets_list = [args.dataset]
        models_list = [args.model]
        optimizers_list = OPTIMIZERS if args.optimizer == 'all' else [args.optimizer]

    results = {}
    for opt in optimizers_list:
        print(f"\n[{opt}] Iniciando treinamento no dataset {args.dataset} com modelo {args.model} ({runs_cfg} runs)...")
        opt_final_scores = []
        opt_losses = []
        opt_scores = []
        time_takens = []
        
        try:
            for run_idx in range(runs_cfg):
                seed = 42 + run_idx
                final_score, time_taken, losses, scores, metric_name = run_experiment(
                    args.dataset, args.model, opt, seed, epochs_cfg, lr_cfg,
                    weight_decay_cfg, patience_cfg, hidden_channels_cfg, num_layers_cfg, dropout_cfg, batch_size_cfg
                )
                opt_final_scores.append(final_score)
                opt_losses.append(losses)
                opt_scores.append(scores)
                time_takens.append(time_taken)
            
            mean_final_score = np.mean(opt_final_scores)
            std_final_score = np.std(opt_final_scores)
            mean_time = np.mean(time_takens)
            
            print(f"> [{opt}] Tempo medio: {mean_time:.2f}s | Final {metric_name}: {mean_final_score:.4f} ± {std_final_score:.4f}")
            results[opt] = {
                'losses': np.array(opt_losses),
                'scores': np.array(opt_scores),
                'mean_final_score': mean_final_score,
                'std_final_score': std_final_score,
                'time': mean_time,
                'metric_name': metric_name
            }
        except Exception as e:
            print(f"> Erro ao rodar otimizador {opt}: {str(e)}")
            
    if results:
        # Gerar grafico
        import matplotlib.pyplot as plt
        import os
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Loss
        plt.subplot(1, 2, 1)
        for opt, hist in results.items():
            mean_loss = hist['losses'].mean(axis=0)
            std_loss = hist['losses'].std(axis=0)
            epochs_range = range(1, args.epochs + 1)
            line, = plt.plot(epochs_range, mean_loss, label=f"{opt}")
            if args.runs > 1:
                plt.fill_between(epochs_range, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2, color=line.get_color())
        plt.title(f'Treinamento Loss ({args.dataset} / {args.model})')
        plt.xlabel('Epoca')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Score
        plt.subplot(1, 2, 2)
        # Pega a metrica usada do primeiro resultado valido no dicionario
        metric = list(results.values())[0]['metric_name']
        for opt, hist in results.items():
            mean_score = hist['scores'].mean(axis=0)
            std_score = hist['scores'].std(axis=0)
            epochs_range = range(1, args.epochs + 1)
            line, = plt.plot(epochs_range, mean_score, label=f"{opt} ({metric} final={hist['mean_final_score']:.3f}±{hist['std_final_score']:.3f})")
            if args.runs > 1:
                plt.fill_between(epochs_range, mean_score - std_score, mean_score + std_score, alpha=0.2, color=line.get_color())
        plt.title(f'Score de Validacao/Teste ({metric})')
        plt.xlabel('Epoca')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"comparativo_{args.dataset}_{args.model}.png"
        plt.savefig(plot_path)
        print(f"\nGrafico comparativo salvo com sucesso em: {plot_path}")