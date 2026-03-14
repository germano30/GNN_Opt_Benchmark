import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models.gat import GAT
from models.gcn import GCN
from models.rgat import RGAT
from models.rgcn import RGCN
from models.gin import GIN

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator as LinkEvaluator
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator as NodeEvaluator
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator as GraphEvaluator

from muon import MuonWithAuxAdam
# Assume other optimizers
# from shampoo import Shampoo
# from soap import SOAP

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

# Placeholder for optimizers
def get_optimizer(name, params, lr):
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr)
    elif name == 'SGD':
        return torch.optim.SGD(params, lr=lr)
    elif name == 'Muon':
        return MuonWithAuxAdam(params, lr=lr)
    # Add others
    else:
        raise ValueError(f"Unknown optimizer {name}")

# Dataset info
DATASETS = {
    'ogbl-collab': {'type': 'link', 'graph_type': 'homogeneous'},
    'ogbn-proteins': {'type': 'node', 'graph_type': 'homogeneous'},
    'ogbg-ppa': {'type': 'graph', 'graph_type': 'homogeneous'},
    'Cora': {'type': 'node', 'graph_type': 'homogeneous'},
    'WordNet18RR': {'type': 'link', 'graph_type': 'heterogeneous'},
    'OGB-BioKG': {'type': 'link', 'graph_type': 'heterogeneous'},
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
        split_edge = dataset.get_edge_split()
        return data, split_edge, evaluator, 'link'
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

def train_link_prediction(gnn, predictor, data, split_edge, optimizer, device, graph_type):
    gnn.train()
    predictor.train()
    optimizer.zero_grad()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if graph_type == 'heterogeneous':
        edge_type = data.edge_attr.to(device)
        h = gnn(x, edge_index, edge_type)
    else:
        adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).t().to(device)
        h = gnn(x, adj_t)
    
    pos_train_edge = split_edge['train']['edge'].to(device)
    neg_train_edge = split_edge['train']['edge_neg'].to(device)
    
    pos_out = predictor(h[pos_train_edge[:, 0]], h[pos_train_edge[:, 1]])
    neg_out = predictor(h[neg_train_edge[:, 0]], h[neg_train_edge[:, 1]])
    
    pos_loss = -torch.log(pos_out + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
    loss = pos_loss + neg_loss
    
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_link_prediction(gnn, predictor, data, split_edge, evaluator, device, graph_type):
    gnn.eval()
    predictor.eval()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if graph_type == 'heterogeneous':
        edge_type = data.edge_attr.to(device)
        h = gnn(x, edge_index, edge_type)
    else:
        adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).t().to(device)
        h = gnn(x, adj_t)
    
    pos_test_edge = split_edge['test']['edge'].to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(device)
    
    pos_out = predictor(h[pos_test_edge[:, 0]], h[pos_test_edge[:, 1]])
    neg_out = predictor(h[neg_test_edge[:, 0]], h[neg_test_edge[:, 1]])
    
    y_pred_pos = torch.cat([pos_out, neg_out], dim=0)
    y_pred_neg = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)], dim=0)
    
    result = evaluator.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})
    return result['hits@20']  # Assuming

def train_node_classification(model, data, split_idx, optimizer, device, graph_type):
    model.train()
    optimizer.zero_grad()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    
    if graph_type == 'heterogeneous':
        edge_type = data.edge_attr.to(device)
        out = model(x, edge_index, edge_type)
    else:
        adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).t().to(device)
        out = model(x, adj_t)
    
    loss = F.cross_entropy(out[split_idx['train']], y[split_idx['train']])
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_node_classification(model, data, split_idx, evaluator, device, graph_type):
    model.eval()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    
    if graph_type == 'heterogeneous':
        edge_type = data.edge_attr.to(device)
        out = model(x, edge_index, edge_type)
    else:
        adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).t().to(device)
        out = model(x, adj_t)
    
    pred = out.argmax(dim=1)
    if evaluator:
        result = evaluator.eval({'y_true': y[split_idx['test']], 'y_pred': pred[split_idx['test']]})
        return result['acc']
    else:
        correct = (pred[split_idx['test']] == y[split_idx['test']]).sum().item()
        return correct / split_idx['test'].sum().item()

# Placeholder for graph classification
def train_graph_classification(model, dataset, split_idx, optimizer, device):
    # Placeholder
    return 0.0

def eval_graph_classification(model, dataset, split_idx, evaluator, device):
    # Placeholder
    return 0.0

def run_experiment(dataset_name, model_name, optimizer_name, epochs=10, lr=0.01, hidden_channels=256, num_layers=3, dropout=0.5):
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
        out_channels = data.y.max().item() + 1 if dataset_name != 'Cora' else 7  # Cora has 7 classes
        model = get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, num_relations)
        model.to(device)
        params = model.parameters()
    else:
        # Graph
        dataset, split_idx, evaluator, _ = data_info
        # Placeholder
        return 0.0, 0.0
    
    optimizer = get_optimizer(optimizer_name, params, lr)
    
    start_time = time.time()
    for epoch in range(epochs):
        if task == 'link':
            loss = train_link_prediction(gnn, predictor, data, split_edge, optimizer, device, graph_type)
        elif task == 'node':
            loss = train_node_classification(model, data, split_idx, optimizer, device, graph_type)
        print(f'Epoch {epoch}: Loss {loss:.4f}')
    training_time = time.time() - start_time
    
    # Evaluate
    if task == 'link':
        score = eval_link_prediction(gnn, predictor, data, split_edge, evaluator, device, graph_type)
    elif task == 'node':
        score = eval_node_classification(model, data, split_idx, evaluator, device, graph_type)
    
    return score, training_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys())
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--optimizer', type=str, required=True, choices=OPTIMIZERS)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    
    # Check model
    if DATASETS[args.dataset]['graph_type'] == 'homogeneous' and args.model not in MODELS_HOMO:
        raise ValueError("Invalid model for homogeneous graph")
    if DATASETS[args.dataset]['graph_type'] == 'heterogeneous' and args.model not in MODELS_HETERO:
        raise ValueError("Invalid model for heterogeneous graph")
    
    score, time_taken = run_experiment(args.dataset, args.model, args.optimizer, args.epochs, args.lr)
    print(f'Dataset: {args.dataset}, Model: {args.model}, Optimizer: {args.optimizer}, Score: {score:.4f}, Time: {time_taken:.2f}s')