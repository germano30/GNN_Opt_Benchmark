import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import global_mean_pool

def train_link_prediction(gnn, predictor, data, split_edge, optimizer, device, graph_type, batch_size=1024):
    gnn.train()
    predictor.train()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if graph_type == 'heterogeneous':
        edge_type = getattr(data, 'edge_type', getattr(data, 'edge_attr', None))
        if edge_type is not None: edge_type = edge_type.to(device)
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
        edge_type = getattr(data, 'edge_type', getattr(data, 'edge_attr', None))
        if edge_type is not None: edge_type = edge_type.to(device)
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
        edge_type = getattr(data, 'edge_type', getattr(data, 'edge_attr', None))
        if edge_type is not None: edge_type = edge_type.to(device)
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
        edge_type = getattr(data, 'edge_type', getattr(data, 'edge_attr', None))
        if edge_type is not None: edge_type = edge_type.to(device)
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


def train_graph_classification(gnn, predictor, dataset, split_idx, optimizer, device, graph_type, batch_size=32):
    gnn.train()
    predictor.train()
    
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=batch_size, shuffle=True)
    
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        
        if graph_type == 'heterogeneous':
            b_edge_t = getattr(batch, 'edge_type', getattr(batch, 'edge_attr', None))
            if b_edge_t is not None: b_edge_t = b_edge_t.to(device)
            h = gnn(batch.x, batch.edge_index, b_edge_t)
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
