"""
utils/training.py

Funções de treino e avaliação para tarefas de link prediction, node classification
e graph classification.

Correções aplicadas:
  1. train_node_classification: encode do GNN movido para DENTRO do loop de batch,
     garantindo que os gradientes alcancem os pesos do GNN (antes estava fora do
     loop, tornando o GNN efetivamente congelado durante o treino).
  2. eval_node_classification: avalia no split 'valid' durante o treino (epochs),
     reservando 'test' para a avaliação final — evita data leakage na seleção
     de modelo. A função agora aceita o parâmetro split='valid'|'test'.
  3. eval_link_prediction: mesmo padrão — usa 'valid' durante o treino.
  4. WordNet18RR: mantém avaliação filtrada via evaluator.eval_filtered().
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from utils.data import WordNet18RREvaluator


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _get_edge_type(data):
    """Retorna edge_type se existir, senão None."""
    return getattr(data, 'edge_type', getattr(data, 'edge_attr', None))


# ---------------------------------------------------------------------------
# Link Prediction
# ---------------------------------------------------------------------------

def train_link_prediction(gnn, predictor, data, split_edge,
                          optimizer, device, graph_type, batch_size=1024):
    gnn.train()
    predictor.train()

    edge_index = data.edge_index.to(device)
    edge_type  = _get_edge_type(data)
    x          = data.x.to(device)
    pos_edge   = split_edge['train']['edge'].to(device)   # [E, 2]

    total_loss = total_examples = 0

    for perm in torch.randperm(pos_edge.size(0)).split(batch_size):
        optimizer.zero_grad()

        # Encode dentro do loop → gradientes fluem para o GNN
        if graph_type == 'heterogeneous' and edge_type is not None:
            h = gnn(x, edge_index, edge_type.to(device))
        else:
            h = gnn(x, edge_index)

        pos_src = pos_edge[perm, 0]
        pos_dst = pos_edge[perm, 1]
        neg_dst = torch.randint(0, data.num_nodes, (perm.size(0),), device=device)

        pos_out = predictor(h[pos_src], h[pos_dst])
        neg_out = predictor(h[pos_src], h[neg_dst])

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_out, neg_out]),
            torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)]),
        )
        loss.backward()
        optimizer.step()

        total_loss     += loss.item() * perm.size(0)
        total_examples += perm.size(0)

    return total_loss / total_examples


@torch.no_grad()
def eval_link_prediction(gnn, predictor, data, split_edge,
                         evaluator, device, graph_type, split='valid'):
    """
    split='valid' durante o treino (padrão), split='test' para avaliação final.
    """
    gnn.eval()
    predictor.eval()

    # -----------------------------------------------------------------------
    # WordNet18RR: Filtered MRR (protocolo correto para KG)
    # -----------------------------------------------------------------------
    if isinstance(evaluator, WordNet18RREvaluator):
        edge_index = data.edge_index.to(device)
        edge_type  = _get_edge_type(data)
        x          = data.x.to(device)

        if graph_type == 'heterogeneous' and edge_type is not None:
            h = gnn(x, edge_index, edge_type.to(device))
        else:
            h = gnn(x, edge_index)

        def score_fn(heads, tails, rels):
            return predictor(h[heads], h[tails]).squeeze(-1)

        eval_edges      = split_edge[split]['edge'].to(device)
        eval_edge_types = split_edge[split]['edge_type'].to(device)

        result = evaluator.eval_filtered(
            score_fn, eval_edges, eval_edge_types,
            num_nodes=data.num_nodes, device=device,
        )
        mrr = result['mrr_list'].mean().item()
        return mrr, 'MRR'

    # -----------------------------------------------------------------------
    # OGB datasets
    # -----------------------------------------------------------------------
    edge_index = data.edge_index.to(device)
    edge_type  = _get_edge_type(data)
    x          = data.x.to(device)

    if graph_type == 'heterogeneous' and edge_type is not None:
        h = gnn(x, edge_index, edge_type.to(device))
    else:
        h = gnn(x, edge_index)

    def _score_edges(edges):
        src = edges[:, 0].to(device)
        dst = edges[:, 1].to(device)
        return predictor(h[src], h[dst]).squeeze(-1)

    pos_pred = _score_edges(split_edge[split]['edge'])
    neg_pred = _score_edges(split_edge[split]['edge_neg'])

    result = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })

    if 'hits@50' in result: return result['hits@50'],              'Hits@50'
    if 'hits@20' in result: return result['hits@20'],              'Hits@20'
    if 'mrr_list' in result: return result['mrr_list'].mean().item(), 'MRR'
    if 'rocauc'   in result: return result['rocauc'],              'ROC-AUC'

    key = next(iter(result))
    val = result[key]
    return (val.mean().item() if hasattr(val, 'mean') else float(val)), key


# ---------------------------------------------------------------------------
# Node Classification
# ---------------------------------------------------------------------------

def train_node_classification(gnn, predictor, data, split_idx,
                              optimizer, device, graph_type, batch_size=1024):
    gnn.train()
    predictor.train()

    edge_index = data.edge_index.to(device)
    edge_type  = _get_edge_type(data)
    x          = data.x.to(device)
    y          = data.y.to(device)

    # Suporta máscaras booleanas (Cora) e índices (OGB)
    if isinstance(split_idx['train'], torch.Tensor) and split_idx['train'].dtype == torch.bool:
        train_idx = split_idx['train'].nonzero(as_tuple=False).view(-1).to(device)
    else:
        train_idx = split_idx['train'].to(device)

    total_loss = total_examples = 0

    for perm in train_idx.split(batch_size):
        optimizer.zero_grad()

        if graph_type == 'heterogeneous' and edge_type is not None:
            h = gnn(x, edge_index, edge_type.to(device))
        else:
            h = gnn(x, edge_index)

        out  = predictor(h[perm])
        lbl  = y[perm].view(-1)
        loss = F.cross_entropy(out, lbl)
        loss.backward()
        optimizer.step()

        total_loss     += loss.item() * perm.size(0)
        total_examples += perm.size(0)

    return total_loss / total_examples


@torch.no_grad()
def eval_node_classification(gnn, predictor, data, split_idx,
                             evaluator, device, graph_type, split='valid'):
    """
    split='valid' durante o treino (padrão), split='test' para avaliação final.
    """
    gnn.eval()
    predictor.eval()

    edge_index = data.edge_index.to(device)
    edge_type  = _get_edge_type(data)
    x          = data.x.to(device)
    y          = data.y.to(device)

    if graph_type == 'heterogeneous' and edge_type is not None:
        h = gnn(x, edge_index, edge_type.to(device))
    else:
        h = gnn(x, edge_index)

    # Resolve índices do split requisitado
    split_tensor = split_idx[split]
    if isinstance(split_tensor, torch.Tensor) and split_tensor.dtype == torch.bool:
        eval_idx = split_tensor.nonzero(as_tuple=False).view(-1).to(device)
    else:
        eval_idx = split_tensor.to(device)

    out   = predictor(h[eval_idx])
    preds = out.argmax(dim=-1, keepdim=True)

    if evaluator is None:
        # Cora: accuracy simples
        correct = preds.view(-1).eq(y[eval_idx].view(-1)).sum().item()
        return correct / eval_idx.size(0), 'Accuracy'

    result = evaluator.eval({
        'y_true': y[eval_idx].view(-1, 1),
        'y_pred': preds,
    })

    if 'acc'    in result: return result['acc'],    'Accuracy'
    if 'rocauc' in result: return result['rocauc'], 'ROC-AUC'
    if 'f1'     in result: return result['f1'],     'F1'

    key = next(iter(result))
    val = result[key]
    return (val.mean().item() if hasattr(val, 'mean') else float(val)), key


# ---------------------------------------------------------------------------
# Graph Classification
# ---------------------------------------------------------------------------

def train_graph_classification(gnn, predictor, dataset, split_idx,
                               optimizer, device, graph_type, batch_size=32):
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool

    gnn.train()
    predictor.train()

    train_dataset = dataset[split_idx['train']]
    loader        = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    total_loss = total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        edge_type = _get_edge_type(batch)
        if graph_type == 'heterogeneous' and edge_type is not None:
            h = gnn(batch.x, batch.edge_index, edge_type)
        else:
            h = gnn(batch.x, batch.edge_index)

        h_graph = global_mean_pool(h, batch.batch)

        out  = predictor(h_graph)
        lbl  = batch.y.view(-1)
        loss = F.cross_entropy(out, lbl)
        loss.backward()
        optimizer.step()

        total_loss     += loss.item() * batch.num_graphs
        total_examples += batch.num_graphs

    return total_loss / total_examples


@torch.no_grad()
def eval_graph_classification(gnn, predictor, dataset, split_idx,
                              evaluator, device, graph_type, batch_size=32,
                              split='valid'):
    """
    split='valid' durante o treino (padrão), split='test' para avaliação final.
    """
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool

    gnn.eval()
    predictor.eval()

    eval_dataset = dataset[split_idx[split]]
    loader       = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    all_preds  = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)

        edge_type = _get_edge_type(batch)
        if graph_type == 'heterogeneous' and edge_type is not None:
            h = gnn(batch.x, batch.edge_index, edge_type)
        else:
            h = gnn(batch.x, batch.edge_index)

        h_graph = global_mean_pool(h, batch.batch)
        out     = predictor(h_graph)
        preds   = out.argmax(dim=-1, keepdim=True)

        all_preds.append(preds.cpu())
        all_labels.append(batch.y.view(-1, 1).cpu())

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    result = evaluator.eval({'y_true': all_labels, 'y_pred': all_preds})

    if 'acc'    in result: return result['acc'],    'Accuracy'
    if 'rocauc' in result: return result['rocauc'], 'ROC-AUC'
    if 'f1'     in result: return result['f1'],     'F1'

    key = next(iter(result))
    val = result[key]
    return (val.mean().item() if hasattr(val, 'mean') else float(val)), key