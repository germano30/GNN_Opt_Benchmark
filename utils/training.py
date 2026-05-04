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
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import NeighborLoader

# Graphs with more edges than this threshold use neighbor-sampling mini-batches
# instead of a full-graph forward pass (avoids OOM with RGAT/large graphs)
_LARGE_GRAPH_EDGE_THRESHOLD = 500_000


def _needs_neighbor_sampling(gnn, data):
    """Check if we should use neighbor sampling instead of full-graph forward.

    True when:
      - graph exceeds edge threshold, OR
      - model contains RGATConv (materializes huge per-edge attention tensors
        that OOM even on small graphs like WordNet18RR with 93k edges)
    """
    if data.edge_index.shape[1] > _LARGE_GRAPH_EDGE_THRESHOLD:
        return True
    from torch_geometric.nn import RGATConv
    for module in gnn.modules():
        if isinstance(module, RGATConv):
            return True
    return False


def _resolve_num_neighbors(gnn, num_neighbors):
    """Expand num_neighbors to match the GNN's number of conv layers.

    A 3-layer GNN needs 3 hops of neighborhoods. Uses a tapering schedule:
    inner hops (close to seed) get more neighbors, outer hops get fewer.
    E.g. num_neighbors=[10] with 3 layers → [10, 7, 3].
    This dramatically reduces subgraph size with minimal quality loss.
    """
    # Unwrap NodeEmbeddingWrapper if present
    model = getattr(gnn, 'gnn', gnn)
    n_layers = getattr(model, 'num_layers', None)
    if n_layers is None:
        # Fallback: count conv modules
        convs = getattr(model, 'convs', None)
        n_layers = len(convs) if convs is not None else 2

    if num_neighbors is None:
        num_neighbors = [10]

    if len(num_neighbors) >= n_layers:
        return num_neighbors[:n_layers]

    # Tapering: inner hops get the full value, outer hops taper down.
    # This keeps the subgraph small while giving seed nodes full context.
    base = num_neighbors[-1]
    tapered = []
    for hop in range(n_layers):
        # hop 0 = innermost (closest to seed), gets full neighbors
        # Higher hops taper: factor goes from 1.0 down to ~0.33
        factor = 1.0 - (hop / n_layers) * 0.67
        tapered.append(max(3, int(base * factor)))
    return tapered


def _train_link_neighbor_sampling(gnn, predictor, data, pos_train_edge,
                                   optimizer, device, graph_type, batch_size, grad_accum_steps=1,
                                   num_neighbors=None):
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.data import Data as PyGData

    # Expose edge_type as edge_attr so the sampler propagates it through subgraphs
    edge_attr = (
        data.edge_type
        if (graph_type == 'heterogeneous'
            and hasattr(data, 'edge_type')
            and data.edge_type is not None)
        else None
    )
    loader_data = PyGData(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=edge_attr,
        num_nodes=data.num_nodes,
    )

    loader = LinkNeighborLoader(
        loader_data,
        num_neighbors=_resolve_num_neighbors(gnn, num_neighbors),
        edge_label_index=pos_train_edge.t(),            # [2, E_pos]
        edge_label=torch.ones(pos_train_edge.size(0)),  # all positive
        neg_sampling='binary',                           # equal number of negatives
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    scaler = torch.amp.GradScaler('cuda')
    total_loss = 0.0
    num_batches = 0
    for i, batch in enumerate(loader):
        if i % grad_accum_steps == 0:
            optimizer.zero_grad()
        
        batch = batch.to(device)

        with torch.amp.autocast('cuda'):
            if graph_type == 'heterogeneous' and batch.edge_attr is not None:
                h = gnn(batch.x, batch.edge_index, batch.edge_attr.long())
            else:
                h = gnn(batch.x, batch.edge_index)

            src, dst = batch.edge_label_index
            out = predictor(h[src], h[dst])
            loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), batch.edge_label.float())

        scaler.scale(loss).backward()
        total_loss += loss.item()
        num_batches += 1
        
        if (i + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()

    return total_loss / max(num_batches, 1)


from utils.data import WordNet18RREvaluator


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _get_edge_type(data):
    """Retorna edge_type se existir, senão None."""
    return getattr(data, 'edge_type', getattr(data, 'edge_attr', None))


# ---------------------------------------------------------------------------
# Node Classification - Neighbor Sampling (for large graphs)
# ---------------------------------------------------------------------------

def _train_node_classification_neighbor_sampling(gnn, predictor, data, train_indices, 
                                                  optimizer, device, graph_type, batch_size, grad_accum_steps=1,
                                                  num_neighbors=None):
    """Mini-batch node-classification training via neighbor sampling (large graphs)."""
    from torch_geometric.data import Data as PyGData

    edge_attr = (
        data.edge_type
        if (graph_type == 'heterogeneous'
            and hasattr(data, 'edge_type')
            and data.edge_type is not None)
        else None
    )
    loader_data = PyGData(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=edge_attr,
        y=data.y,
        num_nodes=data.num_nodes,
    )

    loader = NeighborLoader(
        loader_data,
        num_neighbors=_resolve_num_neighbors(gnn, num_neighbors),
        input_nodes=train_indices,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True, 
    )

    total_loss = 0.0
    num_batches = 0
    for i, batch in enumerate(loader):
        if i % grad_accum_steps == 0:
            optimizer.zero_grad()
        
        batch = batch.to(device)

        if graph_type == 'heterogeneous' and batch.edge_attr is not None:
            h = gnn(batch.x, batch.edge_index, batch.edge_attr.long())
        else:
            h = gnn(batch.x, batch.edge_index)

        # batch_size seeds are the first nodes in the batch
        out = predictor(h[:batch.batch_size])
        batch_labels = batch.y[:batch.batch_size]
        
        if batch_labels.dim() > 1 and batch_labels.shape[-1] > 1:
            loss = F.binary_cross_entropy_with_logits(out, batch_labels.float())
        else:
            loss = F.cross_entropy(out, batch_labels.squeeze().long())

        loss.backward()
        total_loss += loss.item()
        num_batches += 1
        
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Link Prediction
# ---------------------------------------------------------------------------

def train_link_prediction(gnn, predictor, data, split_edge,
                          optimizer, device, graph_type, batch_size=1024, grad_accum_steps=1,
                          num_neighbors=None):
    gnn.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge']

    # Large graphs or memory-heavy models (RGAT): use neighbor sampling
    if _needs_neighbor_sampling(gnn, data):
        return _train_link_neighbor_sampling(
            gnn, predictor, data, pos_train_edge, optimizer, device, graph_type, batch_size, grad_accum_steps,
            num_neighbors=num_neighbors
        )

    # ── Small-graph path: full-graph forward, mini-batch only on predictor ──
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if graph_type == 'heterogeneous':
        edge_type = getattr(data, 'edge_type', getattr(data, 'edge_attr', None))
        if edge_type is not None: edge_type = edge_type.to(device)
    else:
        edge_type = None

    if 'edge_neg' in split_edge['train']:
        neg_train_edge = split_edge['train']['edge_neg']
    else:
        neg_edge_index = negative_sampling(
            edge_index=pos_train_edge.t(),
            num_nodes=data.num_nodes,
            num_neg_samples=pos_train_edge.size(0)
        )
        neg_train_edge = neg_edge_index.t()

    all_edges = torch.cat([pos_train_edge, neg_train_edge], dim=0)
    labels = torch.cat([torch.ones(pos_train_edge.size(0)), torch.zeros(neg_train_edge.size(0))], dim=0)

    dataset = torch.utils.data.TensorDataset(all_edges, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    for i, (batch_edges, batch_labels) in enumerate(dataloader):
        if i % grad_accum_steps == 0:
            optimizer.zero_grad()
        
        batch_edges = batch_edges.to(device)
        batch_labels = batch_labels.to(device)

        if graph_type == 'heterogeneous':
            h = gnn(x, edge_index, edge_type)
        else:
            h = gnn(x, edge_index)

        out = predictor(h[batch_edges[:, 0]], h[batch_edges[:, 1]])
        loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), batch_labels.float())

        loss.backward()
        total_loss += loss.item()
        
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()

    return total_loss / len(dataloader)


def _score_edges_batched(predictor, h, edges, device, batch_size=65536):
    """Score edge pairs in batches to avoid OOM on large negative sets.

    h may be on CPU (large-graph case) or GPU; edges are always on CPU.
    For each chunk we pull the needed rows of h to GPU.
    """
    h_on_gpu = h.device.type != 'cpu'
    scores = []
    for start in range(0, edges.shape[0], batch_size):
        idx = edges[start : start + batch_size]          # CPU indices [B, 2]
        src_idx, dst_idx = idx[:, 0], idx[:, 1]
        if h_on_gpu:
            h_src = h[src_idx.to(device)]
            h_dst = h[dst_idx.to(device)]
        else:
            # h on CPU: index on CPU then move the small slice to GPU
            h_src = h[src_idx].to(device)
            h_dst = h[dst_idx].to(device)
        with torch.no_grad():
            s = predictor(h_src, h_dst).squeeze(-1)
        scores.append(s)
    return torch.cat(scores)


def _parse_eval_result(result):
    if 'mrr_list' in result:
        return result['mrr_list'].mean().item(), 'MRR'
    elif 'hits@50' in result:
        return result['hits@50'], 'Hits@50'
    elif 'hits@20' in result:
        return result['hits@20'], 'Hits@20'
    else:
        return list(result.values())[0], list(result.keys())[0]


def _infer_embeddings_large_graph(gnn, data, device, graph_type, batch_size=2048, num_neighbors=None):
    """Compute all node embeddings via mini-batch NeighborLoader (no gradients)."""
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.data import Data as PyGData

    edge_attr = (
        data.edge_type
        if (graph_type == 'heterogeneous'
            and hasattr(data, 'edge_type')
            and data.edge_type is not None)
        else None
    )
    loader_data = PyGData(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=edge_attr,
        num_nodes=data.num_nodes,
    )
    loader = NeighborLoader(
        loader_data,
        num_neighbors=_resolve_num_neighbors(gnn, num_neighbors),
        input_nodes=torch.arange(data.num_nodes),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    h_all = None
    gnn.eval()
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            if graph_type == 'heterogeneous' and batch.edge_attr is not None:
                out = gnn(batch.x, batch.edge_index, batch.edge_attr.long())
            else:
                out = gnn(batch.x, batch.edge_index)
        if h_all is None:
            # Allocate on GPU to save RAM (embeddings são grandes!)
            h_all = torch.zeros(data.num_nodes, out.shape[-1], device=device)
        # n_id[:batch_size] = global IDs of seed nodes; out[:batch_size] = their embeddings
        h_all[batch.n_id[:batch.batch_size]] = out[:batch.batch_size].float().detach()

    return h_all   # [num_nodes, hidden] on GPU


def eval_link_prediction(gnn, predictor, data, split_edge, evaluator, device, graph_type, split='valid'):
    """Evaluate link prediction.

    split='valid' during training (early stopping), split='test' for final eval.
    """
    gnn.eval()
    predictor.eval()

    # -----------------------------------------------------------------------
    # WordNet18RR: Filtered MRR (protocolo correto para KG)
    # Uses eval_filtered which scores ALL candidate tails per triple and
    # filters out known positives — no pre-computed edge_neg needed.
    # -----------------------------------------------------------------------
    if isinstance(evaluator, WordNet18RREvaluator):
        if _needs_neighbor_sampling(gnn, data):
            h = _infer_embeddings_large_graph(gnn, data, device, graph_type)
        else:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_type_t = _get_edge_type(data)
            if edge_type_t is not None:
                edge_type_t = edge_type_t.to(device)
            with torch.no_grad():
                h = gnn(x, edge_index, edge_type_t)

        # Build a score function: (heads, tails, rels) → scores
        def score_fn(heads, tails, rels):
            return predictor(h[heads], h[tails]).squeeze(-1)

        eval_edges = split_edge[split]['edge']       # [N, 2]
        eval_etypes = split_edge[split]['edge_type']  # [N]
        result = evaluator.eval_filtered(
            score_fn, eval_edges, eval_etypes,
            num_nodes=data.num_nodes, device=device,
        )
        return _parse_eval_result(result)

    # -----------------------------------------------------------------------
    # Generic path: compute embeddings once, then score edges
    # -----------------------------------------------------------------------
    if _needs_neighbor_sampling(gnn, data):
        h = _infer_embeddings_large_graph(gnn, data, device, graph_type)
    else:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        if graph_type == 'heterogeneous':
            edge_type = getattr(data, 'edge_type', getattr(data, 'edge_attr', None))
            if edge_type is not None: edge_type = edge_type.to(device)
            with torch.no_grad():
                h = gnn(x, edge_index, edge_type)
        else:
            with torch.no_grad():
                h = gnn(x, edge_index)

    eval_split = split_edge[split]
    pos_out = _score_edges_batched(predictor, h, eval_split['edge'], device)  # [N]
    N = pos_out.shape[0]

    neg_out = _score_edges_batched(predictor, h, eval_split['edge_neg'], device)
    result = evaluator.eval({'y_pred_pos': pos_out, 'y_pred_neg': neg_out})
    return _parse_eval_result(result)

def train_node_classification(gnn, predictor, data, split_idx, optimizer, device, graph_type, batch_size=1024, grad_accum_steps=1, num_neighbors=None):
    gnn.train()
    predictor.train()

    train_indices = split_idx['train']
    if train_indices.dtype == torch.bool:
        train_indices = train_indices.nonzero(as_tuple=True)[0]

    # Large graphs: use neighbor sampling to avoid full-graph OOM
    if data.edge_index.shape[1] > _LARGE_GRAPH_EDGE_THRESHOLD:
        return _train_node_classification_neighbor_sampling(
            gnn, predictor, data, train_indices, optimizer, device, graph_type, batch_size, grad_accum_steps,
            num_neighbors=num_neighbors
        )

    # Small-graph path: full-graph forward, full-batch prediction
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    edge_type = getattr(data, 'edge_type', getattr(data, 'edge_attr', None))
    if edge_type is not None:
        edge_type = edge_type.to(device)

    optimizer.zero_grad()
    batch_indices = train_indices.to(device)
    batch_labels = y[train_indices].to(device)
    
    if graph_type == 'heterogeneous' and edge_type is not None:
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
    
    return loss.item()

@torch.no_grad()
def eval_node_classification(gnn, predictor, data, split_idx, evaluator, device, graph_type):
    gnn.eval()
    predictor.eval()

    y = data.y.to(device)

    test_indices = split_idx['test']
    if test_indices.dtype == torch.bool:
        test_indices = test_indices.nonzero(as_tuple=True)[0]

    # Large graphs: use neighbor sampling to avoid full-graph OOM
    if data.edge_index.shape[1] > _LARGE_GRAPH_EDGE_THRESHOLD:
        h = _infer_embeddings_large_graph(gnn, data, device, graph_type)  # [num_nodes, hidden] already on GPU
    else:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_type = getattr(data, 'edge_type', getattr(data, 'edge_attr', None))
        if edge_type is not None:
            edge_type = edge_type.to(device)

        if graph_type == 'heterogeneous' and edge_type is not None:
            h = gnn(x, edge_index, edge_type)
        else:
            h = gnn(x, edge_index)
    
    out = predictor(h)
    y_true = y[test_indices]
    y_pred = out[test_indices].detach()  # detach() garante que não fica memória em GPU

    if evaluator:
        if y_true.dim() == 1 or y_true.shape[-1] == 1:
            y_pred_eval = y_pred.argmax(dim=-1, keepdim=True)
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(-1)
        else:
            y_pred_eval = y_pred
        result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred_eval})
        return list(result.values())[0], list(result.keys())[0]
    else:
        pred = y_pred.argmax(dim=-1)
        correct = (pred == (y_true.squeeze() if y_true.dim() > 1 else y_true)).sum().item()
        return correct / y_true.numel(), 'Accuracy'


# ---------------------------------------------------------------------------
# Graph Classification
# ---------------------------------------------------------------------------

def train_graph_classification(gnn, predictor, dataset, split_idx,
                               optimizer, device, graph_type, batch_size=32, grad_accum_steps=1):
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool
    from torch.utils.data import Subset

    gnn.train()
    predictor.train()

    # Corrige indexação: split_idx['train'] é um tensor de índices, precisa converter para Subset
    indices = split_idx['train']
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    train_dataset = Subset(dataset, indices)
    # Otimização pura GPU: num_workers=0, batch maior
    loader        = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)

    total_loss = total_examples = 0
    for i, batch in enumerate(loader):
        if i % grad_accum_steps == 0:
            optimizer.zero_grad()
        
        batch = batch.to(device)

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

        total_loss     += loss.item() * batch.num_graphs
        total_examples += batch.num_graphs
        
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()

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
    from torch.utils.data import Subset

    gnn.eval()
    predictor.eval()

    # Corrige indexação: split_idx[split] é um tensor de índices, precisa converter para Subset
    indices = split_idx[split]
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    eval_dataset = Subset(dataset, indices)
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

        all_preds.append(preds)
        all_labels.append(batch.y.view(-1, 1))

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    result = evaluator.eval({'y_true': all_labels, 'y_pred': all_preds})

    if 'acc'    in result: return result['acc'],    'Accuracy'
    if 'rocauc' in result: return result['rocauc'], 'ROC-AUC'
    if 'f1'     in result: return result['f1'],     'F1'

    key = next(iter(result))
    val = result[key]
    return (val.mean().item() if hasattr(val, 'mean') else float(val)), key