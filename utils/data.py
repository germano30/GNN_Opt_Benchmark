import torch
import numpy as np
from torch_geometric.datasets import Planetoid, WordNet18RR
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator as LinkEvaluator
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator as NodeEvaluator
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator as GraphEvaluator


# ---------------------------------------------------------------------------
# WordNet18RR: Filtered MRR evaluator
# ---------------------------------------------------------------------------

class WordNet18RREvaluator:
    """
    Filtered MRR para Knowledge Graphs (protocolo padrão da literatura).

    Para cada tripla positiva (h, r, t) do split avaliado:
      - Substitui o nó-cauda por TODOS os nós do grafo → candidatos brutos
      - Remove candidatos que existam como positivos em qualquer split
        (train ∪ valid ∪ test) com a mesma (h, r) — isso é o "filtered"
      - Calcula o rank do verdadeiro t entre os candidatos restantes

    Isso é exatamente o protocolo usado em:
      TransE (Bordes et al., 2013), RotatE, ComplEx, etc.
    e produz MRR na faixa 0.45–0.57 para modelos relacionais no WN18RR.

    Uso:
        evaluator = WordNet18RREvaluator()
        evaluator.build_filter(all_true_triples)   # chamado uma vez
        mrr = evaluator.eval_filtered(model_score_fn, test_triples)
    """

    def __init__(self):
        self._filter_set = None   # set de (h, r, t) verdadeiros

    # ------------------------------------------------------------------
    # Constrói o conjunto de filtragem a partir de TODAS as triplas
    # (train + valid + test). Deve ser chamado antes de eval_filtered.
    # ------------------------------------------------------------------
    def build_filter(self, all_triples: torch.Tensor):
        """
        all_triples : LongTensor de shape [N, 3] — colunas (head, tail, rel)
                      ou (head, rel, tail), conforme o formato do split_idx.
                      Aqui esperamos (head, tail) em colunas 0,1 e rel em 2,
                      i.e. o formato que vem de split_idx['train']['edge'] ++ edge_type.
        """
        self._filter_set = set(
            (int(h), int(r), int(t))
            for h, t, r in all_triples.tolist()
        )

    # ------------------------------------------------------------------
    # Interface simples usada internamente por eval_link_prediction
    # ------------------------------------------------------------------
    def eval(self, input_dict):
        """
        input_dict:
            y_pred_pos : Tensor [N]       — scores das triplas positivas
            y_pred_neg : Tensor [N, num_neg] — scores dos negativos
                         (já filtrados externamente, ou sem filtragem)

        Retorna {'mrr_list': Tensor([mrr_value])}.
        Compatível com a interface OGB.
        """
        p = input_dict['y_pred_pos'].view(-1, 1)          # [N, 1]
        n = input_dict['y_pred_neg']                       # [N, K]
        if n.dim() == 1:
            n = n.view(p.shape[0], -1)

        # Rank = número de negativos com score >= positivo + 1
        ranks = (n >= p).sum(dim=-1).float() + 1.0        # [N]
        mrr = (1.0 / ranks).mean()
        return {'mrr_list': torch.tensor([mrr])}

    # ------------------------------------------------------------------
    # Avaliação filtrada completa (chamada de eval_link_prediction)
    # ------------------------------------------------------------------
    def eval_filtered(self, score_fn, triples, edge_types, num_nodes, device,
                      batch_size=256):
        """
        score_fn(heads, tails, rels) → Tensor [B] de scores.
        triples  : [N, 2] LongTensor (head, tail)
        edge_types: [N]   LongTensor de tipos de relação
        num_nodes : int
        """
        assert self._filter_set is not None, \
            "Chame build_filter() antes de eval_filtered()."

        all_nodes = torch.arange(num_nodes, device=device)
        ranks = []

        for i in range(0, triples.shape[0], batch_size):
            batch_edges = triples[i: i + batch_size].to(device)   # [B, 2]
            batch_rels  = edge_types[i: i + batch_size].to(device) # [B]

            for j in range(batch_edges.shape[0]):
                h   = int(batch_edges[j, 0])
                t   = int(batch_edges[j, 1])
                r   = int(batch_rels[j])

                # Candidatos: todos os nós substituindo a cauda
                cand_tails = all_nodes                             # [num_nodes]
                heads_rep  = torch.full_like(cand_tails, h)
                rels_rep   = torch.full_like(cand_tails, r)

                with torch.no_grad():
                    scores = score_fn(heads_rep, cand_tails, rels_rep)  # [num_nodes]

                # Score da tripla positiva real
                pos_score = scores[t]

                # Máscara de filtragem: remove candidatos que são positivos
                # verdadeiros (exceto a própria tripla avaliada)
                mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
                for t_prime in range(num_nodes):
                    if t_prime != t and (h, r, t_prime) in self._filter_set:
                        mask[t_prime] = False

                filtered_scores = scores[mask]
                rank = (filtered_scores >= pos_score).sum().float() + 1.0
                ranks.append(rank)

        ranks_t = torch.stack(ranks)
        mrr = (1.0 / ranks_t).mean().item()
        return {'mrr_list': torch.tensor([mrr])}


# ---------------------------------------------------------------------------
# Catálogo de datasets e modelos
# ---------------------------------------------------------------------------

DATASETS = {
    'ogbl-collab':    {'type': 'link',  'graph_type': 'homogeneous'},
    'ogbn-proteins':  {'type': 'node',  'graph_type': 'homogeneous'},
    'ogbg-ppa':       {'type': 'graph', 'graph_type': 'homogeneous'},
    'Cora':           {'type': 'node',  'graph_type': 'homogeneous'},
    'WordNet18RR':    {'type': 'link',  'graph_type': 'heterogeneous'},
    'ogbl-biokg':     {'type': 'link',  'graph_type': 'heterogeneous'},
}

MODELS_HOMO   = ['GCN', 'GAT', 'GIN', 'GIN_batch', 'GIN_layer', 'GIN_none']
MODELS_HETERO = ['RGCN', 'RGAT']
OPTIMIZERS    = ['AdamW', 'SGD', 'Muon', 'Shampoo', 'SOAP']


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_dataset(name):
    import os

    path = f'dataset/processed_{name}.pt'
    if not os.path.exists(path):
        raise RuntimeError(
            f"Dataset {name} not prepared. "
            f"Please run `python prepare_datasets.py --dataset {name}` first."
        )

    data, split_idx, task = torch.load(path, weights_only=False)

    if name == 'WordNet18RR':
        evaluator = WordNet18RREvaluator()

        # Constrói o conjunto de filtragem com TODAS as triplas (train+val+test)
        # Formato esperado: edge [N,2] (head, tail) + edge_type [N]
        all_parts = []
        for split in ('train', 'valid', 'test'):
            e  = split_idx[split]['edge']          # [N, 2]
            et = split_idx[split]['edge_type']     # [N]
            # empilha como [N, 3]: head, tail, rel
            all_parts.append(torch.cat([e, et.unsqueeze(1)], dim=1))
        all_triples = torch.cat(all_parts, dim=0)  # [total, 3]
        evaluator.build_filter(all_triples)

    elif task == 'link':
        evaluator = LinkEvaluator(name=name)
    elif task == 'node':
        evaluator = NodeEvaluator(name=name) if name != 'Cora' else None
    elif task == 'graph':
        evaluator = GraphEvaluator(name=name)
    else:
        raise ValueError(f"Unknown task type: {task}")

    return data, split_idx, evaluator, task