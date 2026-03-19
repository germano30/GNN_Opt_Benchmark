import argparse
import time
import torch
import warnings
import numpy as np

# Custom torch initialization overrides
_original_load = torch.load
def custom_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = custom_load

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import torch.distributed as dist
dist.get_world_size = lambda *args, **kwargs: 1
dist.get_rank = lambda *args, **kwargs: 0
dist.all_gather = lambda *args, **kwargs: None

from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage

try:
    torch.serialization.safe_globals([
        Data, DataEdgeAttr, DataTensorAttr, GlobalStorage, NodeStorage, EdgeStorage
    ])
except AttributeError:
    pass

import random

# Importacoes dos nossos modulos especializados extraidos
from models.gat import GAT
from models.gcn import GCN
from models.rgat import RGAT
from models.rgcn import RGCN
from models.gin import GIN

from utils.data import DATASETS, MODELS_HOMO, MODELS_HETERO, OPTIMIZERS, load_dataset
from utils.optimizers import get_optimizer
from utils.predictors import LinkPredictor, NodePredictor, GraphPredictor
from utils.training import train_link_prediction, eval_link_prediction
from utils.training import train_node_classification, eval_node_classification
from utils.training import train_graph_classification, eval_graph_classification
from utils.wrappers import NodeEmbeddingWrapper


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def get_model(name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, num_relations=None):
    if graph_type == 'homogeneous':
        if name == 'GCN': return GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        elif name == 'GAT': return GAT(in_channels, hidden_channels, out_channels, num_layers, dropout)
        elif name == 'GIN': return GIN(in_channels, hidden_channels, out_channels, num_layers, dropout)
    else:
        if name == 'RGCN': return RGCN(in_channels, hidden_channels, out_channels, num_layers, dropout, num_relations)
        elif name == 'RGAT': return RGAT(in_channels, hidden_channels, out_channels, num_layers, dropout, num_relations)

def run_experiment(dataset_name, model_name, optimizer_name, epochs=10, lr=0.01, hidden_channels=256, num_layers=3, dropout=0.5, batch_size=1024):
    set_seed(42) # Garante que todo otimizador inicie com exatamente os MESMOS pesos sorteados
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_info = load_dataset(dataset_name)
    task = DATASETS[dataset_name]['type']
    graph_type = DATASETS[dataset_name]['graph_type']
    
    if task == 'link':
        data, split_edge, evaluator, _ = data_info
        num_relations = data.edge_attr.max().item() + 1 if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1
        
        # O TRUQUE DO GNNDELETE
        if hasattr(data, 'x') and data.x is not None:
            in_channels = data.x.size(1)
            use_node_embedding = False
        else:
            in_channels = hidden_channels
            data.x = torch.arange(data.num_nodes, dtype=torch.long)
            use_node_embedding = True
            
        out_channels = hidden_channels
        gnn = get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, num_relations)
        predictor = LinkPredictor(hidden_channels, hidden_channels, 1, 2, dropout)

    elif task == 'node':
        data, split_idx, evaluator, _ = data_info
        num_relations = data.edge_attr.max().item() + 1 if hasattr(data, 'edge_attr') and data.edge_attr is not None else 1
        
        # O TRUQUE DO GNNDELETE
        if hasattr(data, 'x') and data.x is not None:
            in_channels = data.x.size(1)
            use_node_embedding = False
        else:
            in_channels = hidden_channels
            data.x = torch.arange(data.num_nodes, dtype=torch.long)
            use_node_embedding = True
            
        out_channels = hidden_channels
        gnn = get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, num_relations)
        num_classes = data.y.max().item() + 1 if dataset_name != 'Cora' else 7
        predictor = NodePredictor(hidden_channels, num_classes, 2, dropout)

    else:
        # Graph Task
        dataset, split_idx, evaluator, _ = data_info
        if hasattr(dataset, 'num_node_features') and dataset.num_node_features > 0:
            in_channels = dataset.num_node_features
            use_node_embedding = False
        elif hasattr(dataset[0], 'x') and dataset[0].x is not None:
            in_channels = dataset[0].x.size(1)
            use_node_embedding = False
        else:
            in_channels = hidden_channels
            use_node_embedding = True
        
        out_channels = hidden_channels
        gnn = get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, 1)
        num_classes = dataset.num_classes
        predictor = GraphPredictor(hidden_channels, num_classes, 2, dropout)
    
    # WRAP O MODELO: Blindagem e Injecao Automatica do PyTorch Embedding
    if use_node_embedding:
        print(f"\n[AVISO] Aplicando o GNNDelete Wrapper Trick ao dataset cru: '{dataset_name}'!\n")
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else 100000 # Fallback para graph tasks
        gnn = NodeEmbeddingWrapper(gnn, num_nodes, hidden_channels)

    gnn.to(device)
    predictor.to(device)
    
    named_params = list(gnn.named_parameters()) + list(predictor.named_parameters())
    optimizer = get_optimizer(optimizer_name, named_params, lr)
    
    train_losses = []
    eval_scores = []
    metric_name = "Score"
    
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
        print(f'Epoch {epoch+1:03d}/{epochs}: Train Loss {loss:.4f} | Val/Test {metric_name}: {score:.4f}')
        
    training_time = time.time() - start_time
    return eval_scores[-1], training_time, train_losses, eval_scores, metric_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys())
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--optimizer', type=str, required=True, choices=OPTIMIZERS + ['all'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    
    if DATASETS[args.dataset]['graph_type'] == 'homogeneous' and args.model not in MODELS_HOMO:
        raise ValueError("Invalid model for homogeneous graph")
    if DATASETS[args.dataset]['graph_type'] == 'heterogeneous' and args.model not in MODELS_HETERO:
        raise ValueError("Invalid model for heterogeneous graph")
    
    if args.optimizer == 'all':
        opts_to_run = OPTIMIZERS
    else:
        opts_to_run = [args.optimizer]
        
    results = {}
    for opt in opts_to_run:
        print(f"\n[{opt}] Iniciando treinamento no dataset {args.dataset} com modelo {args.model}...")
        try:
            final_score, time_taken, losses, scores, metric_name = run_experiment(
                args.dataset, args.model, opt, args.epochs, args.lr, batch_size=args.batch_size
            )
            print(f"> [{opt}] Tempo: {time_taken:.2f}s | Final {metric_name}: {final_score:.4f}")
            results[opt] = {
                'losses': losses,
                'scores': scores,
                'final_score': final_score,
                'time': time_taken,
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
            plt.plot(range(1, args.epochs + 1), hist['losses'], label=f"{opt}")
        plt.title(f'Treinamento Loss ({args.dataset} / {args.model})')
        plt.xlabel('Epoca')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Score
        plt.subplot(1, 2, 2)
        metric = list(results.values())[0]['metric_name']
        for opt, hist in results.items():
            plt.plot(range(1, args.epochs + 1), hist['scores'], label=f"{opt} ({metric} final={hist['final_score']:.3f})")
        plt.title(f'Score de Validacao/Teste ({metric})')
        plt.xlabel('Epoca')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"comparativo_{args.dataset}_{args.model}.png"
        plt.savefig(plot_path)
        print(f"\nGrafico comparativo salvo com sucesso em: {plot_path}")
