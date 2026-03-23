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

def run_experiment(dataset_name, model_name, optimizer_name, seed=42, epochs=10, lr=0.01, 
                   weight_decay=None, patience=None,
                   hidden_channels=256, num_layers=3, dropout=0.5, batch_size=1024):
    set_seed(seed) # Garante que todo otimizador inicie com os MESMOS pesos sorteados por rodada
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_info = load_dataset(dataset_name)
    task = DATASETS[dataset_name]['type']
    graph_type = DATASETS[dataset_name]['graph_type']
    
    if task == 'link':
        data, split_edge, evaluator, _ = data_info
        edge_t = getattr(data, 'edge_type', getattr(data, 'edge_attr', None))
        num_relations = edge_t.max().item() + 1 if edge_t is not None else 1
        
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
        edge_t = getattr(data, 'edge_type', getattr(data, 'edge_attr', None))
        num_relations = edge_t.max().item() + 1 if edge_t is not None else 1
        
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
        pass
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else 100000 # Fallback para graph tasks
        gnn = NodeEmbeddingWrapper(gnn, num_nodes, hidden_channels)

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
    parser.add_argument('--config', type=str, default=None, help='Caminho para o arquivo YAML de configuracao')
    parser.add_argument('--dataset', type=str, default='Cora', choices=list(DATASETS.keys()) + [None], help='(Opcional) Dataset se nao usar YAML')
    parser.add_argument('--model', type=str, default='GCN', help='(Opcional) Modelo se nao usar YAML')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=OPTIMIZERS + ['all', None], help='(Opcional) Optimizer se nao usar YAML')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--runs', type=int, default=1, help='Numero de execucoes com seeds diferentes')
    args = parser.parse_args()
    
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

    # Main Grid-Search Loop
    for dataset in datasets_list:
        if dataset not in DATASETS:
            print(f"Dataset inválido: {dataset}. Pulando...")
            continue
            
        for model in models_list:
            if DATASETS[dataset]['graph_type'] == 'homogeneous' and model not in MODELS_HOMO:
                print(f"Modelo {model} invalido para dataset homogeneo {dataset}. Pulando...")
                continue
            if DATASETS[dataset]['graph_type'] == 'heterogeneous' and model not in MODELS_HETERO:
                print(f"Modelo {model} invalido para dataset heterogeneo {dataset}. Pulando...")
                continue
                
            results = {}
            for opt in optimizers_list:
                print(f"\n[{opt}] Iniciando treinamento no dataset {dataset} com modelo {model} ({runs_cfg} runs)...")
                opt_final_scores = []
                opt_losses = []
                opt_scores = []
                time_takens = []
                
                try:
                    for run_idx in range(runs_cfg):
                        seed = 42 + run_idx
                        final_score, time_taken, losses, scores, metric_name = run_experiment(
                            dataset, model, opt, seed, epochs_cfg, lr_cfg,
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
                    import traceback
                    print(f"> Erro ao rodar otimizador {opt}: {str(e)}")
                    traceback.print_exc()
                    
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
                    epochs_range = range(1, epochs_cfg + 1)
                    line, = plt.plot(epochs_range, mean_loss, label=f"{opt}")
                    if runs_cfg > 1:
                        plt.fill_between(epochs_range, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2, color=line.get_color())
                plt.title(f'Treinamento Loss ({dataset} / {model})')
                plt.xlabel('Epoca')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Subplot 2: Score
                plt.subplot(1, 2, 2)
                metric = list(results.values())[0]['metric_name']
                for opt, hist in results.items():
                    mean_score = hist['scores'].mean(axis=0)
                    std_score = hist['scores'].std(axis=0)
                    epochs_range = range(1, epochs_cfg + 1)
                    line, = plt.plot(epochs_range, mean_score, label=f"{opt} ({metric} final={hist['mean_final_score']:.3f}±{hist['std_final_score']:.3f})")
                    if runs_cfg > 1:
                        plt.fill_between(epochs_range, mean_score - std_score, mean_score + std_score, alpha=0.2, color=line.get_color())
                plt.title(f'Score de Validacao/Teste ({metric})')
                plt.xlabel('Epoca')
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = f"comparativo_v2_{dataset}_{model}.png"
                plt.savefig(plot_path)
                print(f"\nGrafico comparativo salvo com sucesso em: {plot_path}")
