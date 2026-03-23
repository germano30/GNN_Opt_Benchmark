import argparse
import os
import torch
from torch_geometric.datasets import Planetoid, WordNet18RR

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset

from utils.data import DATASETS

def negative_sampling_kg(edge_index, edge_type):
    '''Generate negative samples but keep the node type the same (GNNDelete_tmp inspired)'''
    edge_index_copy = edge_index.clone()
    for et in edge_type.unique():
        mask = (edge_type == et)
        old_source = edge_index_copy[0, mask]
        new_index = torch.randperm(old_source.shape[0])
        new_source = old_source[new_index]
        edge_index_copy[0, mask] = new_source
    return edge_index_copy

def prepare_dataset(name):
    print(f"Preparing dataset: {name}")
    os.makedirs('dataset', exist_ok=True)
    
    if name == 'Cora':
        dataset = Planetoid(root='dataset/Cora', name='Cora')
        data = dataset[0]
        split_idx = {'train': data.train_mask, 'valid': data.val_mask, 'test': data.test_mask}
        task = 'node'
        
    elif DATASETS[name]['type'] == 'link':
        if name == 'WordNet18RR':
            dataset = WordNet18RR(root='dataset/WordNet18RR')
            data = dataset[0]
            val_edge_index = data.edge_index[:, data.val_mask]
            test_edge_index = data.edge_index[:, data.test_mask]
            
            val_neg_edge_index = negative_sampling_kg(val_edge_index, data.edge_type[data.val_mask])
            test_neg_edge_index = negative_sampling_kg(test_edge_index, data.edge_type[data.test_mask])
            
            split_idx = {
                'train': {'edge': data.edge_index[:, data.train_mask].t()},
                'valid': {'edge': val_edge_index.t(), 'edge_neg': val_neg_edge_index.t()},
                'test': {'edge': test_edge_index.t(), 'edge_neg': test_neg_edge_index.t()}
            }
            task = 'link'
        else:
            dataset = PygLinkPropPredDataset(name=name, root='dataset')
            data = dataset[0]
            split_idx = dataset.get_edge_split()
            task = 'link'
            
    elif DATASETS[name]['type'] == 'node':
        dataset = PygNodePropPredDataset(name=name, root='dataset')
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        task = 'node'
        
    elif DATASETS[name]['type'] == 'graph':
        dataset = PygGraphPropPredDataset(name=name, root='dataset')
        data = dataset # For graph tasks, data is the entire dataset object
        split_idx = dataset.get_idx_split()
        task = 'graph'
    else:
        raise ValueError(f"Unknown dataset type for {name}")

    out_path = f'dataset/processed_{name}.pt'
    torch.save((data, split_idx, task), out_path)
    print(f"Saved {name} to {out_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', choices=list(DATASETS.keys()) + ['all'])
    args = parser.parse_args()
    
    if args.dataset == 'all':
        for ds in DATASETS.keys():
            prepare_dataset(ds)
    else:
        prepare_dataset(args.dataset)
