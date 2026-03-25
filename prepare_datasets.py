import argparse
import os
import torch

_original_load = torch.load
def _custom_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _custom_load

from torch_geometric.datasets import Planetoid, WordNet18RR
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from utils.data import DATASETS


def prepare_dataset(name):
    print(f"Preparing dataset: {name}")
    os.makedirs('dataset', exist_ok=True)

    if name == 'Cora':
        dataset = Planetoid(root='dataset/Cora', name='Cora')
        data = dataset[0]
        split_idx = {
            'train': data.train_mask,
            'valid': data.val_mask,
            'test':  data.test_mask,
        }
        task = 'node'

    elif DATASETS[name]['type'] == 'link':
        if name == 'WordNet18RR':
            dataset = WordNet18RR(root='dataset/WordNet18RR')
            data = dataset[0]
            train_edge   = data.edge_index[:, data.train_mask]
            train_etype  = data.edge_type[data.train_mask]
            val_edge     = data.edge_index[:, data.val_mask]
            val_etype    = data.edge_type[data.val_mask]
            test_edge    = data.edge_index[:, data.test_mask]
            test_etype   = data.edge_type[data.test_mask]

            split_idx = {
                'train': {
                    'edge':      train_edge.t(),   # [E_train, 2]
                    'edge_type': train_etype,       # [E_train]
                },
                'valid': {
                    'edge':      val_edge.t(),      # [E_val, 2]
                    'edge_type': val_etype,
                },
                'test': {
                    'edge':      test_edge.t(),     # [E_test, 2]
                    'edge_type': test_etype,
                },
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
        data = dataset
        split_idx = dataset.get_idx_split()
        task = 'graph'

    else:
        raise ValueError(f"Unknown dataset type for {name}")

    out_path = f'dataset/processed_{name}.pt'
    torch.save((data, split_idx, task), out_path)
    print(f"Saved {name} to {out_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=list(DATASETS.keys()) + ['all'],
    )
    args = parser.parse_args()

    if args.dataset == 'all':
        for ds in DATASETS.keys():
            prepare_dataset(ds)
    else:
        prepare_dataset(args.dataset)