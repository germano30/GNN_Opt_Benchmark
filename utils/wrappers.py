import torch
import torch.nn as nn

class NodeEmbeddingWrapper(nn.Module):
    """
    Wraps any PyG model to perfectly simulate GNNDelete's embedding behavior.
    If the dataset lacks initial continuous features (e.g. WordNet18RR), benchmark.py
    will inject integer IDs via torch.arange(). This wrapper intercepts these tensors,
    and automatically passes them through an internal lookup Embedding table before 
    forwarding the dense float vectors to the underlying Homogeneous/Heterogeneous GNN.
    
    The internal Embedding connects safely to our optimizer filtering mechanics since
    its variable name ('node_emb') contains 'embed', steering it away from Muon.
    """
    def __init__(self, gnn_model, num_nodes, hidden_channels):
        super().__init__()
        self.gnn = gnn_model
        
        # 'node_emb' is mathematically safe from Newton-Schulz due to our get_optimizer filters
        self.node_emb = nn.Embedding(num_nodes, hidden_channels)
        
    def forward(self, x, *args, **kwargs):
        # GNNDelete Trick interceptor: Integer IDs signify 'I need an embedding projection'
        # Or x is None (graph datasets with no initial node features)
        if x is None or x.dtype == torch.long or not x.is_floating_point():
            # For graph classification datasets, x might be None — use index range
            if x is None:
                # Assume this is a graph batch; we need sequential node indices
                # This is a fallback; ideally graphs should pre-index nodes
                num_nodes = args[0].max().item() + 1 if len(args) > 0 and hasattr(args[0], 'max') else 100
                x = torch.arange(num_nodes, dtype=torch.long, device=self.node_emb.weight.device)
            x = self.node_emb(x)
            if hasattr(x, 'squeeze'):
                x = x.squeeze() # Ensures [num_nodes, emb_dim] layout
        
        return self.gnn(x, *args, **kwargs)
