import torch
import torch.nn as nn

class NodeEmbeddingWrapper(nn.Module):
    """
    Wraps any PyG model to perfectly simulate GNNDelete's embedding behavior.
    If the dataset lacks initial continuous features (ogbl-biokg, WordNet), benchmark.py 
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
        if x.dtype == torch.long or not x.is_floating_point():
            x = self.node_emb(x)
            if hasattr(x, 'squeeze'):
                x = x.squeeze() # Ensures [num_nodes, emb_dim] layout
        
        return self.gnn(x, *args, **kwargs)
