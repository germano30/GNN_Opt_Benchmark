import torch
import math
from muon import MuonWithAuxAdam

def get_optimizer(name, named_params, lr):
    params = [p for n, p in named_params] # Extrai apenas os tensores para os otimizadores padroes genericos
    
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr)
    elif name == 'SGD':
        return torch.optim.SGD(params, lr=lr)
    elif name == 'Muon':            
        # Filtra ativamente qualquer parametro de 'embedding' para longe do Muon, mesmo que seja 2D.
        hidden_weights = [p for n, p in named_params if p.ndim >= 2 and 'embed' not in n.lower()]
        hidden_gains_biases = [p for n, p in named_params if p.ndim < 2 or 'embed' in n.lower()]

        param_groups = []
        
        # Simula o comportamento do adjust_lr_fn="match_rms_adamw" da implementacao nativa
        for p in hidden_weights:
            A, B = p.shape[:2]
            adjusted_ratio = 0.2 * math.sqrt(max(A, B))
            
            param_groups.append(dict(
                params=[p],
                use_muon=True,
                lr=lr * adjusted_ratio,
                # O nativo decai em (1 - base_lr * wd). Aqui dividimos o wd pelo ratio p/ compensar.
                weight_decay=5e-4 / adjusted_ratio
            ))
            
        param_groups.append(
            dict(
                params=hidden_gains_biases,
                use_muon=False,
                lr=lr, # AdamW passa a usar a mesma taxa do Muon (baseline)
                betas=(0.9, 0.95),
                weight_decay=5e-4
            )
        )
        return MuonWithAuxAdam(param_groups)
    elif name == 'Shampoo':
        try:
            import torch_optimizer as optim
            return optim.Shampoo(params, lr=lr)
        except ImportError:
            raise ImportError("Please install torch_optimizer for Shampoo: pip install torch-optimizer")
    elif name == 'SOAP':
        try:
            from utils.soap import SOAP
            return SOAP(params, lr=lr)
        except ImportError:
            raise ImportError("Please ensure SOAP optimizer is available (e.g. from soap import SOAP)")
    else:
        raise ValueError(f"Unknown optimizer {name}")
