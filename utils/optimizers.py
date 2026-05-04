import torch
import math
from muon import MuonWithAuxAdam

def get_optimizer(name, named_params, lr, weight_decay=None):
    params = [p for n, p in named_params] # Extrai apenas os tensores para os otimizadores padroes genericos
    
    if name == 'AdamW':
        wd = weight_decay if weight_decay is not None else 1e-2
        return torch.optim.AdamW(params, lr=0.001, weight_decay=wd)
    elif name == 'SGD':
        wd = weight_decay if weight_decay is not None else 0
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    elif name == 'Muon':            
        # Filtra ativamente qualquer parametro de 'embedding' para longe do Muon, mesmo que seja 2D.
        hidden_weights = [p for n, p in named_params if p.ndim == 2 and 'embed' not in n.lower()]
        hidden_gains_biases = [p for n, p in named_params if p.ndim != 2 or 'embed' in n.lower()]

        param_groups = []
        
        # Muon: após ortogonalização, pesos têm magnitude ~1, então precisa LR MENOR
        # Usa adjust_lr_fn="match_rms_adamw": adjusted_ratio = 0.2 * sqrt(max(A, B))
        # DIVIDE o LR pelo adjusted_ratio (não multiplica!)
        # print(f"[Muon Config] Base LR: {lr}")
        for i, p in enumerate(hidden_weights):
            A, B = p.shape[:2]
            adjusted_ratio = 0.2 * math.sqrt(max(A, B))
            effective_lr = lr * adjusted_ratio 
            # print(f"  Layer {i}: shape={p.shape}, A={A}, B={B}, ratio={adjusted_ratio:.4f}, effective_lr={effective_lr:.6f}")
            
            param_groups.append(dict(
                params=[p],
                use_muon=True,
                lr=0.01,
                weight_decay=(weight_decay if weight_decay is not None else 5e-4)
            ))
        
        # print(f"  Adam (biases): lr={lr}")    
        param_groups.append(
            dict(
                params=hidden_gains_biases,
                use_muon=False,
                lr=0.001,
                betas=(0.9, 0.95),
                weight_decay=(weight_decay if weight_decay is not None else 5e-4)
            )
        )
        return MuonWithAuxAdam(param_groups)
    elif name == 'Shampoo':
        try:
            import torch_optimizer as optim
            wd = weight_decay if weight_decay is not None else 0
            return optim.Shampoo(params, lr=lr, weight_decay=wd, update_freq=10, momentum=0.0)
        except ImportError:
            raise ImportError("Please install torch_optimizer for Shampoo: pip install torch-optimizer")
    elif name == 'SOAP':
        try:
            from utils.soap import SOAP

            soap_lr = 3e-3  # Use o default do paper
            wd = weight_decay if weight_decay is not None else 5e-4  # Reduz weight decay
            return SOAP(params, lr=soap_lr, weight_decay=.01, precondition_frequency=10)
        except ImportError:
            raise ImportError("Please ensure SOAP optimizer is available (e.g. from soap import SOAP)")
    else:
        raise ValueError(f"Unknown optimizer {name}")
