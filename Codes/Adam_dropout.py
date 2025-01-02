import torch
from typing import Dict, Optional

class Adam_dropout(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), dropout: float = 0.1):
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Invalid dropout probability: {}".format(dropout))
        
        defaults = dict(lr=lr, betas=betas, dropout=dropout)
        super(Adam_dropout, self).__init__(params, defaults)

        self.base_optimizer = torch.optim.Adam(self.param_groups, lr, betas=betas)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                if group['dropout'] > 0.0 and self.training:
                    mask = torch.rand_like(p.grad, dtype=torch.float32) > group['dropout']
                    p.grad.mul_(mask)

        self.base_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Clears the gradients of all optimized parameters."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)