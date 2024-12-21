import torch
from typing import Dict, Optional

class GrokFast(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0.0, nesterov=False,
                 alpha: float = 0.98, lamb: float = 2.0, use_grad_filter: bool = True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(GrokFast, self).__init__(params, defaults)

        self.alpha = alpha
        self.lamb = lamb
        self.use_grad_filter = use_grad_filter

        # Initialize gradient storage using self.param_groups which contains all parameters
        self.grads_ema = {id(p): torch.zeros_like(p.data) for group in self.param_groups for p in group['params'] if p.requires_grad}

        # Initialize base_optimizer using self.param_groups to ensure consistency
        self.base_optimizer = torch.optim.SGD(self.param_groups, lr, momentum=momentum, dampening=dampening, 
                                              weight_decay=weight_decay, nesterov=nesterov)

    @torch.no_grad()
    def gradfilter_ema(
        self,
        grads: Dict[int, torch.Tensor],
        alpha: float,
        lamb: float,
    ) -> Dict[int, torch.Tensor]:
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and p.grad is not None:
                    pid = id(p)
                    # Update EMA without modifying the original gradient
                    grads[pid] = grads[pid] * alpha + p.grad.data.detach() * (1 - alpha)
                    # Apply filtered gradient as an additional term
                    p.grad.data.add_(grads[pid], alpha=lamb)

        return grads

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Apply gradfilter_ema only if use_grad_filter is True
        if self.use_grad_filter:
            self.grads_ema = self.gradfilter_ema(
                self.grads_ema, self.alpha, self.lamb
            )

        self.base_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Clears the gradients of all optimized parameters."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)