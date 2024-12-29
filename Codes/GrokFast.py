import torch
from typing import Dict, Optional, Deque, Literal
from collections import deque

class GrokFast(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0.0, nesterov=False,
                 window_size: int = 100, lamb: float = 5.0, filter_type: Literal['mean', 'sum'] = 'mean',
                 use_grad_filter: bool = True):
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

        self.window_size = window_size
        self.lamb = lamb
        self.filter_type = filter_type
        self.use_grad_filter = use_grad_filter

        self.grads_ma = {id(p): deque(maxlen=window_size) for group in self.param_groups for p in group['params'] if p.requires_grad}

        self.base_optimizer = torch.optim.SGD(self.param_groups, lr, momentum=momentum, dampening=dampening, 
                                              weight_decay=weight_decay, nesterov=nesterov)

    @torch.no_grad()
    def gradfilter_ma(
        self,
        grads: Dict[int, Deque[torch.Tensor]],
        window_size: int,
        lamb: float,
        filter_type: Literal['mean', 'sum'],
        warmup: bool = True,
        trigger: bool = False, 
    ) -> Dict[int, Deque[torch.Tensor]]:
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and p.grad is not None:
                    pid = id(p)
                    grads[pid].append(p.grad.data.detach())

                    if not warmup or len(grads[pid]) == window_size and not trigger:
                        if filter_type == "mean":
                            avg = sum(grads[pid]) / len(grads[pid])
                        elif filter_type == "sum":
                            avg = sum(grads[pid])
                        else:
                            raise ValueError(f"Unrecognized filter_type {filter_type}")
                        p.grad.data.add_(avg, alpha=lamb)

        return grads

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.use_grad_filter:
            self.grads_ma = self.gradfilter_ma(
                self.grads_ma, self.window_size, self.lamb, self.filter_type
            )

        self.base_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Clears the gradients of all optimized parameters."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
