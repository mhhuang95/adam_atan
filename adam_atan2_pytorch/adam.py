from __future__ import annotations
from typing import Callable

import torch
from torch.optim.optimizer import Optimizer

def exists(val):
    return val is not None

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        eps = 1e-8,
        weight_decay = 0.,
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.

        defaults = dict(
            lr = lr,
            betas = betas,
            eps = eps,
            weight_decay = weight_decay,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self, 
        closure: Callable | None = None
    ):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state, eps = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p], self.eps

                # weight decay

                if wd > 0.:
                    p.mul(1. - lr * wd)
                
                # init state if needed

                if len(state) == 0:
                    state['steps'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                
                exp_avg, exp_avg_sq, steps = state['exp_avg'], state['exp_avg_sq'], state['steps']

                steps += 1

                #bias corrections

                bias_correction1 = 1. - beta1 ** steps
                bias_correction2 = 1. - beta2 ** steps

                #EMA

                exp_avg.lerp_(grad, 1. - beta1)
                exp_avg_sq.lerp_(grad * grad, 1. - beta2)

                denom = exp_avg_sq.div(bias_correction2).sqrt()
                num = exp_avg.div(bias_correction1).add(eps)
                update = num / denom

                p.add_(update, alpha = -lr)

                state['steps'] = steps

        return loss