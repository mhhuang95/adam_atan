from __future__ import annotations
from typing import Callable

import torch
from torch import atan2, sqrt
from torch.optim.optimizer import Optimizer

def exists(val):
    return val is not None

class AdamAtan2(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay = 0.,
        a = 1.27,
        b = 1.,
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.

        self._init_lr = lr

        defaults = dict(
            lr = lr,
            betas = betas,
            a = a,
            b = b,
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

                grad, lr, wd, beta1, beta2, a, b, state, init_lr = p.grad, group['lr'], group['weight_decay'], *group['betas'], group['a'], group['b'], self.state[p], self._init_lr

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

                denom = exp_avg_sq.mul(b * b / bias_correction2).sqrt()
                update = exp_avg.div(bias_correction1).atan2_(denom)

                p.add_(update, alpha = -lr * a)

                state['steps'] = steps

        return loss