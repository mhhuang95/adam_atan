from __future__ import annotations
from typing import Callable

import torch
from torch.optim.optimizer import Optimizer

def exists(val):
    return val is not None

class Adopt(Optimizer):
    """
    Implementation of Adopt optimizer in https://arxiv.org/pdf/2411.02853
    """
    def __init__(
        self,
        params,
        lr = 1e-4,
        eps = 1e-6,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay = 0.,
        decoupled_wd = True,
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.

        self._init_lr = lr
        self.decoupled_wd = decoupled_wd

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

                grad, lr, wd, beta1, beta2, eps, state, init_lr = p.grad, group['lr'], group['weight_decay'], *group['betas'], group['eps'], self.state[p], self._init_lr

                if self.decoupled_wd:
                    wd /= init_lr

                # weight decay

                if wd > 0.:
                    p.mul(1. - lr * wd)
                
                
                # init state if needed

                if len(state) == 0:
                    state['steps'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = grad * grad

                exp_avg, exp_avg_sq, steps = state['exp_avg'], state['exp_avg_sq'], state['steps']

                # do nothing in the first step

                if steps == 0:
                    state['steps'] += 1
                    continue

                update = grad.div(exp_avg_sq.sqrt().clamp(min = eps))

                #clip with t ^ 0.25

                clip_value = steps ** 0.25
                update.clamp(min = -clip_value, max = clip_value)

                exp_avg.lerp_(update, 1. - beta1)

                p.add_(exp_avg, alpha = -lr)

                exp_avg_sq.lerp_(grad * grad, 1. - beta2)                

                state['steps'] += 1

        return loss