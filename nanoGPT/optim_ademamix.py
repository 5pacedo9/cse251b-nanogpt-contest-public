"""AdEMAMix: AdamW augmented with a slow EMA of the gradient.

Reference: Pagliardini et al., "The AdEMAMix Optimizer: Better, Faster, Older"
           https://arxiv.org/abs/2409.03137

Algorithm 1 (decoupled wd):
    m1 = β1 m1 + (1-β1) g            (fast EMA, bias-corrected)
    m2 = β3 m2 + (1-β3) g            (slow EMA, NOT bias-corrected, by design)
    v  = β2 v  + (1-β2) g²
    update = (m1 / (1-β1^t) + α m2) / (sqrt(v / (1-β2^t)) + ε)
    θ ← (1 - lr·λ) θ  -  lr · update

Drop-in for AdamW: same LR / wd ranges work; α=5, β3=0.9999 are the only new knobs.
"""

from __future__ import annotations

import math

import torch
from torch.optim.optimizer import Optimizer


class AdEMAMix(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"invalid lr: {lr}")
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m1, m2, v = state["m1"], state["m2"], state["v"]

                m1.mul_(beta1).add_(g, alpha=1 - beta1)
                m2.mul_(beta3).add_(g, alpha=1 - beta3)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t

                # decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                denom = (v / bc2).sqrt().add_(eps)
                update = (m1 / bc1).add_(m2, alpha=alpha)
                p.addcdiv_(update, denom, value=-lr)

        return loss


def build_ademamix_optimizer(model, lr: float, weight_decay: float, betas: tuple, device_type: str):
    """Same decay/no-decay grouping as AdamW. β = (β1, β2, β3); we pass through user β2 and pin β3=0.9999."""
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    print(f"[ademamix] decay: {len(decay)} tensors / {sum(p.numel() for p in decay):,} params")
    print(f"[ademamix] nodecay: {len(nodecay)} tensors / {sum(p.numel() for p in nodecay):,} params")
    optim_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    beta1, beta2 = betas
    return AdEMAMix(optim_groups, lr=lr, betas=(beta1, beta2, 0.9999), alpha=5.0, eps=1e-8)
