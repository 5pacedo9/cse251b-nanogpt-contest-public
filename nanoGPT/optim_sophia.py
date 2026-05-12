"""Sophia-G: Scalable second-order optimizer with Gauss-Newton-Bartlett Hessian estimator.

Reference: Liu et al., "Sophia: A Scalable Stochastic Second-order Optimizer
           for Language Model Pre-training" (2023), https://arxiv.org/abs/2305.14342
           Repo: https://github.com/Liuhong99/Sophia

Key idea: maintain a per-parameter EMA of the *diagonal Hessian estimate* (h_t)
and clip the AdamW-style update by max(min(m_t / max(h_t, eps), ρ), -ρ), where ρ is a
clipping threshold. Hessian is updated every k steps via the Gauss-Newton-Bartlett
trick: sample y_hat ~ p(·|x; θ), compute g_hat = ∇log p(y_hat), then h ≈ B · g_hat²
(scaled by batch size B).

This implementation is the simplified "Sophia-G" variant. We expose `update_hessian()`
as a method to be called externally every k steps (after the model has produced
logits) so the train loop can perform the GNB sampling.
"""

from __future__ import annotations

import math

import torch
from torch.optim.optimizer import Optimizer


class SophiaG(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.0,
        eps: float = 1e-12,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= rho:
            raise ValueError(f"invalid rho: {rho}")
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self, bs: int):
        """Update diagonal Hessian estimate (call this after a separate y_hat-sampled bwd pass).

        Assumes the gradients in p.grad currently come from the GNB sampling step
        (∇log p(y_hat | x; θ), with y_hat ~ p(·|x;θ)). Per Algorithm 2:
            h_t ← β2 h_{t-1} + (1-β2) · bs · g_hat ⊙ g_hat
        """
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p)
                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=(1 - beta2) * bs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            wd = group["weight_decay"]
            eps = group["eps"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p)
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1

                m = state["exp_avg"]
                h = state["hessian"]

                m.mul_(beta1).add_(g, alpha=1 - beta1)

                # decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Sophia update: clip(m / max(h, eps), [-rho, rho])
                ratio = m / h.clamp(min=eps)
                ratio.clamp_(min=-rho, max=rho)
                p.add_(ratio, alpha=-lr)

        return loss


def build_sophia_optimizer(
    model,
    lr: float,
    weight_decay: float,
    betas: tuple,
    device_type: str,
    rho: float = 0.04,
):
    """Same decay/no-decay grouping as AdamW.

    Sophia's recommended LR is similar to AdamW (often 1-3× lower); for 96M / 10B token
    we start at AdamW's LR (lr arg) and let Phase 2 sweep find the right peak.
    """
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    print(f"[sophia] decay: {len(decay)} tensors / {sum(p.numel() for p in decay):,} params")
    print(f"[sophia] nodecay: {len(nodecay)} tensors / {sum(p.numel() for p in nodecay):,} params")
    optim_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return SophiaG(optim_groups, lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
