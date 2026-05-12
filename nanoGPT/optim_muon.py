"""Muon: orthogonalized momentum SGD for hidden 2D matrices.

Reference: Keller Jordan, https://github.com/KellerJordan/Muon
           and the nanoGPT-speedrun usage pattern.

Muon is applied ONLY to "hidden" 2D weight matrices (attn / mlp linears).
Embeddings (wte, wpe), output head (lm_head), 1D weights (LayerNorm, biases),
and any tied parameters are routed to AdamW. We expose a `MuonAdamW` wrapper
that holds both sub-optimizers and forwards step / zero_grad / state_dict.

Per-group LR scheduling:
  Each param_group records `initial_lr` (the base LR it was constructed with).
  train.py's cosine scheduler should set `pg['lr'] = pg['initial_lr'] * lr_scale`,
  where lr_scale = current_lr / nominal_max_lr — see train.py lr-setting block.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power of G (semi-orthogonal factor of G).

    Quintic iteration with Karpathy/Jordan coefficients (a, b, c) tuned for fast convergence
    on bf16. Operates on bf16 internally then casts back to G.dtype.
    """
    assert G.ndim == 2, f"expected 2D matrix, got shape {tuple(G.shape)}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, initial_lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise RuntimeError(
                        f"Muon expects 2D params; got shape {tuple(p.shape)}. "
                        f"Route this param to AdamW instead."
                    )
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                # Scale to compensate for non-square matrices: sqrt(max(1, fan_out / fan_in)).
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(update, alpha=-lr * scale)

        return loss


class MuonAdamW:
    """Wrapper that holds [Muon, AdamW] and forwards optimizer ops to both.

    Exposes `param_groups` as the concatenation of both sub-optimizers' groups so
    train.py's lr-setting loop sees every group. Each group has 'initial_lr' stamped
    at construction so per-group cosine scheduling works.
    """

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW):
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self):
        # Mutating returned dicts (e.g. setting 'lr') updates the underlying optimizers
        # because dicts are returned by reference.
        return list(self.muon.param_groups) + list(self.adamw.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, sd):
        self.muon.load_state_dict(sd["muon"])
        self.adamw.load_state_dict(sd["adamw"])


def _split_params(model):
    """Hidden 2D weights → Muon; embeddings, output head, 1D weights → AdamW."""
    muon_params = []
    adamw_decay = []
    adamw_nodecay = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_2d = p.dim() == 2
        is_embed_or_head = ("wte" in n) or ("wpe" in n) or ("lm_head" in n)
        if is_2d and not is_embed_or_head:
            muon_params.append((n, p))
        elif p.dim() >= 2:
            adamw_decay.append((n, p))
        else:
            adamw_nodecay.append((n, p))
    return muon_params, adamw_decay, adamw_nodecay


def build_muon_optimizer(
    model,
    muon_lr: float,
    adamw_lr: float,
    weight_decay: float,
    betas: tuple,
    device_type: str,
) -> MuonAdamW:
    muon_pairs, adamw_decay_pairs, adamw_nodecay_pairs = _split_params(model)
    muon_params = [p for _, p in muon_pairs]
    adamw_decay = [p for _, p in adamw_decay_pairs]
    adamw_nodecay = [p for _, p in adamw_nodecay_pairs]

    print(f"[muon] hidden 2D (Muon group): {len(muon_params)} tensors / "
          f"{sum(p.numel() for p in muon_params):,} params @ lr={muon_lr}")
    print(f"[muon] embed+head decay (AdamW group): {len(adamw_decay)} tensors / "
          f"{sum(p.numel() for p in adamw_decay):,} params @ lr={adamw_lr}")
    print(f"[muon] nodecay (AdamW group): {len(adamw_nodecay)} tensors / "
          f"{sum(p.numel() for p in adamw_nodecay):,} params @ lr={adamw_lr}")

    muon = Muon(muon_params, lr=muon_lr, momentum=0.95, nesterov=True, ns_steps=5)

    fused = device_type == "cuda"
    adamw_groups = [
        {"params": adamw_decay, "weight_decay": weight_decay, "initial_lr": adamw_lr},
        {"params": adamw_nodecay, "weight_decay": 0.0, "initial_lr": adamw_lr},
    ]
    adamw = torch.optim.AdamW(adamw_groups, lr=adamw_lr, betas=betas, fused=fused)
    return MuonAdamW(muon, adamw)
