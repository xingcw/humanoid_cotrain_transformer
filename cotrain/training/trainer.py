"""Training step + optimizer construction (PROJECT_PLAN_1.md §5).

§8 step 9 spec: 'Run a 10-step train_step on synthetic batches with the
real nnx.jit + sharding. Shape and sharding bugs surface here, where the
iteration is fast.'

This module wires:
  - the §5.3 default optimizer (warmup-cosine adamw + global-norm clip),
  - `train_step` under `nnx.jit`,
  - a `synthetic_batch` helper for the 10-step smoke test.

DINO is intentionally *out of the gradient tape* in v1: the trainer holds
the encoder separately and feeds vis features as a regular array into the
transformer's batch dict. `jax.lax.stop_gradient` on the vis features
keeps the encoder frozen even when wrapped inside the same jit
(§3.2). For §8.9 we don't run the encoder at all — synthetic batches
include synthetic vis features. The encoder wires in at §8.10.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from cotrain.data.schemas import (
    DEFAULT_D_A,
    DEFAULT_D_H,
    DEFAULT_D_P,
)
from cotrain.models.heads.projection import DINO_FEATURE_DIM
from cotrain.models.transformer import CoTrainTransformer, TransformerConfig
from cotrain.training.losses import LossWeights, compute_loss


# --- optimizer -----------------------------------------------------------

@dataclass(frozen=True)
class OptimizerConfig:
    """§5.3 defaults. lr_peak scales with sqrt(batch_ratio) at runtime."""
    lr_peak: float = 3e-4
    warmup_steps: int = 2_000
    decay_steps: int = 200_000
    weight_decay: float = 0.1
    b1: float = 0.9
    b2: float = 0.95
    grad_clip_norm: float = 1.0


def make_lr_schedule(cfg: OptimizerConfig) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr_peak,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.decay_steps,
    )


def make_optimizer(model: nnx.Module, cfg: OptimizerConfig) -> nnx.Optimizer:
    schedule = make_lr_schedule(cfg)
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.b1,
            b2=cfg.b2,
            weight_decay=cfg.weight_decay,
        ),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# --- training step -------------------------------------------------------

@nnx.jit
def train_step(
    model: CoTrainTransformer,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
):
    """One optimizer update. Returns (loss, aux) — model + optimizer state
    are mutated in place per NNX semantics."""

    def _loss_fn(model: CoTrainTransformer):
        preds = model(batch, deterministic=False)
        loss, aux = compute_loss(preds, batch, weights=LossWeights())
        return loss, aux

    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(model)
    optimizer.update(model, grads)
    return loss, aux


@nnx.jit
def eval_step(
    model: CoTrainTransformer,
    batch: dict[str, jnp.ndarray],
):
    """Loss-only forward (deterministic=True). For per-step validation."""
    preds = model(batch, deterministic=True)
    return compute_loss(preds, batch, weights=LossWeights())


# --- synthetic data ------------------------------------------------------

def synthetic_batch(
    *,
    B: int,
    T: int,
    seed: int,
    D_p: int = DEFAULT_D_P,
    D_h: int = DEFAULT_D_H,
    D_a: int = DEFAULT_D_A,
    dino_dim: int = DINO_FEATURE_DIM,
    num_phases: int = 5,
    contact_dim: int = 3,
    box_dim: int = 7,
) -> dict[str, jnp.ndarray]:
    """Make a batch with all the keys the model + loss require.

    Used by the §8.9 10-step smoke test. The vis feature is a plain
    Gaussian array — no DINO. The encoder integration test lives in §8.10.
    """
    rng = np.random.default_rng(seed)
    return {
        "vis":         jnp.asarray(rng.normal(size=(B, T, dino_dim)).astype(np.float32)),
        "state_robot": jnp.asarray(rng.normal(size=(B, T, D_p)).astype(np.float32)),
        "state_human": jnp.asarray(rng.normal(size=(B, T, D_h)).astype(np.float32)),
        "box":         jnp.asarray(np.concatenate([
                            rng.normal(size=(B, T, 3)).astype(np.float32),
                            _unit_quats(rng, B, T),
                        ], axis=-1)),
        "phase":       jnp.asarray(rng.integers(0, num_phases, size=(B, T, 1)).astype(np.int8)),
        "contact":     jnp.asarray(rng.integers(0, 2, size=(B, T, contact_dim)).astype(np.float32)),
        "action":      jnp.asarray(rng.normal(size=(B, T, D_a)).astype(np.float32)),
        "source_mask": jnp.asarray([(i % 2 == 0) for i in range(B)], dtype=bool),
    }


def _unit_quats(rng: np.random.Generator, B: int, T: int) -> np.ndarray:
    q = rng.normal(size=(B, T, 4)).astype(np.float32)
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


def tiny_config(T: int = 4) -> TransformerConfig:
    """Tiny transformer config for tests/CI — d_model 64, 2 layers, 4 heads."""
    return TransformerConfig(
        d_model=64, n_layers=2, n_heads=4, T=T,
        D_p=DEFAULT_D_P, D_h=DEFAULT_D_H, D_a=DEFAULT_D_A,
        dino_dim=DINO_FEATURE_DIM, dropout_rate=0.0,
    )
