"""Per-modality losses and the §3.6 total objective.

Per PROJECT_PLAN_1.md §2.4 the prediction target for a token at time t is
the *next* timestep's same-slot value, so we drop the last position when
forming (pred, target) pairs:

    pred:   slot[:, :T-1]   target: slot[:, 1:T]

The loss masks coming out of `cotrain.training.masking` are (B, T) and
carry the §3.4 / §3.5 "vis is never supervised; action is robot-only;
bridge is always" semantics. The loss module multiplies them (also sliced
to T-1) into per-token losses before averaging.

For STATE there are *two* heads (robot, human) and *two* targets
(`state_robot`, `state_human` in the batch dict). The trainer routes them
per row via the source_mask: robot rows supervise state_robot against
proprio, human rows supervise state_human against human_kin. The
masking module's `state` loss mask is all-ones; the per-source dispatch
happens *here*.

For BOX, the plan §2.3 calls for "MSE on translation, geodesic on quat".
We implement geodesic as `1 - |q_pred · q_target|` after L2-normalizing
both — this is range [0, 1], cheap to compute, and JIT-clean.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


# --- §3.6 weights ---------------------------------------------------------

@dataclass(frozen=True)
class LossWeights:
    """§3.6 defaults. Bridge slots are pulled hard (don't down-weight them)."""
    action: float = 1.0
    state: float = 0.2
    box: float = 1.0      # bridge
    phase: float = 0.5    # bridge
    contact: float = 0.5  # bridge


# --- per-token reductions -------------------------------------------------

def _per_token_mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """(B, T, D) -> (B, T) mean-squared error along the feature axis."""
    return jnp.mean(jnp.square(pred - target), axis=-1)


def _per_token_l1(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """L1 / mean absolute error along the feature axis. §2.3 says L1 with
    Huber as alternative — we keep L1 here; swap by editing this function."""
    return jnp.mean(jnp.abs(pred - target), axis=-1)


def _per_token_bce(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Numerically-stable per-token BCE-with-logits, mean over feature axis."""
    pos = jnp.maximum(logits, 0.0)
    bce = pos - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    return jnp.mean(bce, axis=-1)


def _per_token_ce(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Per-token cross-entropy. logits (B, T, K); targets (B, T) int.

    Returns (B, T)."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)


def _per_token_box(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """MSE on translation + geodesic on the quaternion. (B, T, 7) -> (B, T)."""
    trans_loss = jnp.mean(jnp.square(pred[..., :3] - target[..., :3]), axis=-1)
    pq = pred[..., 3:]
    tq = target[..., 3:]
    pq = pq / (jnp.linalg.norm(pq, axis=-1, keepdims=True) + 1e-8)
    tq = tq / (jnp.linalg.norm(tq, axis=-1, keepdims=True) + 1e-8)
    # 1 - |cos(angle/2)|. Equivalent to angular distance up to a monotone map.
    quat_loss = 1.0 - jnp.abs(jnp.sum(pq * tq, axis=-1))
    return trans_loss + quat_loss


# --- helpers --------------------------------------------------------------

def _shift_pairs(pred: jnp.ndarray, target: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """§2.4: pred at time t predicts target at time t+1. Drop the last
    pred (no target) and the first target (no preceding pred)."""
    return pred[:, :-1], target[:, 1:]


def _masked_mean(per_token: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Reduce a (B, T-1) per-token loss to a scalar with safe div by zero."""
    return jnp.sum(per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)


# --- the §3.6 total loss --------------------------------------------------

def compute_loss(
    preds: dict[str, jnp.ndarray],
    batch: dict[str, jnp.ndarray],
    weights: LossWeights = LossWeights(),
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute the §3.6 weighted total loss + per-slot aux scalars.

    Args:
      preds: dict from `CoTrainTransformer.__call__` — per-slot predictions
        plus `_loss_masks` (the §3.4 (B, T) masks).
      batch: input batch dict with the original (unshifted) per-slot
        sequences. Targets are batch[slot][:, 1:] per §2.4.
      weights: §3.6 weights.

    Returns:
      (loss_scalar, aux). aux has scalar entries `loss/total`,
      `loss/{state,box,phase,contact,action}` and the active-row counts
      that downstream metrics may want.
    """
    lm = preds["_loss_masks"]                                  # dict slot -> (B, T)
    src = batch["source_mask"].astype(jnp.float32)             # (B,)

    # STATE: dual-head, source-dispatched
    p_r, t_r = _shift_pairs(preds["state_robot"], batch["state_robot"])
    p_h, t_h = _shift_pairs(preds["state_human"], batch["state_human"])
    err_r = _per_token_mse(p_r, t_r)                           # (B, T-1)
    err_h = _per_token_mse(p_h, t_h)
    src_T = jnp.broadcast_to(src[:, None], err_r.shape)        # (B, T-1)
    L_state = (
        _masked_mean(err_r, src_T)             # robot rows only
        + _masked_mean(err_h, 1.0 - src_T)     # human rows only
    )

    # BOX: bridge — both sources contribute.
    p, t = _shift_pairs(preds["box"], batch["box"])
    L_box = _masked_mean(_per_token_box(p, t), lm["box"][:, :-1])

    # PHASE: cross-entropy on integer targets, bridge.
    p, t = _shift_pairs(preds["phase"], batch["phase"])
    L_phase = _masked_mean(
        _per_token_ce(p, t.squeeze(-1).astype(jnp.int32)),
        lm["phase"][:, :-1],
    )

    # CONTACT: BCE-with-logits, bridge.
    p, t = _shift_pairs(preds["contact"], batch["contact"])
    L_contact = _masked_mean(_per_token_bce(p, t), lm["contact"][:, :-1])

    # ACTION: L1, robot-only (the §3.4 action mask zeros human rows).
    p, t = _shift_pairs(preds["action"], batch["action"])
    L_action = _masked_mean(_per_token_l1(p, t), lm["action"][:, :-1])

    total = (
        weights.action * L_action
        + weights.state * L_state
        + weights.box * L_box
        + weights.phase * L_phase
        + weights.contact * L_contact
    )
    aux = {
        "loss/total":   total,
        "loss/action":  L_action,
        "loss/state":   L_state,
        "loss/box":     L_box,
        "loss/phase":   L_phase,
        "loss/contact": L_contact,
        "active_rows/robot": jnp.sum(src),
        "active_rows/human": jnp.sum(1.0 - src),
    }
    return total, aux
