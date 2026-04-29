"""Tests for the §3.6 loss module (PROJECT_PLAN_1.md §8 step 7).

Plan spec: 'zero loss on a batch where predictions equal targets;
nonzero otherwise; loss masks zero out the right contributions.' We
also pin per-slot independence — perturbing one slot only moves that
slot's loss.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cotrain.training.losses import (
    LossWeights,
    _per_token_box,
    _per_token_ce,
    compute_loss,
)


B, T, D_P, D_H, D_A = 4, 6, 5, 9, 4
NUM_PHASES = 5


def _ground_truth_batch(seed: int = 0) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "state_robot": jnp.asarray(rng.normal(size=(B, T, D_P)).astype(np.float32)),
        "state_human": jnp.asarray(rng.normal(size=(B, T, D_H)).astype(np.float32)),
        "box":         jnp.asarray(np.concatenate([
                            rng.normal(size=(B, T, 3)).astype(np.float32),
                            # Unit-quat targets so the geodesic loss is well-defined.
                            _unit_quats(rng, B, T),
                        ], axis=-1)),
        "phase":       jnp.asarray(rng.integers(0, NUM_PHASES, size=(B, T, 1)).astype(np.int8)),
        "contact":     jnp.asarray(rng.integers(0, 2, size=(B, T, 3)).astype(np.float32)),
        "action":      jnp.asarray(rng.normal(size=(B, T, D_A)).astype(np.float32)),
        "source_mask": jnp.asarray([(i % 2 == 0) for i in range(B)], dtype=bool),
    }


def _unit_quats(rng, B, T):
    q = rng.normal(size=(B, T, 4)).astype(np.float32)
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


def _perfect_preds_from_batch(batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """Build a `preds` dict where every prediction at time t equals the
    target at time t+1 — i.e., the optimal forecast. Phase prediction is
    one-hot logits at the right class; contact prediction is logits with
    sign matching the target. STATE values for the inactive source are
    zero (matches the zero-pad in batch[..._human/_robot])."""
    src = batch["source_mask"]
    is_robot = src[:, None, None].astype(jnp.float32)

    state_r = batch["state_robot"]
    state_h = batch["state_human"]
    box = batch["box"]
    phase = batch["phase"][..., 0].astype(jnp.int32)            # (B, T)
    contact = batch["contact"]
    action = batch["action"]

    # Construct predicted-at-t = target-at-t+1; the last position is unused.
    preds = {
        "state_robot": _shift_left(state_r),
        "state_human": _shift_left(state_h),
        "box":         _shift_left(box),
        # Phase preds are logits; large positive at the target class,
        # large negative elsewhere -> CE → 0.
        "phase":       _phase_one_hot_logits(phase, NUM_PHASES),
        "contact":     _shift_left(_signed_logits(contact)),
        "action":      _shift_left(action),
    }
    # Loss masks per the §3.4 spec. We hand-build them here so the test
    # doesn't depend on the masking module.
    ones = jnp.ones((B, T), dtype=jnp.float32)
    zeros = jnp.zeros((B, T), dtype=jnp.float32)
    action_mask = jnp.where(src[:, None], ones, zeros)
    preds["_loss_masks"] = {
        "vis":     zeros,
        "state":   ones,
        "box":     ones,
        "phase":   ones,
        "contact": ones,
        "action":  action_mask,
    }
    # Squelch unused vars.
    _ = is_robot
    return preds


def _shift_left(x: jnp.ndarray) -> jnp.ndarray:
    """Build pred[:, t] = target[:, t+1]; position T-1 is irrelevant."""
    pad = jnp.zeros_like(x[:, :1])
    return jnp.concatenate([x[:, 1:], pad], axis=1)


def _phase_one_hot_logits(target_int: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """(B, T) int -> (B, T, K) logits with target class at +50, others at -50.
    After softmax this puts ~1.0 prob on the target — CE ≈ 0."""
    target_shift = jnp.concatenate([target_int[:, 1:], jnp.zeros_like(target_int[:, :1])], axis=1)
    one_hot = jax.nn.one_hot(target_shift, num_classes=num_classes, dtype=jnp.float32)
    return one_hot * 100.0 - 50.0


def _signed_logits(target: jnp.ndarray) -> jnp.ndarray:
    """target in {0, 1} -> logits +50 / -50. After sigmoid → ~1 / ~0."""
    return target * 100.0 - 50.0


# --- tests ----------------------------------------------------------------


def test_zero_loss_at_perfect_predictions() -> None:
    """The loss should bottom out at near-zero when predictions match the
    next-step targets exactly (modulo soft-but-saturated phase/contact)."""
    batch = _ground_truth_batch()
    preds = _perfect_preds_from_batch(batch)
    total, aux = compute_loss(preds, batch)
    np.testing.assert_allclose(float(total), 0.0, atol=5e-3)
    for k in ("loss/state", "loss/box", "loss/phase", "loss/contact", "loss/action"):
        np.testing.assert_allclose(float(aux[k]), 0.0, atol=5e-3)


def test_perturbing_action_only_moves_action_loss() -> None:
    """Bumping the action prediction should change L_action and L_total but
    not L_state / L_box / L_phase / L_contact. Catches accidental
    cross-slot bleed in compute_loss."""
    batch = _ground_truth_batch(seed=1)
    base = _perfect_preds_from_batch(batch)
    perturbed = dict(base)
    perturbed["action"] = base["action"] + 1.0  # uniform offset
    a_total, a_aux = compute_loss(base, batch)
    b_total, b_aux = compute_loss(perturbed, batch)
    assert float(b_aux["loss/action"]) > float(a_aux["loss/action"]) + 0.5
    for k in ("loss/state", "loss/box", "loss/phase", "loss/contact"):
        np.testing.assert_allclose(float(a_aux[k]), float(b_aux[k]), atol=1e-6)


def test_action_mask_zeros_human_rows() -> None:
    """The §3.4 action mask must turn off action-loss contribution from
    human rows. We do this by:
      - making half the batch human (source_mask = [T, F, T, F])
      - giving action predictions wildly wrong values for human rows
      - giving action predictions identical to targets for robot rows
    Action loss should be ~0 because the wrong human rows are masked."""
    batch = _ground_truth_batch(seed=2)
    preds = _perfect_preds_from_batch(batch)
    src = np.asarray(batch["source_mask"])
    # Make human rows' action prediction garbage.
    bad = np.asarray(preds["action"]).copy()
    bad[~src] = 1000.0
    preds["action"] = jnp.asarray(bad)
    _, aux = compute_loss(preds, batch)
    # Action loss should remain ~0 because all error is on masked rows.
    np.testing.assert_allclose(float(aux["loss/action"]), 0.0, atol=5e-3)


def test_state_dispatch_routes_per_source() -> None:
    """Garbage state_human prediction shouldn't affect a fully-robot batch's
    L_state, and vice versa."""
    batch = _ground_truth_batch(seed=3)
    # Force all-robot.
    batch["source_mask"] = jnp.asarray([True] * B)
    preds = _perfect_preds_from_batch(batch)
    bad = np.asarray(preds["state_human"]).copy()
    bad[:] = 999.0
    preds["state_human"] = jnp.asarray(bad)
    _, aux = compute_loss(preds, batch)
    np.testing.assert_allclose(float(aux["loss/state"]), 0.0, atol=5e-3)


def test_compute_loss_under_jit() -> None:
    batch = _ground_truth_batch(seed=4)
    preds = _perfect_preds_from_batch(batch)

    def run(preds, batch):
        return compute_loss(preds, batch)
    fn = jax.jit(run)
    eager_total, _ = run(preds, batch)
    jit_total, _ = fn(preds, batch)
    np.testing.assert_allclose(float(eager_total), float(jit_total), atol=1e-6)


def test_box_loss_geodesic_zero_at_negated_quat() -> None:
    """Quaternions q and -q describe the same orientation; the geodesic
    component of _per_token_box must be zero between (t, q) and (t, -q)."""
    rng = np.random.default_rng(0)
    trans = jnp.asarray(rng.normal(size=(2, 3, 3)).astype(np.float32))
    q = _unit_quats(rng, 2, 3)
    pred = jnp.concatenate([trans, jnp.asarray(q)], axis=-1)
    target = jnp.concatenate([trans, jnp.asarray(-q)], axis=-1)
    err = _per_token_box(pred, target)
    np.testing.assert_allclose(np.asarray(err), 0.0, atol=1e-5)


def test_phase_ce_zero_at_one_hot_logits() -> None:
    targets = jnp.asarray([[0, 1, 2], [3, 4, 0]])           # (2, 3)
    logits = jax.nn.one_hot(targets, num_classes=NUM_PHASES) * 100.0 - 50.0
    err = _per_token_ce(logits, targets)
    np.testing.assert_allclose(np.asarray(err), 0.0, atol=1e-3)


def test_loss_weights_mix() -> None:
    """Weighted sum must equal manual sum of the per-slot losses."""
    batch = _ground_truth_batch(seed=5)
    preds = _perfect_preds_from_batch(batch)
    # Make every slot's loss non-zero.
    for slot in ("state_robot", "state_human", "box", "contact", "action"):
        preds[slot] = preds[slot] + 0.5
    weights = LossWeights(action=1.5, state=0.3, box=2.0, phase=1.0, contact=0.7)
    total, aux = compute_loss(preds, batch, weights=weights)
    expected = (
        weights.action * float(aux["loss/action"])
        + weights.state * float(aux["loss/state"])
        + weights.box * float(aux["loss/box"])
        + weights.phase * float(aux["loss/phase"])
        + weights.contact * float(aux["loss/contact"])
    )
    np.testing.assert_allclose(float(total), expected, rtol=1e-5)
