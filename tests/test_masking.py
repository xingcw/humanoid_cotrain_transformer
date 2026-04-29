"""§8 step 6 — the test that catches the most bugs (PROJECT_PLAN_1.md §3.4).

The plan flags this module as the single function most likely to break
under nnx.jit when an agent ports from PyTorch. Tests intentionally hit
all three risk axes:

1. **Correctness** — robot rows get vis_mask, action preserved; human rows
   get action_mask, vis preserved; bridge slots untouched in both cases.
2. **JIT-traceability** — running the function inside `jax.jit` must not
   change behavior (catches Python-level branching).
3. **Loss-mask correctness** — vis_loss = 0 always; action_loss = 1 only
   for robot rows; bridge slots all-ones.

We also stress edge cases (all-robot, all-human) and a randomized property
check across 16 random source_mask draws.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cotrain.training.masking import apply_modality_masks

D = 8
T = 4


def _make_tokens(B: int, *, seed: int) -> dict[str, jnp.ndarray]:
    """Per-slot tokens (B, T, D) with each slot's values offset so we can
    spot any cross-slot bleed."""
    rng = np.random.default_rng(seed)
    return {
        "vis":     jnp.asarray(rng.normal(size=(B, T, D)).astype(np.float32) + 100),
        "state":   jnp.asarray(rng.normal(size=(B, T, D)).astype(np.float32) + 200),
        "box":     jnp.asarray(rng.normal(size=(B, T, D)).astype(np.float32) + 300),
        "phase":   jnp.asarray(rng.normal(size=(B, T, D)).astype(np.float32) + 400),
        "contact": jnp.asarray(rng.normal(size=(B, T, D)).astype(np.float32) + 500),
        "action":  jnp.asarray(rng.normal(size=(B, T, D)).astype(np.float32) + 600),
    }


VIS_TAG = jnp.full((D,), -1.0)
ACT_TAG = jnp.full((D,), -2.0)


def test_robot_vis_replaced_action_kept() -> None:
    tokens = _make_tokens(B=4, seed=0)
    src = jnp.asarray([True, True, True, True])
    masked, _ = apply_modality_masks(tokens, src, VIS_TAG, ACT_TAG)
    np.testing.assert_array_equal(
        np.asarray(masked["vis"]),
        np.broadcast_to(np.asarray(VIS_TAG), (4, T, D)),
    )
    np.testing.assert_array_equal(
        np.asarray(masked["action"]), np.asarray(tokens["action"]),
    )


def test_human_action_replaced_vis_kept() -> None:
    tokens = _make_tokens(B=4, seed=1)
    src = jnp.asarray([False, False, False, False])
    masked, _ = apply_modality_masks(tokens, src, VIS_TAG, ACT_TAG)
    np.testing.assert_array_equal(
        np.asarray(masked["action"]),
        np.broadcast_to(np.asarray(ACT_TAG), (4, T, D)),
    )
    np.testing.assert_array_equal(
        np.asarray(masked["vis"]), np.asarray(tokens["vis"]),
    )


def test_mixed_batch_routes_per_row() -> None:
    """Heart of §3.4: two rows robot, two rows human."""
    tokens = _make_tokens(B=4, seed=2)
    src = jnp.asarray([True, False, True, False])
    masked, _ = apply_modality_masks(tokens, src, VIS_TAG, ACT_TAG)
    src_np = np.asarray(src)
    # Robot rows.
    for i in np.where(src_np)[0]:
        np.testing.assert_array_equal(
            np.asarray(masked["vis"][i]),
            np.broadcast_to(np.asarray(VIS_TAG), (T, D)),
        )
        np.testing.assert_array_equal(
            np.asarray(masked["action"][i]), np.asarray(tokens["action"][i]),
        )
    # Human rows.
    for i in np.where(~src_np)[0]:
        np.testing.assert_array_equal(
            np.asarray(masked["action"][i]),
            np.broadcast_to(np.asarray(ACT_TAG), (T, D)),
        )
        np.testing.assert_array_equal(
            np.asarray(masked["vis"][i]), np.asarray(tokens["vis"][i]),
        )


def test_bridge_slots_passthrough() -> None:
    """STATE/BOX/PHASE/CONTACT must come out unchanged for any source_mask."""
    tokens = _make_tokens(B=4, seed=3)
    src = jnp.asarray([True, False, True, False])
    masked, _ = apply_modality_masks(tokens, src, VIS_TAG, ACT_TAG)
    for slot in ("state", "box", "phase", "contact"):
        np.testing.assert_array_equal(
            np.asarray(masked[slot]), np.asarray(tokens[slot]),
        )


def test_loss_masks_correctness() -> None:
    tokens = _make_tokens(B=4, seed=4)
    src = jnp.asarray([True, False, True, False])
    _, lm = apply_modality_masks(tokens, src, VIS_TAG, ACT_TAG)
    # vis is never supervised.
    np.testing.assert_array_equal(np.asarray(lm["vis"]), np.zeros((4, T)))
    # bridge slots: always supervised.
    for slot in ("state", "box", "phase", "contact"):
        np.testing.assert_array_equal(np.asarray(lm[slot]), np.ones((4, T)))
    # action: 1 for robot rows, 0 for human rows.
    expected = np.array([[1] * T, [0] * T, [1] * T, [0] * T], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(lm["action"]), expected)


def test_traceability_under_jit() -> None:
    """The §3.4 named failure mode: Python branching silently breaks under
    jit. We compare jit-compiled output to the eager output and require
    them bit-identical."""
    tokens = _make_tokens(B=6, seed=5)
    src = jnp.asarray([True, False, True, True, False, True])

    eager = apply_modality_masks(tokens, src, VIS_TAG, ACT_TAG)
    fn = jax.jit(apply_modality_masks)
    jitted = fn(tokens, src, VIS_TAG, ACT_TAG)

    for slot in ("vis", "state", "box", "phase", "contact", "action"):
        np.testing.assert_array_equal(np.asarray(eager[0][slot]),
                                       np.asarray(jitted[0][slot]))
        np.testing.assert_array_equal(np.asarray(eager[1][slot]),
                                       np.asarray(jitted[1][slot]))


def test_dtype_promotion_for_non_bool_source_mask() -> None:
    """Trainer might pass a uint8/int8 source_mask through grain; we cast
    to bool internally so the forward stays correct."""
    tokens = _make_tokens(B=2, seed=6)
    src = jnp.asarray([1, 0], dtype=jnp.int32)        # not bool
    masked, lm = apply_modality_masks(tokens, src, VIS_TAG, ACT_TAG)
    # Row 0 should be robot: VIS replaced.
    np.testing.assert_array_equal(np.asarray(masked["vis"][0]),
                                   np.broadcast_to(np.asarray(VIS_TAG), (T, D)))
    # Row 1 should be human: ACTION replaced.
    np.testing.assert_array_equal(np.asarray(masked["action"][1]),
                                   np.broadcast_to(np.asarray(ACT_TAG), (T, D)))


def test_missing_required_slot_raises() -> None:
    tokens = _make_tokens(B=2, seed=7)
    del tokens["vis"]
    src = jnp.asarray([True, False])
    with pytest.raises(KeyError, match="vis.*action"):
        apply_modality_masks(tokens, src, VIS_TAG, ACT_TAG)


@pytest.mark.parametrize("seed", range(8))
def test_property_random_source_masks(seed: int) -> None:
    """For a random source_mask, every row must satisfy the per-row spec."""
    rng = np.random.default_rng(seed * 31 + 17)
    B = int(rng.integers(1, 8))
    src_np = rng.integers(0, 2, size=(B,)).astype(bool)
    src = jnp.asarray(src_np)
    tokens = _make_tokens(B=B, seed=seed)
    masked, lm = apply_modality_masks(tokens, src, VIS_TAG, ACT_TAG)

    for i in range(B):
        if src_np[i]:                                  # robot
            np.testing.assert_array_equal(np.asarray(masked["vis"][i]),
                                           np.broadcast_to(np.asarray(VIS_TAG), (T, D)))
            np.testing.assert_array_equal(np.asarray(masked["action"][i]),
                                           np.asarray(tokens["action"][i]))
            np.testing.assert_array_equal(np.asarray(lm["action"][i]), np.ones(T))
        else:                                          # human
            np.testing.assert_array_equal(np.asarray(masked["action"][i]),
                                           np.broadcast_to(np.asarray(ACT_TAG), (T, D)))
            np.testing.assert_array_equal(np.asarray(masked["vis"][i]),
                                           np.asarray(tokens["vis"][i]))
            np.testing.assert_array_equal(np.asarray(lm["action"][i]), np.zeros(T))
