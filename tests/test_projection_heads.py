"""Smoke tests for the §2.2 projection heads (PROJECT_PLAN_1.md §8 step 2).

Required by the plan:
  *Test:* shape & dtype on dummy inputs; nnx.split(model) produces a
  non-empty state tree.

We exceed that floor with two extra checks the plan flags as easy bug
sources:
  - Robot vs human source_mask routes the right STATE head (jnp.where, not
    Python branching — §3.4 warning).
  - vis_mask / action_mask vectors live in the state tree as nnx.Param so
    the optimizer will update them (§2.2 final paragraph).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cotrain.data.schemas import DEFAULT_D_A, DEFAULT_D_H, DEFAULT_D_P
from cotrain.models.heads import (
    BOX_DIM,
    CONTACT_DIM,
    DINO_FEATURE_DIM,
    NUM_PHASES,
    ProjectionHeads,
)


D_MODEL = 64  # smaller than prod (768) to keep tests fast


def _make_heads(seed: int = 0) -> ProjectionHeads:
    return ProjectionHeads(
        d_model=D_MODEL,
        D_p=DEFAULT_D_P,
        D_h=DEFAULT_D_H,
        D_a=DEFAULT_D_A,
        rngs=nnx.Rngs(seed),
    )


def _dummy_batch(B: int = 4, T: int = 8, *, seed: int = 1) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "vis":         jnp.asarray(rng.normal(size=(B, T, DINO_FEATURE_DIM)).astype(np.float32)),
        "state_robot": jnp.asarray(rng.normal(size=(B, T, DEFAULT_D_P)).astype(np.float32)),
        "state_human": jnp.asarray(rng.normal(size=(B, T, DEFAULT_D_H)).astype(np.float32)),
        "box":         jnp.asarray(rng.normal(size=(B, T, BOX_DIM)).astype(np.float32)),
        "phase":       jnp.asarray(rng.integers(0, NUM_PHASES, size=(B, T, 1)).astype(np.int8)),
        "contact":     jnp.asarray(rng.integers(0, 2, size=(B, T, CONTACT_DIM)).astype(np.float32)),
        "action":      jnp.asarray(rng.normal(size=(B, T, DEFAULT_D_A)).astype(np.float32)),
        # Alternating robot/human so any B has a mix:
        "source_mask": jnp.asarray([(i % 2 == 0) for i in range(B)], dtype=bool),
    }


def test_output_shapes_and_dtypes() -> None:
    heads = _make_heads()
    batch = _dummy_batch(B=4, T=8)
    out = heads(**batch)
    expected = {"vis", "state", "box", "phase", "contact", "action"}
    assert set(out) == expected, set(out)
    for slot, tok in out.items():
        assert tok.shape == (4, 8, D_MODEL), f"{slot}: {tok.shape}"
        assert tok.dtype == jnp.float32, f"{slot}: {tok.dtype}"


def test_state_tree_is_non_empty_and_includes_mask_params() -> None:
    """nnx.split should expose every nnx.Param so the optimizer sees them."""
    heads = _make_heads()
    graph_def, state = nnx.split(heads)
    flat = jax.tree.leaves(state)
    assert flat, "state tree is empty — nothing for the optimizer to update"
    # The two learned mask vectors must be present and zero-initialized.
    assert "vis_mask" in state, list(state.keys())
    assert "action_mask" in state, list(state.keys())
    vis_mask_val = state["vis_mask"].value
    action_mask_val = state["action_mask"].value
    assert vis_mask_val.shape == (D_MODEL,)
    assert action_mask_val.shape == (D_MODEL,)
    assert jnp.allclose(vis_mask_val, 0.0)
    assert jnp.allclose(action_mask_val, 0.0)


def test_state_routing_uses_source_mask() -> None:
    """Robot rows must reflect state_robot; human rows must reflect state_human.

    Diagnostic: zero out one source's input and check the other source's
    rows are unaffected. Catches a class of bug where a Python-level branch
    on `source_mask` would have hard-coded one branch."""
    heads = _make_heads()
    batch = _dummy_batch(B=4, T=8)

    # Run with state_human zeroed: robot rows unchanged, human rows go to
    # whatever state_human(zeros) projects to (deterministic).
    batch_a = dict(batch)
    batch_a["state_human"] = jnp.zeros_like(batch["state_human"])
    out_a = heads(**batch_a)["state"]

    # Run with state_robot zeroed: human rows unchanged.
    batch_b = dict(batch)
    batch_b["state_robot"] = jnp.zeros_like(batch["state_robot"])
    out_b = heads(**batch_b)["state"]

    out_ref = heads(**batch)["state"]
    src = np.asarray(batch["source_mask"])
    robot_idx = np.where(src)[0]
    human_idx = np.where(~src)[0]
    # Robot rows must match between batch_a (state_human zeroed) and the ref.
    np.testing.assert_allclose(np.asarray(out_a[robot_idx]), np.asarray(out_ref[robot_idx]),
                               atol=1e-6, rtol=1e-5)
    # Human rows must match between batch_b (state_robot zeroed) and the ref.
    np.testing.assert_allclose(np.asarray(out_b[human_idx]), np.asarray(out_ref[human_idx]),
                               atol=1e-6, rtol=1e-5)


def test_runs_under_nnx_jit() -> None:
    """The plan calls out (§3.4, §5.5 #1) that nnx.jit is the failure mode
    where Python branching silently breaks. Running once under nnx.jit is
    the cheap check that the heads are JIT-clean."""
    heads = _make_heads()

    @nnx.jit
    def fwd(model: ProjectionHeads, batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        return model(**batch)

    batch = _dummy_batch(B=4, T=8)
    out = fwd(heads, batch)
    for tok in out.values():
        assert tok.shape == (4, 8, D_MODEL)


@pytest.mark.parametrize("B,T", [(1, 1), (2, 16), (8, 4)])
def test_shapes_parametric(B: int, T: int) -> None:
    heads = _make_heads()
    batch = _dummy_batch(B=B, T=T, seed=B * 100 + T)
    out = heads(**batch)
    for slot, tok in out.items():
        assert tok.shape == (B, T, D_MODEL), f"{slot}@(B={B},T={T}): {tok.shape}"
