"""§8 step 5 smoke tests for CoTrainTransformer.

Per the plan: '*Tests:* (a) forward pass on CPU first (`JAX_PLATFORMS=cpu`),
(b) then on a 1-chip TPU, (c) then on the full mesh. Each step catches a
different class of bug.' This file covers (b) — TPU forward in the current
process. (a) is exercised by running this same file with
`JAX_PLATFORMS=cpu pytest tests/test_transformer.py`. (c) lights up once
the SPMD trainer runs in §8 step 9.

We use a tiny config (d_model=64, n_layers=2) for every test so JIT
compile time stays under a few seconds.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cotrain.data.schemas import DEFAULT_D_A, DEFAULT_D_H, DEFAULT_D_P
from cotrain.models.heads.projection import DINO_FEATURE_DIM
from cotrain.models.transformer import (
    NUM_SLOTS,
    CoTrainTransformer,
    TransformerConfig,
)


def _tiny_cfg(T: int = 4) -> TransformerConfig:
    return TransformerConfig(
        d_model=64, n_layers=2, n_heads=4, T=T,
        D_p=DEFAULT_D_P, D_h=DEFAULT_D_H, D_a=DEFAULT_D_A,
        dino_dim=DINO_FEATURE_DIM, dropout_rate=0.0,
    )


def _dummy_batch(B: int, T: int, *, seed: int = 0) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "vis":         jnp.asarray(rng.normal(size=(B, T, DINO_FEATURE_DIM)).astype(np.float32)),
        "state_robot": jnp.asarray(rng.normal(size=(B, T, DEFAULT_D_P)).astype(np.float32)),
        "state_human": jnp.asarray(rng.normal(size=(B, T, DEFAULT_D_H)).astype(np.float32)),
        "box":         jnp.asarray(rng.normal(size=(B, T, 7)).astype(np.float32)),
        "phase":       jnp.asarray(rng.integers(0, 5, size=(B, T, 1)).astype(np.int8)),
        "contact":     jnp.asarray(rng.integers(0, 2, size=(B, T, 3)).astype(np.float32)),
        "action":      jnp.asarray(rng.normal(size=(B, T, DEFAULT_D_A)).astype(np.float32)),
        "source_mask": jnp.asarray([(i % 2 == 0) for i in range(B)], dtype=bool),
    }


def test_forward_output_shapes() -> None:
    """Each predicted slot is (B, T, expected_dim_for_slot)."""
    cfg = _tiny_cfg(T=4)
    model = CoTrainTransformer(cfg, rngs=nnx.Rngs(0))
    batch = _dummy_batch(B=4, T=cfg.T)
    out = model(batch, deterministic=True)
    pred_slots = {k for k in out if not k.startswith("_")}
    assert pred_slots == {"state_robot", "state_human", "box", "phase", "contact", "action"}
    assert out["state_robot"].shape == (4, cfg.T, cfg.D_p)
    assert out["state_human"].shape == (4, cfg.T, cfg.D_h)
    assert out["box"].shape         == (4, cfg.T, 7)
    assert out["phase"].shape       == (4, cfg.T, 5)
    assert out["contact"].shape     == (4, cfg.T, 3)
    assert out["action"].shape      == (4, cfg.T, cfg.D_a)
    # Loss masks are stashed under _loss_masks for the trainer.
    assert "_loss_masks" in out
    lm = out["_loss_masks"]
    assert set(lm) == {"vis", "state", "box", "phase", "contact", "action"}
    assert lm["box"].shape == (4, cfg.T)


def test_runs_under_nnx_jit() -> None:
    """Most §3.4 / §3.2 traceability bugs surface here."""
    cfg = _tiny_cfg(T=4)
    model = CoTrainTransformer(cfg, rngs=nnx.Rngs(0))

    @nnx.jit
    def fwd(model: CoTrainTransformer, batch: dict) -> dict:
        return model(batch, deterministic=True)

    batch = _dummy_batch(B=4, T=cfg.T)
    out = fwd(model, batch)
    assert out["action"].shape == (4, cfg.T, cfg.D_a)


def test_state_tree_includes_block_weights() -> None:
    """Sanity: every block's weights are in the state tree (not lost via
    mis-iteration of the list-of-blocks). One missed leaf would silently
    skip optimization on that layer."""
    cfg = _tiny_cfg(T=4)
    model = CoTrainTransformer(cfg, rngs=nnx.Rngs(0))
    state = nnx.state(model, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    assert len(leaves) > 30, f"too few Param leaves: {len(leaves)}"


def test_modality_masking_uses_standalone_module() -> None:
    """End-to-end check that the backbone actually invokes the §3.4
    masking module (not a stale shim). We stash a sentinel in vis_mask /
    action_mask, run forward, and verify the loss_masks returned line up
    with what cotrain.training.masking.apply_modality_masks computes
    directly on the same inputs."""
    from cotrain.training.masking import apply_modality_masks

    cfg = _tiny_cfg(T=4)
    model = CoTrainTransformer(cfg, rngs=nnx.Rngs(0))
    model.heads.vis_mask.value = jnp.full((cfg.d_model,), 7.0)
    model.heads.action_mask.value = jnp.full((cfg.d_model,), -3.0)

    batch = _dummy_batch(B=4, T=cfg.T)
    out = model(batch, deterministic=True)
    lm = out["_loss_masks"]

    # Compute expected loss masks via the standalone module on stub tokens.
    stub = {s: jnp.zeros((4, cfg.T, cfg.d_model)) for s in
            ("vis", "state", "box", "phase", "contact", "action")}
    _, expected = apply_modality_masks(
        stub, batch["source_mask"],
        model.heads.vis_mask.value, model.heads.action_mask.value,
    )
    for slot, m in expected.items():
        np.testing.assert_array_equal(np.asarray(lm[slot]), np.asarray(m))


def test_dropout_off_at_eval_on_at_train() -> None:
    """deterministic=True should yield bitwise-identical outputs across
    repeated forwards; deterministic=False should diverge."""
    cfg = _tiny_cfg(T=4)
    cfg = TransformerConfig(**{**cfg.__dict__, "dropout_rate": 0.5})
    model = CoTrainTransformer(cfg, rngs=nnx.Rngs(0))
    batch = _dummy_batch(B=2, T=cfg.T)

    a = np.asarray(model(batch, deterministic=True)["action"])
    b = np.asarray(model(batch, deterministic=True)["action"])
    np.testing.assert_array_equal(a, b)

    c = np.asarray(model(batch, deterministic=False)["action"])
    d = np.asarray(model(batch, deterministic=False)["action"])
    assert not np.array_equal(c, d), "dropout should produce different outputs"


@pytest.mark.parametrize("B,T", [(1, 4), (2, 8), (4, 4)])
def test_param_shapes_are_jit_stable(B: int, T: int) -> None:
    """Different (B, T) shapes should compile cleanly without state-tree drift."""
    cfg = _tiny_cfg(T=T)
    model = CoTrainTransformer(cfg, rngs=nnx.Rngs(0))

    @nnx.jit
    def fwd(model, batch):
        return model(batch, deterministic=True)

    out = fwd(model, _dummy_batch(B=B, T=T, seed=B * 100 + T))
    assert out["box"].shape == (B, T, 7)
