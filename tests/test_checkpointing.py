"""Round-trip tests for Orbax v1 checkpointing (PROJECT_PLAN_1.md §5.4 / §8 step 10)."""
from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from flax import nnx

from cotrain.models.transformer import CoTrainTransformer
from cotrain.training.checkpointing import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from cotrain.training.trainer import (
    OptimizerConfig,
    make_optimizer,
    synthetic_batch,
    tiny_config,
    train_step,
)


def _build():
    cfg = tiny_config(T=4)
    model = CoTrainTransformer(cfg, rngs=nnx.Rngs(0))
    optimizer = make_optimizer(model, OptimizerConfig(
        lr_peak=1e-3, warmup_steps=2, decay_steps=20, weight_decay=0.0,
    ))
    return model, optimizer


def test_round_trip_model_state(tmp_path: Path) -> None:
    """After save/load, the restored model produces identical outputs."""
    model_a, opt_a = _build()
    # Take a few steps so the params aren't at init.
    for s in range(3):
        train_step(model_a, opt_a, synthetic_batch(B=4, T=4, seed=s))

    save_checkpoint(tmp_path, model=model_a, optimizer=opt_a, step=10)

    # Build a *fresh* model with different rngs, then restore.
    model_b, opt_b = _build()
    model_b_state_before = jax.tree.leaves(nnx.state(model_b, nnx.Param))

    step, _ = load_checkpoint(tmp_path / "00000010", model=model_b, optimizer=opt_b)
    assert step == 10

    # Verify Params actually changed by the restore.
    model_b_state_after = jax.tree.leaves(nnx.state(model_b, nnx.Param))
    diffs = [np.abs(np.asarray(b) - np.asarray(a)).max()
             for a, b in zip(model_b_state_before, model_b_state_after)]
    assert max(diffs) > 1e-6, "load_checkpoint didn't update Params"

    # Now A and B must produce identical outputs on the same input.
    batch = synthetic_batch(B=4, T=4, seed=99)
    a = model_a(batch, deterministic=True)
    b = model_b(batch, deterministic=True)
    for slot in ("box", "phase", "contact", "action"):
        np.testing.assert_allclose(np.asarray(a[slot]), np.asarray(b[slot]),
                                    atol=1e-5, rtol=1e-5)


def test_metadata_round_trip(tmp_path: Path) -> None:
    model, opt = _build()
    save_checkpoint(tmp_path, model=model, optimizer=opt, step=42,
                    metadata={"w": 0.25, "wandb_id": "abc123"})
    _, meta = load_checkpoint(tmp_path / "00000042", model=model, optimizer=opt)
    assert meta == {"w": 0.25, "wandb_id": "abc123"}


def test_latest_checkpoint(tmp_path: Path) -> None:
    model, opt = _build()
    save_checkpoint(tmp_path, model=model, optimizer=opt, step=100)
    save_checkpoint(tmp_path, model=model, optimizer=opt, step=500)
    save_checkpoint(tmp_path, model=model, optimizer=opt, step=200)
    latest = latest_checkpoint(tmp_path)
    assert latest is not None and latest.name == "00000500"


def test_latest_returns_none_when_empty(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path / "nope") is None
