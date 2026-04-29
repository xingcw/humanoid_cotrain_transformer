"""§8 step 10 end-to-end smoke test (PROJECT_PLAN_1.md spec).

The plan: '§5 training loop end-to-end on 100 episodes per side, 1k steps,
single host. *Test:* loss decreases; no NaNs; Orbax checkpoint saves and
loads correctly.'

This file is the *fast* version — small synthetic dataset, T=4, tiny
transformer config, ~50 steps. It exercises every wire (loader → encoder
→ transformer → loss → optimizer → checkpoint) but completes in under a
minute. The full 1k-step run is launched manually via
`python -m cotrain.scripts.train` (lands as a separate piece in §8.10
follow-up).
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cotrain.models.encoders import DinoV2Encoder
from cotrain.models.transformer import CoTrainTransformer
from cotrain.scripts.build_grain_shards import build_shards_for_source
from cotrain.scripts.generate_synthetic_data import generate
from cotrain.training.checkpointing import load_checkpoint, save_checkpoint
from cotrain.training.sampler import make_mixed_loader
from cotrain.training.trainer import (
    OptimizerConfig,
    make_optimizer,
    tiny_config,
    train_step_with_encoder,
)

T = 4
B = 4
N_STEPS = 30
SAVE_AT = 15


_MODEL_KEYS = {
    "rgb", "state_robot", "state_human", "box", "phase", "contact", "action",
    "source_mask",
}


def _to_jax(batch: dict) -> dict:
    """Keep only the keys the trainer/model consume. Drops loader
    bookkeeping fields like episode_ids / window_starts."""
    return {k: jnp.asarray(batch[k]) for k in _MODEL_KEYS}


@pytest.fixture(scope="module")
def shards_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("e2e_shards")
    generate(root, n_robot=10, n_human=10, seed=99, T_range=(16, 24))
    for source in ("robot", "human"):
        build_shards_for_source(root / source, T=T, stride=2, examples_per_shard=128)
    return root


def _build():
    cfg = tiny_config(T=T)
    model = CoTrainTransformer(cfg, rngs=nnx.Rngs(0))
    optimizer = make_optimizer(model, OptimizerConfig(
        lr_peak=2e-3, warmup_steps=2, decay_steps=N_STEPS, weight_decay=0.0,
    ))
    # pretrained=False to keep this test offline; the wire path is identical.
    encoder = DinoV2Encoder(rngs=nnx.Rngs(1), pretrained=False)
    return cfg, model, optimizer, encoder


def test_e2e_loss_decreases_no_nans(shards_root: Path) -> None:
    """Loader → encoder → transformer → loss → optimizer for N_STEPS."""
    cfg, model, optimizer, encoder = _build()
    loader = make_mixed_loader(shards_root, w=0.5, batch_size=B, seed=0)

    losses: list[float] = []
    for step in range(N_STEPS):
        batch = _to_jax(next(loader))
        loss, aux = train_step_with_encoder(model, optimizer, encoder, batch)
        loss_v = float(loss)
        assert np.isfinite(loss_v), f"step {step}: loss = {loss_v}"
        for k, v in aux.items():
            assert np.isfinite(float(v)), f"step {step}: {k} = {float(v)}"
        losses.append(loss_v)

    # First 5 vs last 5 mean — robust to per-step noise.
    first = float(np.mean(losses[:5]))
    last = float(np.mean(losses[-5:]))
    assert last < first - 1e-3, f"loss didn't decrease: first5={first:.4f}, last5={last:.4f}"


def test_e2e_checkpoint_save_then_match(shards_root: Path, tmp_path: Path) -> None:
    """Save mid-run; restore into a fresh model/optimizer; verify the
    restored pair predicts identically to the saved pair."""
    cfg, model_a, opt_a, encoder = _build()
    loader = make_mixed_loader(shards_root, w=0.5, batch_size=B, seed=7)

    for step in range(SAVE_AT):
        train_step_with_encoder(model_a, opt_a, encoder, _to_jax(next(loader)))

    save_dir = save_checkpoint(tmp_path / "ckpt", model=model_a, optimizer=opt_a, step=SAVE_AT)
    assert save_dir.is_dir()

    # Fresh model with different rngs.
    cfg, model_b, opt_b, _ = _build()
    step, _ = load_checkpoint(save_dir, model=model_b, optimizer=opt_b)
    assert step == SAVE_AT

    # Build a probe that the model accepts (no rgb; explicit vis).
    probe = {k: v for k, v in _to_jax(next(loader)).items() if k != "rgb"}
    fake_vis = jnp.asarray(np.random.default_rng(0).normal(
        size=(B, T, 768)).astype(np.float32))
    a = model_a({**probe, "vis": fake_vis}, deterministic=True)
    b = model_b({**probe, "vis": fake_vis}, deterministic=True)
    for slot in ("box", "phase", "contact", "action"):
        np.testing.assert_allclose(np.asarray(a[slot]), np.asarray(b[slot]),
                                    atol=1e-5, rtol=1e-5)
