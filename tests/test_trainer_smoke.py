"""§8 step 9 smoke test: 10 train_step iterations on synthetic batches.

The plan: 'Run a 10-step train_step on synthetic batches with the real
nnx.jit + sharding. Shape and sharding bugs surface here, where the
iteration is fast.'

We use a tiny transformer config (d_model=64, 2 layers) so a 10-step
loop completes in a few seconds. Three things must hold:
  1. Loss is finite (no NaN/Inf at any step).
  2. Loss decreases between step 0 and step 9 — this catches optimizer
     wiring bugs where gradients don't actually flow.
  3. The forward runs cleanly under sharding on TPU (or CPU when
     JAX_PLATFORMS=cpu — the same mesh path).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from cotrain.models.transformer import CoTrainTransformer
from cotrain.training.sharding import data_sharding, make_mesh, shard_batch
from cotrain.training.trainer import (
    OptimizerConfig,
    eval_step,
    make_optimizer,
    synthetic_batch,
    tiny_config,
    train_step,
)


B = 8
T = 4
N_STEPS = 10


def _build() -> tuple[CoTrainTransformer, nnx.Optimizer]:
    cfg = tiny_config(T=T)
    model = CoTrainTransformer(cfg, rngs=nnx.Rngs(0))
    # Quick warmup so the first 10 steps actually move the optimizer.
    opt_cfg = OptimizerConfig(
        lr_peak=1e-3, warmup_steps=2, decay_steps=N_STEPS,
        weight_decay=0.0, grad_clip_norm=1.0,
    )
    optimizer = make_optimizer(model, opt_cfg)
    return model, optimizer


def test_ten_step_loss_decreases() -> None:
    model, optimizer = _build()
    losses: list[float] = []
    for step in range(N_STEPS):
        batch = synthetic_batch(B=B, T=T, seed=step)
        loss, aux = train_step(model, optimizer, batch)
        loss_val = float(loss)
        assert np.isfinite(loss_val), f"step {step}: loss = {loss_val}"
        for k, v in aux.items():
            v_np = float(v)
            assert np.isfinite(v_np), f"step {step}: {k} = {v_np}"
        losses.append(loss_val)

    # The plan's 'gradients flow' check: the average of the last 3 steps
    # should be lower than the first step. Asking for monotonic decrease
    # would be brittle on stochastic batches.
    assert np.mean(losses[-3:]) < losses[0] - 1e-3, (
        f"loss didn't decrease: first={losses[0]:.4f}, "
        f"last3_mean={np.mean(losses[-3:]):.4f}"
    )


def test_train_step_is_jit_compiled_once() -> None:
    """Repeated train_step calls with same shapes shouldn't recompile.

    We measure compile-cache hits indirectly: after the first call, every
    subsequent call must return in less time than the first by a wide
    margin (no fresh tracing)."""
    import time

    model, optimizer = _build()
    batch = synthetic_batch(B=B, T=T, seed=0)

    t0 = time.perf_counter()
    train_step(model, optimizer, batch)
    jax.block_until_ready(jnp.array(0.0))
    first_dt = time.perf_counter() - t0

    # Steady-state.
    times = []
    for step in range(1, 5):
        b = synthetic_batch(B=B, T=T, seed=step)
        t0 = time.perf_counter()
        train_step(model, optimizer, b)
        jax.block_until_ready(jnp.array(0.0))
        times.append(time.perf_counter() - t0)

    # Steady step time must be << first-step time. 4× margin is generous.
    assert max(times) * 4 < first_dt, (
        f"recompile suspected: first={first_dt:.3f}s, steady_max={max(times):.3f}s"
    )


def test_forward_under_data_sharding() -> None:
    """Smoke-test the SPMD path: shard the batch over jax.devices() and
    run one step. Catches sharding-spec mistakes."""
    model, optimizer = _build()
    mesh = make_mesh()
    sharding = data_sharding(mesh)

    batch = synthetic_batch(B=B, T=T, seed=0)
    sharded = shard_batch(batch, sharding)

    loss, _ = train_step(model, optimizer, sharded)
    assert np.isfinite(float(loss))


def test_eval_step_does_not_mutate_optimizer() -> None:
    """eval_step is loss-only — opt state must be unchanged after it runs."""
    model, optimizer = _build()
    batch = synthetic_batch(B=B, T=T, seed=0)

    before_state = nnx.state(optimizer)
    eval_step(model, batch)
    after_state = nnx.state(optimizer)
    # Cheap sanity: opt step counter shouldn't have advanced.
    flat_before = jax.tree.leaves(before_state)
    flat_after = jax.tree.leaves(after_state)
    assert len(flat_before) == len(flat_after)
    for a, b in zip(flat_before, flat_after):
        if hasattr(a, "shape") and hasattr(b, "shape") and a.shape == b.shape:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
