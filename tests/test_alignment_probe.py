"""§6.3 alignment-probe tests (PROJECT_PLAN_1.md §8 step 11).

Plan spec: 'on a batch where robot and human samples are identical
(synthetic), Wasserstein ≈ 0 and discriminator accuracy ≈ 50%.' We pin
both directions: identical distributions collapse to ~0 / ~0.5; clearly
separated distributions saturate to large W and ~1.0 accuracy.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cotrain.eval.alignment_probe import (
    AlignmentMetrics,
    compute_metrics,
    discriminator_accuracy,
    extract_bridge_features,
    plot_umap,
    sliced_wasserstein,
)


D = 32


def _make_features(n: int, mean: float, *, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(loc=mean, size=(n, D)).astype(np.float32)


def test_wasserstein_near_zero_for_identical_distributions() -> None:
    """The plan's headline check: identical distributions => W ≈ 0."""
    a = _make_features(800, 0.0, seed=0)
    b = _make_features(800, 0.0, seed=1)
    sw = sliced_wasserstein(a, b, n_projections=128, seed=0)
    # Empirical W from finite samples doesn't go to literal 0; it scales
    # like O(1/sqrt(N)). With N=800, expect well under 0.1 here.
    assert sw < 0.1, sw


def test_wasserstein_grows_with_separation() -> None:
    a = _make_features(800, 0.0, seed=0)
    b = _make_features(800, 5.0, seed=1)
    sw = sliced_wasserstein(a, b, n_projections=128, seed=0)
    assert sw > 4.0, sw  # Means 5 apart in every direction.


def test_discriminator_chance_on_identical_distributions() -> None:
    """The plan's headline: identical distributions => acc ≈ 0.5."""
    a = _make_features(2000, 0.0, seed=0)
    b = _make_features(2000, 0.0, seed=1)
    feats = np.concatenate([a, b], axis=0)
    labels = np.concatenate([np.ones(len(a), dtype=int),
                              np.zeros(len(b), dtype=int)])
    acc = discriminator_accuracy(feats, labels, n_steps=200, seed=0)
    # Random classifier: ~0.5. Allow a 0.1 band for finite samples.
    assert 0.4 <= acc <= 0.6, acc


def test_discriminator_high_when_distributions_separated() -> None:
    a = _make_features(2000, 0.0, seed=0)
    b = _make_features(2000, 5.0, seed=1)
    feats = np.concatenate([a, b], axis=0)
    labels = np.concatenate([np.ones(len(a), dtype=int),
                              np.zeros(len(b), dtype=int)])
    acc = discriminator_accuracy(feats, labels, n_steps=200, seed=0)
    assert acc > 0.95, acc


def test_extract_bridge_features_shape() -> None:
    """Bridge tokens are slots 2, 3, 4 in the SLOT_ORDER, so for (B=4, T=3)
    we get 4*3*3 = 36 rows."""
    B, T, dim = 4, 3, 16
    seq = jnp.asarray(np.random.default_rng(0).normal(
        size=(B, T * 6, dim)).astype(np.float32))
    feats = extract_bridge_features(seq)
    assert feats.shape == (B * 3 * T, dim)


def test_compute_metrics_round_trip_with_umap() -> None:
    """End-to-end: build features, get a full AlignmentMetrics object."""
    a = _make_features(400, 0.0, seed=0)
    b = _make_features(400, 1.0, seed=1)
    feats = np.concatenate([a, b], axis=0)
    labels = np.concatenate([np.ones(len(a), dtype=bool),
                              np.zeros(len(b), dtype=bool)])
    m = compute_metrics(feats, labels, discriminator_steps=200, compute_umap=True)
    assert isinstance(m, AlignmentMetrics)
    assert m.n_robot == 400 and m.n_human == 400
    assert m.umap_2d is not None and m.umap_2d.shape == (800, 2)
    # Mean-1.0 separation should produce nontrivial Wasserstein and ~80%+ acc.
    assert m.sliced_wasserstein > 0.5, m.sliced_wasserstein
    assert m.discriminator_accuracy > 0.7, m.discriminator_accuracy


def test_compute_metrics_log_dict_shape() -> None:
    a = _make_features(400, 0.0, seed=0)
    b = _make_features(400, 1.0, seed=1)
    feats = np.concatenate([a, b], axis=0)
    labels = np.concatenate([np.ones(len(a), dtype=bool),
                              np.zeros(len(b), dtype=bool)])
    m = compute_metrics(feats, labels, discriminator_steps=50, compute_umap=False)
    log = m.to_log_dict("test")
    assert set(log) == {
        "test/sliced_wasserstein",
        "test/discriminator_accuracy",
        "test/n_robot",
        "test/n_human",
    }


def test_plot_umap_writes_file(tmp_path: Path) -> None:
    a = _make_features(200, 0.0, seed=0)
    b = _make_features(200, 1.0, seed=1)
    feats = np.concatenate([a, b], axis=0)
    labels = np.concatenate([np.ones(len(a), dtype=bool),
                              np.zeros(len(b), dtype=bool)])
    m = compute_metrics(feats, labels, discriminator_steps=50, compute_umap=True)
    out = plot_umap(m, tmp_path / "umap_step_0.png")
    assert out.is_file() and out.stat().st_size > 1000  # actual PNG, not empty


def test_collect_bridge_features_runs_through_model() -> None:
    """Hits the .encode() path on the real backbone."""
    from cotrain.models.transformer import CoTrainTransformer
    from cotrain.training.trainer import synthetic_batch, tiny_config
    from cotrain.eval.alignment_probe import collect_bridge_features

    cfg = tiny_config(T=4)
    model = CoTrainTransformer(cfg, rngs=nnx.Rngs(0))
    batches = [synthetic_batch(B=4, T=cfg.T, seed=i) for i in range(3)]
    feats, labels = collect_bridge_features(model, batches, n_batches=3)
    # Each batch contributes B*3*T tokens. With B=4, T=4: 48 per batch * 3 = 144.
    assert feats.shape == (3 * 4 * 3 * cfg.T, cfg.d_model)
    assert labels.shape == (3 * 4 * 3 * cfg.T,)
    assert labels.dtype == bool


def test_compute_metrics_one_source_raises() -> None:
    feats = _make_features(50, 0.0, seed=0)
    labels = np.ones(50, dtype=bool)
    with pytest.raises(ValueError, match="need both sources"):
        compute_metrics(feats, labels, discriminator_steps=50, compute_umap=False)
