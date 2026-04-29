"""§8 step 8 sampler tests.

Plan spec: 'over 1000 batches, ratio of robot samples ≈ w ± 1%;
deterministic given the same seed.' We use a small synthetic dataset and
1000 batches with B=8 to keep wall time under a few seconds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cotrain.scripts.build_grain_shards import build_shards_for_source
from cotrain.scripts.generate_synthetic_data import generate
from cotrain.training.sampler import (
    MixedBatchSpec,
    MixedShardSampler,
    lower_bound_w,
    make_mixed_loader,
    upper_bound_w,
)

T = 16


@pytest.fixture(scope="module")
def shard_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth_for_sampler")
    generate(root, n_robot=20, n_human=20, seed=33, T_range=(40, 60))
    for source in ("robot", "human"):
        build_shards_for_source(root / source, T=T, stride=8, examples_per_shard=64)
    return root


def test_batch_size_and_split_match_spec(shard_root: Path) -> None:
    loader = make_mixed_loader(shard_root, w=0.25, batch_size=8, seed=0)
    assert loader.spec.n_robot == 2
    assert loader.spec.n_human == 6
    batch = next(iter(loader))
    src = np.asarray(batch["source_mask"])
    assert src.shape == (8,)
    assert src.sum() == 2
    assert batch["rgb"].shape == (8, T, 224, 224, 3)
    assert batch["box_state"].shape == (8, T, 7)
    assert batch["phase"].shape == (8, T, 1)
    assert batch["contact"].shape == (8, T, 3)
    assert batch["state_robot"].shape == (8, T, 50)
    assert batch["state_human"].shape == (8, T, 157)
    assert batch["action"].shape == (8, T, 20)


def test_mixing_ratio_holds_over_1000_batches(shard_root: Path) -> None:
    """The plan's headline test: over many batches, robot fraction ≈ w."""
    w = 0.25
    B = 8
    n_batches = 1000
    loader = make_mixed_loader(shard_root, w=w, batch_size=B, seed=0)
    robot_count = 0
    for i, batch in enumerate(loader):
        robot_count += int(np.asarray(batch["source_mask"]).sum())
        if i + 1 >= n_batches:
            break
    actual = robot_count / (n_batches * B)
    # Per the plan: ~ w ± 1%.  Per-batch n_robot is exactly round(w*B); the
    # only randomness is in WHICH samples get drawn, so the ratio is exact.
    np.testing.assert_allclose(actual, w, atol=0.01)


def test_determinism_given_same_seed(shard_root: Path) -> None:
    loader_a = make_mixed_loader(shard_root, w=0.5, batch_size=4, seed=42)
    loader_b = make_mixed_loader(shard_root, w=0.5, batch_size=4, seed=42)
    n = 20
    for _ in range(n):
        a = next(loader_a)
        b = next(loader_b)
        np.testing.assert_array_equal(a["source_mask"], b["source_mask"])
        np.testing.assert_array_equal(a["rgb"], b["rgb"])
        np.testing.assert_array_equal(a["box_state"], b["box_state"])
        np.testing.assert_array_equal(a["episode_ids"], b["episode_ids"])
        np.testing.assert_array_equal(a["window_starts"], b["window_starts"])


def test_different_seeds_diverge(shard_root: Path) -> None:
    loader_a = make_mixed_loader(shard_root, w=0.5, batch_size=4, seed=1)
    loader_b = make_mixed_loader(shard_root, w=0.5, batch_size=4, seed=2)
    a = next(loader_a)["episode_ids"]
    b = next(loader_b)["episode_ids"]
    # With 20 episodes each side and different seeds, the very first batch
    # almost certainly differs. (Probability of accidental match is < 1e-3.)
    assert not np.array_equal(a, b)


def test_loader_iterates_past_dataset_size(shard_root: Path) -> None:
    """The sampler should auto-reshuffle when one side's permutation is
    exhausted — the plan calls for "infinite" iteration during training."""
    loader = make_mixed_loader(shard_root, w=0.5, batch_size=4, seed=0)
    # Pull ~5x the dataset size of windows to force re-shuffle on both sides.
    seen = 0
    for _ in range(200):
        batch = next(loader)
        seen += batch["source_mask"].shape[0]
    assert seen == 800


def test_lower_and_upper_bound_w() -> None:
    assert lower_bound_w(100, 100) == pytest.approx(0.5)
    assert lower_bound_w(50, 150) == pytest.approx(0.25)

    # When M >> N, plan uses sqrt(N/M).
    w = upper_bound_w(N_robot=10, M_human=100)  # M/N=10 >5 -> sqrt
    assert w == pytest.approx(min(np.sqrt(10 / 100), 0.5))
    # Otherwise, the q-interpolated formula.
    w = upper_bound_w(N_robot=80, M_human=100, q=0.8)
    expected = 80 * 0.8 / ((1 - 0.8) * 100 + 80 * 0.8)
    assert w == pytest.approx(min(expected, 0.5))


def test_w_out_of_range_raises(shard_root: Path) -> None:
    with pytest.raises(ValueError):
        MixedBatchSpec.from_w(-0.1, batch_size=8)
    with pytest.raises(ValueError):
        MixedBatchSpec.from_w(1.5, batch_size=8)


def test_collate_keys_match_model_contract(shard_root: Path) -> None:
    """The keys this loader emits must be a superset of what the model's
    forward expects (vis is added by the trainer after DINO encoding)."""
    loader = make_mixed_loader(shard_root, w=0.5, batch_size=2, seed=0)
    batch = next(loader)
    expected = {
        "rgb", "box_state", "phase", "contact",
        "state_robot", "state_human", "action", "source_mask",
    }
    assert expected.issubset(batch.keys()), expected - batch.keys()
