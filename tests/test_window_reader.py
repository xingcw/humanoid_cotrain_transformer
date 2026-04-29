"""Round-trip tests for the §8.4 window reader.

Plan §8 step 4 spec: 'reads HDF5 → produces a (B, 6T, d_model) array.
*Test:* round-trip a known episode and check no dimension shuffling.'
This file covers the host-side half — read a window from a synthetic
episode and verify each per-slot array is bit-identical to the source
HDF5's slice. The device-side assembly is covered by
test_sequence_assembly.py.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from cotrain.data.pipelines import assert_window_shapes, read_window
from cotrain.data.schemas import DEFAULT_D_A, DEFAULT_D_H, DEFAULT_D_P
from cotrain.scripts.generate_synthetic_data import generate


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth_for_window")
    generate(root, n_robot=2, n_human=2, seed=7, T_range=(40, 60))
    return root


def _first_episode(root: Path, source: str) -> Path:
    return next((root / source).glob("ep_*.h5"))


def test_round_trip_robot(synth_root: Path) -> None:
    """Each per-slot array must equal the HDF5 slice exactly."""
    ep = _first_episode(synth_root, "robot")
    start, T = 5, 16
    win = read_window(ep, start, T)
    with h5py.File(ep, "r") as f:
        for name in ("box_state", "phase", "contact_lift", "proprio", "action"):
            np.testing.assert_array_equal(
                win[name], f[name][start:start + T],
                err_msg=f"{name} drifted from HDF5",
            )
        # Robot HDF5s no longer contain rgb (§1.1 contract update).
        assert "rgb" not in f
    # Padded human-only slots: rgb zeros + human_kin zeros, with the right shape.
    assert win["rgb"].shape == (T, 224, 224, 3) and win["rgb"].dtype == np.uint8
    assert (win["rgb"] == 0).all()
    assert win["human_kin"].shape == (T, DEFAULT_D_H)
    assert (win["human_kin"] == 0).all()
    assert win["source_is_robot"].item() is True
    assert_window_shapes(win, T)


def test_round_trip_human(synth_root: Path) -> None:
    ep = _first_episode(synth_root, "human")
    start, T = 0, 16
    win = read_window(ep, start, T)
    with h5py.File(ep, "r") as f:
        for name in ("rgb", "box_state", "phase", "contact_lift", "human_kin"):
            np.testing.assert_array_equal(win[name], f[name][start:start + T])
    # Robot-only slots are zero-padded for human samples.
    assert win["proprio"].shape == (T, DEFAULT_D_P)
    assert (win["proprio"] == 0).all()
    assert win["action"].shape == (T, DEFAULT_D_A)
    assert (win["action"] == 0).all()
    assert win["source_is_robot"].item() is False


def test_window_bounds_check(synth_root: Path) -> None:
    ep = _first_episode(synth_root, "robot")
    with h5py.File(ep, "r") as f:
        ep_len = f["phase"].shape[0]
    # T just barely fits at the tail.
    read_window(ep, ep_len - 16, 16)
    # One past should raise.
    with pytest.raises(IndexError):
        read_window(ep, ep_len - 15, 16)
    with pytest.raises(IndexError):
        read_window(ep, -1, 16)


def test_pad_other_source_disabled(synth_root: Path) -> None:
    ep = _first_episode(synth_root, "human")
    win = read_window(ep, 0, 16, pad_other_source=False)
    assert "proprio" not in win
    assert "action" not in win
    assert "human_kin" in win


def test_pad_other_source_disabled_for_robot(synth_root: Path) -> None:
    """Without padding, a robot window does not contain rgb / human_kin."""
    ep = _first_episode(synth_root, "robot")
    win = read_window(ep, 0, 16, pad_other_source=False)
    assert "rgb" not in win
    assert "human_kin" not in win
    assert "proprio" in win
    assert "action" in win
