"""Smoke tests for the §1.3 validator.

Per PROJECT_PLAN_1.md §8 step 1: 'validator catches an intentionally-broken
episode'. We verify both directions — clean synthetic data passes, common
breakage modes fail with the right error category.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from cotrain.scripts.generate_synthetic_data import generate
from cotrain.scripts.validate_dataset import (
    ValidationError,
    validate_dataset,
    validate_episode,
)


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth_ds")
    generate(root, n_robot=3, n_human=3, seed=42, T_range=(40, 60))
    return root


def test_clean_synthetic_dataset_passes(synthetic_root: Path) -> None:
    reports = validate_dataset(synthetic_root)
    assert all(r.ok for r in reports), "\n".join(
        f"{r.path}: {r.errors}" for r in reports if not r.ok
    )
    assert len(reports) == 6


def test_validator_catches_illegal_phase(synthetic_root: Path, tmp_path: Path) -> None:
    src = next((synthetic_root / "robot").glob("ep_*.h5"))
    dst = tmp_path / src.name
    dst.write_bytes(src.read_bytes())
    with h5py.File(dst, "r+") as f:
        ph = f["phase"]
        ph[0, 0] = np.int8(99)  # not in {0..4}
    rep = validate_episode(dst)
    assert not rep.ok
    assert any("phase" in e.lower() and "illegal" in e.lower() for e in rep.errors), rep.errors


def test_validator_catches_shape_mismatch(synthetic_root: Path, tmp_path: Path) -> None:
    src = next((synthetic_root / "robot").glob("ep_*.h5"))
    dst = tmp_path / src.name
    dst.write_bytes(src.read_bytes())
    with h5py.File(dst, "r+") as f:
        # box_state should be (T, 7); rewrite with (T, 6).
        T = f["box_state"].shape[0]
        del f["box_state"]
        f.create_dataset("box_state", data=np.zeros((T, 6), dtype=np.float32))
    rep = validate_episode(dst)
    assert not rep.ok
    assert any("box_state" in e and "axis" in e for e in rep.errors), rep.errors


def test_validator_catches_dtype_mismatch(synthetic_root: Path, tmp_path: Path) -> None:
    src = next((synthetic_root / "human").glob("ep_*.h5"))
    dst = tmp_path / src.name
    dst.write_bytes(src.read_bytes())
    with h5py.File(dst, "r+") as f:
        T = f["phase"].shape[0]
        del f["phase"]
        # Phase as float32 instead of int8.
        f.create_dataset("phase", data=np.zeros((T, 1), dtype=np.float32))
    rep = validate_episode(dst)
    assert not rep.ok
    assert any("phase" in e and "dtype" in e for e in rep.errors), rep.errors


def test_validator_catches_box_state_out_of_range(synthetic_root: Path, tmp_path: Path) -> None:
    src = next((synthetic_root / "human").glob("ep_*.h5"))
    dst = tmp_path / src.name
    dst.write_bytes(src.read_bytes())
    with h5py.File(dst, "r+") as f:
        bs = f["box_state"][()]
        bs[0, 0] = 99.0  # way outside the camera-frame range
        f["box_state"][...] = bs
    rep = validate_episode(dst)
    assert not rep.ok
    assert any("box_state" in e and "translation" in e for e in rep.errors), rep.errors


def test_validator_catches_manifest_hash_mismatch(synthetic_root: Path) -> None:
    """Mutating an episode without rewriting the manifest must surface."""
    ep = next((synthetic_root / "robot").glob("ep_*.h5"))
    with h5py.File(ep, "r+") as f:
        # Touch a benign array; this changes the file md5.
        f["box_state"][0, 0] = f["box_state"][0, 0] + np.float32(0.01)
    reports = validate_dataset(synthetic_root, fail_fast=False)
    failures = [r for r in reports if not r.ok]
    assert failures, "expected hash-mismatch failure, got none"
    assert any(
        any("manifest hash mismatch" in err for err in r.errors) for r in failures
    ), [r.errors for r in failures]


def test_missing_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / "robot").mkdir()
    (tmp_path / "human").mkdir()
    with pytest.raises(ValidationError, match="missing manifest"):
        validate_dataset(tmp_path)
