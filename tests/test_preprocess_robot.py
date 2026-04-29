"""End-to-end tests for the holosoma → §1.1 HDF5 robot preprocessor.

We synthesise a "raw rollout" .npz that obeys the cross-env contract
declared in `cotrain.scripts.preprocess_robot.RAW_NPZ_KEYS`, run
`preprocess_dir`, then validate the resulting HDF5 with the §1.3
validator. This pins the bridge: any drift in the .npz schema will fail
this test before training is launched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cotrain.scripts.preprocess_robot import (
    RAW_NPZ_KEYS,
    RawRollout,
    convert_rollout,
    preprocess_dir,
)
from cotrain.scripts.validate_dataset import validate_episode


def _make_raw_rollout(
    out_path: Path,
    *,
    T: int = 100,
    fps: int = 50,
    episode_id: str = "robot_test_000000",
    seed: int = 0,
) -> Path:
    """Synthesise a plausible raw rollout. Box rises from 0.20 m → 0.55 m
    after frame 60 to exercise the LIFT/HOLD phases."""
    rng = np.random.default_rng(seed)

    obj_pos = np.zeros((T, 3), dtype=np.float32)
    obj_pos[:60, 2] = 0.20
    # Linear lift from 0.20 → 0.55 over frames 60..T.
    obj_pos[60:, 2] = np.linspace(0.20, 0.55, T - 60)
    obj_quat = np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)).astype(np.float32)

    head_pos = np.tile([0.0, 0.0, 1.5], (T, 1)).astype(np.float32)
    head_quat = np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)).astype(np.float32)

    # Hands start far, approach, then settle near the box at frame 50.
    hand_offset = np.linspace(1.0, 0.0, T)[:, None] * np.array([1.0, 0.0, 0.0])
    left_hand = obj_pos + hand_offset.astype(np.float32) + np.array([-0.1, 0.0, 0.0], dtype=np.float32)
    right_hand = obj_pos + hand_offset.astype(np.float32) + np.array([+0.1, 0.0, 0.0], dtype=np.float32)

    # Contact forces fire from frame 50 onwards.
    contact_l = np.zeros(T, dtype=np.float32)
    contact_l[50:] = 5.0
    contact_r = np.zeros(T, dtype=np.float32)
    contact_r[50:] = 5.0

    payload = {
        "dof_pos":                  rng.normal(size=(T, 29)).astype(np.float32) * 0.05,
        "dof_vel":                  rng.normal(size=(T, 29)).astype(np.float32) * 0.05,
        "root_pos_w":               rng.normal(size=(T, 3)).astype(np.float32) * 0.01,
        "root_quat_w_xyzw":         np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)).astype(np.float32),
        "root_lin_vel_w":           rng.normal(size=(T, 3)).astype(np.float32) * 0.05,
        "root_ang_vel_w":           rng.normal(size=(T, 3)).astype(np.float32) * 0.05,
        "action":                   rng.normal(size=(T, 29)).astype(np.float32) * 0.1,
        "object_pos_w":             obj_pos,
        "object_quat_w_xyzw":       obj_quat,
        "left_hand_pos_w":          left_hand,
        "right_hand_pos_w":         right_hand,
        "left_hand_contact_force":  contact_l,
        "right_hand_contact_force": contact_r,
        "head_pos_w":               head_pos,
        "head_quat_w_xyzw":         head_quat,
        "fps":                      np.int64(fps),
        "meta_success":             np.bool_(True),
        "meta_episode_id":          np.bytes_(episode_id),
        "meta_box_size_class":      np.bytes_("smallbox"),
        "meta_seed":                np.int64(seed),
    }
    np.savez(out_path, **payload)
    return out_path


def test_raw_npz_keys_match_writer(tmp_path: Path) -> None:
    """The synthesizer should emit every key the loader expects."""
    p = _make_raw_rollout(tmp_path / "rollout.npz")
    with np.load(p) as f:
        for k in RAW_NPZ_KEYS:
            assert k in f, f"raw .npz contract violated: missing {k!r}"


def test_convert_rollout_yields_expected_shapes(tmp_path: Path) -> None:
    p = _make_raw_rollout(tmp_path / "rollout.npz", T=100, fps=50)
    raw = RawRollout.from_npz(p, default_box_size_class="smallbox")
    out = convert_rollout(raw, target_hz=30.0)

    # 100 src frames @ 50 Hz → round(100 * 30/50) = 60 target frames.
    T_target = 60
    assert out["proprio"].shape == (T_target, 68)        # G1 29-DoF default D_p
    assert out["action"].shape == (T_target, 29)
    assert out["box_state"].shape == (T_target, 7)
    assert out["phase"].shape == (T_target, 1)
    assert out["contact_lift"].shape == (T_target, 3)
    # Phase is int8 nearest-resampled.
    assert out["phase"].dtype == np.int8


def test_phase_progression_includes_lift_after_resample(tmp_path: Path) -> None:
    """Sanity: the synthesised trajectory should yield non-trivial phases."""
    p = _make_raw_rollout(tmp_path / "rollout.npz", T=100, fps=50)
    raw = RawRollout.from_npz(p, default_box_size_class="smallbox")
    out = convert_rollout(raw, target_hz=30.0)
    phases_seen = set(int(x) for x in out["phase"][:, 0])
    # We constructed APPROACH → REACH → CONTACT → LIFT → HOLD; at least
    # APPROACH (or REACH) and HOLD must be present, and a contact step.
    assert 0 in phases_seen or 1 in phases_seen, phases_seen
    assert 2 in phases_seen, phases_seen
    assert 3 in phases_seen or 4 in phases_seen, phases_seen


def test_full_pipeline_passes_validator(tmp_path: Path) -> None:
    """The full pipeline: 3 rollouts in → 3 episode HDF5s + manifest. Each
    HDF5 must pass the §1.3 validator."""
    rollouts_dir = tmp_path / "raw"
    rollouts_dir.mkdir()
    for i in range(3):
        _make_raw_rollout(
            rollouts_dir / f"rollout_{i:03d}.npz",
            T=100, fps=50, episode_id=f"robot_{i:06d}", seed=i,
        )

    out_root = tmp_path / "datasets"
    n = preprocess_dir(rollouts_dir, out_root, target_hz=30.0)
    assert n == 3

    robot_dir = out_root / "robot"
    eps = sorted(robot_dir.glob("ep_*.h5"))
    assert len(eps) == 3
    assert (robot_dir / "manifest.parquet").exists()

    for ep_path in eps:
        rep = validate_episode(ep_path)
        assert rep.ok, (ep_path, rep.errors)
        assert rep.source == "robot"
        # 30 Hz target × 2 sec ≈ 60 frames.
        assert rep.n_steps == 60


def test_missing_npz_keys_raises(tmp_path: Path) -> None:
    """A rollout missing a required key surfaces immediately."""
    p = _make_raw_rollout(tmp_path / "rollout.npz")
    # Re-pack without `action` to break the contract.
    with np.load(p) as f:
        items = {k: np.asarray(f[k]) for k in f.files if k != "action"}
    bad = tmp_path / "broken.npz"
    np.savez(bad, **items)
    with pytest.raises(ValueError, match="raw rollout missing keys"):
        RawRollout.from_npz(bad, default_box_size_class="smallbox")


def test_robot_hdf5_has_no_rgb(tmp_path: Path) -> None:
    """§1.1 contract update: robot rollouts have no rgb dataset."""
    import h5py

    rollouts_dir = tmp_path / "raw"
    rollouts_dir.mkdir()
    _make_raw_rollout(rollouts_dir / "rollout_000.npz")
    out_root = tmp_path / "datasets"
    preprocess_dir(rollouts_dir, out_root)
    ep = next((out_root / "robot").glob("ep_*.h5"))
    with h5py.File(ep, "r") as f:
        assert "rgb" not in f, "robot HDF5 must not contain rgb (§1.1)"
        assert "human_kin" not in f, "robot HDF5 must not contain human_kin"
        assert "proprio" in f and "action" in f
        assert f["proprio"].shape[1] == 68 and f["action"].shape[1] == 29
