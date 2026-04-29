"""Stage 0 (holosoma) rollout → §1.1 HDF5 robot episodes.

The cross-env workflow
----------------------
holosoma's WBT lives in its own conda env (`hssim`) and depends on Isaac
Sim, which we don't want to drag into cotrain's JAX-only `.venv`. We
split the work along a `.npz` boundary:

  step 1 (in hssim env, holosoma side):
      run policy in sim, capture per-step state, write
      `<rollouts_dir>/<episode_id>.npz` per episode using `RAW_NPZ_KEYS`
      below (the contract this script reads).

  step 2 (in cotrain's .venv, this script):
      walk `<rollouts_dir>/*.npz`, apply the §3.1 bridge helpers
      (`cotrain.data.pipelines.bridge`), resample 50→30 Hz, and write
      `<out_root>/robot/ep_XXXXXX.h5 + manifest.parquet` per the §1.1
      contract.

Splitting at a numpy boundary keeps cotrain pure JAX and lets us unit-test
the conversion math without Isaac Sim. The holosoma-side capture loop
that produces the raw `.npz` lives in `human-humanoid-cotrain/scripts/`.

§1.1 contract robot HDF5 contents (per `cotrain.data.schemas.episode`):
  - proprio (T, D_p=68): concat of [dof_pos(29), dof_vel(29),
    root_quat_xyzw(4), root_lin_vel(3), root_ang_vel(3)] — holosoma's
    convention is xyzw and we keep it (proprio is opaque to the model).
  - action (T, D_a=29): joint position targets.
  - box_state (T, 7): camera-frame pose `[x, y, z, qw, qx, qy, qz]`
    derived from world-frame object + a virtual head-mounted camera
    frame on the G1 head body.
  - phase (T, 1) int8: §1.1 5-phase enum, derived from hand-box
    distance + contact_lift.
  - contact_lift (T, 3) float32: from contact-force magnitudes + box
    height.
  - meta/{success, episode_id, source='robot'}.

robot HDF5s do NOT contain rgb (PROJECT_PLAN_1.md §1.1; the §3.4 mask
zeros out the VIS slot for every robot sample anyway).

Usage:
    python -m cotrain.scripts.preprocess_robot \\
        --rollouts-dir /path/to/holosoma/raw_rollouts \\
        --out-root datasets \\
        --src-hz 50 --target-hz 30 --box-size-class smallbox
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from cotrain.data.pipelines.bridge import (
    derive_contact_lift,
    derive_phase,
    pack_box_state_camera_frame,
    resample_to_rate,
    world_to_camera_pose,
)
from cotrain.data.schemas import (
    EpisodeMeta,
    file_md5,
    write_manifest,
)


# --- raw .npz contract ---------------------------------------------------

RAW_NPZ_KEYS = (
    "dof_pos",                  # (T, 29) float32 — actual joint positions, world frame independent
    "dof_vel",                  # (T, 29) float32
    "root_pos_w",               # (T, 3)   — robot root (pelvis) position, world frame
    "root_quat_w_xyzw",         # (T, 4)   — robot root orientation, world frame, xyzw
    "root_lin_vel_w",           # (T, 3)
    "root_ang_vel_w",           # (T, 3)
    "action",                   # (T, 29) float32 — joint position targets sent to the PD controller
    "object_pos_w",             # (T, 3)
    "object_quat_w_xyzw",       # (T, 4)
    "left_hand_pos_w",          # (T, 3)   — for hand-box distance
    "right_hand_pos_w",         # (T, 3)
    "left_hand_contact_force",  # (T,) float32 magnitude
    "right_hand_contact_force", # (T,) float32
    "head_pos_w",               # (T, 3)   — virtual camera origin
    "head_quat_w_xyzw",         # (T, 4)   — virtual camera orientation
    "fps",                      # () int   — capture rate
)


@dataclass(frozen=True)
class RawRollout:
    """In-memory view of one raw .npz episode."""
    episode_id: str
    fps: int
    arrays: dict[str, np.ndarray]
    success: bool
    box_size_class: str
    seed: int

    @classmethod
    def from_npz(cls, path: Path, *, default_box_size_class: str) -> "RawRollout":
        with np.load(path, allow_pickle=False) as f:
            missing = [k for k in RAW_NPZ_KEYS if k not in f]
            if missing:
                raise ValueError(f"{path}: raw rollout missing keys {missing}")
            arrays = {k: np.asarray(f[k]) for k in RAW_NPZ_KEYS if k != "fps"}
            fps = int(np.asarray(f["fps"]).item())
            # Optional metadata; fall back to safe defaults so the script
            # works on rollouts the holosoma-side hasn't been updated to
            # emit yet.
            success = bool(f["meta_success"].item()) if "meta_success" in f else True
            episode_id = (
                f["meta_episode_id"].item().decode()
                if "meta_episode_id" in f and isinstance(f["meta_episode_id"].item(), bytes)
                else f["meta_episode_id"].item() if "meta_episode_id" in f
                else path.stem
            )
            box_size_class = (
                f["meta_box_size_class"].item().decode()
                if "meta_box_size_class" in f and isinstance(f["meta_box_size_class"].item(), bytes)
                else f["meta_box_size_class"].item() if "meta_box_size_class" in f
                else default_box_size_class
            )
            seed = int(f["meta_seed"].item()) if "meta_seed" in f else 0
        return cls(
            episode_id=str(episode_id),
            fps=fps,
            arrays=arrays,
            success=success,
            box_size_class=str(box_size_class),
            seed=seed,
        )


# --- conversion ----------------------------------------------------------

def _build_proprio(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """(T, D_p) = concat(dof_pos, dof_vel, root_quat_xyzw, root_lin_vel, root_ang_vel)."""
    parts = [
        arrays["dof_pos"], arrays["dof_vel"],
        arrays["root_quat_w_xyzw"],
        arrays["root_lin_vel_w"], arrays["root_ang_vel_w"],
    ]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def _hand_box_distance(arrays: dict[str, np.ndarray]) -> np.ndarray:
    obj = arrays["object_pos_w"]
    dl = np.linalg.norm(arrays["left_hand_pos_w"]  - obj, axis=-1)
    dr = np.linalg.norm(arrays["right_hand_pos_w"] - obj, axis=-1)
    return np.minimum(dl, dr).astype(np.float32)


def _box_state_camera_frame(arrays: dict[str, np.ndarray]) -> np.ndarray:
    pos_c, quat_c_xyzw = world_to_camera_pose(
        arrays["object_pos_w"],
        arrays["object_quat_w_xyzw"],
        arrays["head_pos_w"],
        arrays["head_quat_w_xyzw"],
    )
    return pack_box_state_camera_frame(pos_c, quat_c_xyzw)


def convert_rollout(
    raw: RawRollout,
    *,
    target_hz: float = 30.0,
    contact_force_threshold: float = 1.0,
    lift_height: float = 0.05,
    reach_distance: float = 0.30,
    hold_frames: int = 30,
) -> dict[str, np.ndarray]:
    """Apply bridge helpers + resample to produce per-slot arrays at target_hz.

    Returns a dict with §1.1 keys: proprio, action, box_state, phase,
    contact_lift. Source-frame is the camera (head body) for box_state.
    Caller is responsible for writing the dict to HDF5 + manifest.
    """
    a = raw.arrays

    # 1. Bridge signals at the source rate (50 Hz typically).
    initial_box_height = float(a["object_pos_w"][0, 2])
    contact_lift_src = derive_contact_lift(
        left_hand_contact_force=a["left_hand_contact_force"].astype(np.float32),
        right_hand_contact_force=a["right_hand_contact_force"].astype(np.float32),
        box_height=a["object_pos_w"][:, 2].astype(np.float32),
        initial_box_height=initial_box_height,
        contact_force_threshold=contact_force_threshold,
        lift_height=lift_height,
    )
    phase_src = derive_phase(
        hand_box_distance=_hand_box_distance(a),
        contact_lift=contact_lift_src,
        reach_distance=reach_distance,
        hold_frames=hold_frames,
    )

    # 2. Pack continuous fields.
    proprio_src   = _build_proprio(a)
    action_src    = a["action"].astype(np.float32)
    box_state_src = _box_state_camera_frame(a)

    # 3. Resample to target rate. Continuous → linear; phase enum → nearest.
    return {
        "proprio":      resample_to_rate(proprio_src,   src_hz=raw.fps, target_hz=target_hz),
        "action":       resample_to_rate(action_src,    src_hz=raw.fps, target_hz=target_hz),
        "box_state":    resample_to_rate(box_state_src, src_hz=raw.fps, target_hz=target_hz),
        "phase":        resample_to_rate(phase_src,     src_hz=raw.fps, target_hz=target_hz, mode="nearest"),
        "contact_lift": resample_to_rate(contact_lift_src, src_hz=raw.fps, target_hz=target_hz, mode="nearest"),
    }


# --- HDF5 writing --------------------------------------------------------

def write_episode_h5(
    out_path: Path,
    *,
    converted: dict[str, np.ndarray],
    raw: RawRollout,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for name in ("proprio", "action", "box_state", "phase", "contact_lift"):
            f.create_dataset(name, data=converted[name],
                             compression="gzip", compression_opts=4)
        meta = f.create_group("meta")
        meta.create_dataset("success",    data=np.bool_(raw.success))
        meta.create_dataset("episode_id", data=np.bytes_(raw.episode_id))
        meta.create_dataset("source",     data=np.bytes_("robot"))


def preprocess_dir(
    rollouts_dir: Path,
    out_root: Path,
    *,
    target_hz: float = 30.0,
    box_size_class_default: str = "smallbox",
    glob_pattern: str = "*.npz",
) -> int:
    """Walk every rollout under `rollouts_dir`, convert, and write
    `<out_root>/robot/ep_NNNNNN.h5 + manifest.parquet`. Returns the count
    of successfully-written episodes."""
    rollouts_dir = Path(rollouts_dir)
    out_dir = Path(out_root) / "robot"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(rollouts_dir.glob(glob_pattern))
    if not npz_files:
        raise FileNotFoundError(f"no {glob_pattern} files under {rollouts_dir}")

    rows: list[dict] = []
    today = dt.date.today().isoformat()
    for i, npz_path in enumerate(tqdm(npz_files, desc="preprocess robot")):
        raw = RawRollout.from_npz(npz_path, default_box_size_class=box_size_class_default)
        converted = convert_rollout(raw, target_hz=target_hz)

        ep_filename = f"ep_{i:06d}.h5"
        ep_path = out_dir / ep_filename
        write_episode_h5(ep_path, converted=converted, raw=raw)

        digest = file_md5(ep_path)
        EpisodeMeta(  # raises on malformed metadata
            episode_id=raw.episode_id,
            source="robot",
            success=raw.success,
            n_steps=converted["phase"].shape[0],
            box_size_class=raw.box_size_class,
            randomization_seed=raw.seed,
            recording_date=today,
            hash=digest,
        )
        rows.append({
            "episode_id":         raw.episode_id,
            "source":             "robot",
            "n_steps":            int(converted["phase"].shape[0]),
            "success":            bool(raw.success),
            "box_size_class":     raw.box_size_class,
            "randomization_seed": int(raw.seed),
            "recording_date":     today,
            "hash":               digest,
            "path":               ep_filename,
        })
    write_manifest(rows, out_dir)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts-dir", type=Path, required=True,
                    help="directory of holosoma raw .npz rollouts")
    ap.add_argument("--out-root", type=Path, default=Path("datasets"),
                    help="dataset root; episodes go to <out-root>/robot/")
    ap.add_argument("--src-hz", type=float, default=50.0,
                    help="(informational; the actual src rate is read from each .npz)")
    ap.add_argument("--target-hz", type=float, default=30.0)
    ap.add_argument("--box-size-class", type=str, default="smallbox",
                    help="fallback when meta_box_size_class is absent from the .npz")
    ap.add_argument("--glob", type=str, default="*.npz")
    args = ap.parse_args()
    n = preprocess_dir(
        args.rollouts_dir, args.out_root,
        target_hz=args.target_hz,
        box_size_class_default=args.box_size_class,
        glob_pattern=args.glob,
    )
    print(f"wrote {n} robot episodes to {args.out_root / 'robot'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
