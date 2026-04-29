"""Generate synthetic robot/human episodes for development.

Stages 0 (sim rollouts) and 1 (Aria capture) are upstream of this plan and
will eventually produce real HDF5 episodes matching the §1.1 contract. Until
that data lands, this script fabricates shape/range-compatible episodes so
the rest of the stack (tokenizer, sampler, transformer) can be built and
unit-tested.

The contract this script obeys is exactly PROJECT_PLAN_1.md §1.1 — anything
that holds for a real episode must hold for these synthetic ones, especially
the bridge slots (`box_state`, `phase`, `contact_lift`) which must use the
same numeric range for both sources or the shared bridge collapses (§3.1).

Usage:
    python -m cotrain.scripts.generate_synthetic_data \\
        --out datasets --robot-episodes 30 --human-episodes 30 --seed 0
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from cotrain.data.schemas import (
    DEFAULT_D_A,
    DEFAULT_D_H,
    DEFAULT_D_P,
    EpisodeMeta,
    Phase,
    RGB_H,
    RGB_W,
    Source,
    file_md5,
    write_manifest,
)


def _phase_trajectory(T: int, rng: np.random.Generator) -> np.ndarray:
    """Monotonic phase sequence over T steps, hitting all 5 phases."""
    # Random partition of T into 5 positive segments.
    cuts = np.sort(rng.choice(np.arange(1, T), size=4, replace=False))
    seg_lens = np.diff(np.concatenate([[0], cuts, [T]]))
    phases = np.repeat(np.array([p.value for p in Phase], dtype=np.int8), seg_lens)
    return phases.reshape(T, 1)


def _contact_from_phase(phases: np.ndarray) -> np.ndarray:
    """Bridge: deterministic mapping phase -> [left_contact, right_contact, lifted]."""
    p = phases[:, 0]
    contact = (p >= Phase.CONTACT.value).astype(np.float32)
    lifted = (p >= Phase.LIFT.value).astype(np.float32)
    out = np.stack([contact, contact, lifted], axis=-1)
    # Smooth the LIFT->HOLD lifted onset over a couple of frames so it's
    # not a perfect step (the bridge tolerates noise; perfect zeros/ones
    # might let downstream code over-fit on synthetic regularity).
    return out.astype(np.float32)


def _box_trajectory(T: int, phases: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Plausible box pose in the camera frame: starts forward+down, rises after LIFT."""
    p = phases[:, 0]
    t_norm = np.linspace(0, 1, T, dtype=np.float32)
    # Translation: starts ~(0.3, 0.0, 0.4); during APPROACH/REACH drifts toward
    # (0.0, 0.0, 0.25); LIFT/HOLD rises to (0.0, 0.0, 0.55). Stays in the
    # camera-frame range called out by validators.
    x = 0.3 * (1.0 - np.minimum(t_norm * 1.5, 1.0))
    y = 0.05 * np.sin(2 * np.pi * t_norm) + rng.normal(0, 0.005, size=T)
    z_base = 0.4 - 0.15 * np.minimum(t_norm * 1.5, 1.0)
    z_lift = 0.30 * (p >= Phase.LIFT.value).astype(np.float32)
    z = z_base + z_lift
    trans = np.stack([x, y, z], axis=-1).astype(np.float32)
    # Quaternion: identity + small noise; renormalize to unit.
    quat = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (T, 1))
    quat = quat + rng.normal(0, 0.02, size=quat.shape).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    return np.concatenate([trans, quat], axis=-1).astype(np.float32)  # (T, 7)


def _proprio(T: int, D: int, rng: np.random.Generator) -> np.ndarray:
    walk = np.cumsum(rng.normal(0, 0.02, size=(T, D)).astype(np.float32), axis=0)
    return walk * 0.1


def _human_kin(T: int, D: int, rng: np.random.Generator) -> np.ndarray:
    return _proprio(T, D, rng)  # same shape; different D — distinct STATE head will project both


def _action(T: int, D: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0, 0.3, size=(T, D)).astype(np.float32)


def _rgb(T: int, phases: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap synthetic RGB: a base color tinted by phase + low-amplitude noise.

    Stored uint8 with HDF5 gzip(level=4) compression — keeps disk per
    episode at ~hundreds of KB instead of ~30 MB raw.
    """
    p = phases[:, 0]
    base = np.zeros((T, RGB_H, RGB_W, 3), dtype=np.uint8)
    # Phase tint: distinct color block in the top-left corner per phase id.
    tints = np.array([
        [40, 40, 40],
        [200, 50, 50],
        [50, 200, 50],
        [50, 50, 200],
        [200, 200, 50],
    ], dtype=np.uint8)
    for t in range(T):
        base[t] = tints[p[t]]
    noise = rng.integers(0, 16, size=base.shape, dtype=np.uint8)
    return np.clip(base.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)


def _write_episode(
    out_path: Path,
    *,
    source: Source,
    episode_id: str,
    seed: int,
    T_range: tuple[int, int] = (60, 180),
    box_size_class: str = "medium",
    success: bool | None = None,
) -> dict:
    """Write one ep_XXXXXX.h5 and return the manifest row dict for it."""
    rng = np.random.default_rng(seed)
    T = int(rng.integers(T_range[0], T_range[1] + 1))
    phases = _phase_trajectory(T, rng)
    box_state = _box_trajectory(T, phases, rng)
    contact_lift = _contact_from_phase(phases)
    rgb = _rgb(T, phases, rng)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        # Bridge slots — written identically for both sources.
        f.create_dataset("box_state",    data=box_state,    compression="gzip", compression_opts=4)
        f.create_dataset("phase",        data=phases,       compression="gzip", compression_opts=4)
        f.create_dataset("contact_lift", data=contact_lift, compression="gzip", compression_opts=4)
        f.create_dataset("rgb",          data=rgb,          compression="gzip", compression_opts=4)
        if source == "robot":
            f.create_dataset("proprio", data=_proprio(T, DEFAULT_D_P, rng),
                             compression="gzip", compression_opts=4)
            f.create_dataset("action",  data=_action(T, DEFAULT_D_A, rng),
                             compression="gzip", compression_opts=4)
        else:
            f.create_dataset("human_kin", data=_human_kin(T, DEFAULT_D_H, rng),
                             compression="gzip", compression_opts=4)

        success_val = bool(success if success is not None else (rng.random() < 0.85))
        meta = f.create_group("meta")
        meta.create_dataset("success",    data=np.bool_(success_val))
        meta.create_dataset("episode_id", data=np.bytes_(episode_id))
        meta.create_dataset("source",     data=np.bytes_(source))

    digest = file_md5(out_path)
    EpisodeMeta(  # raises if anything is malformed
        episode_id=episode_id,
        source=source,
        success=success_val,
        n_steps=T,
        box_size_class=box_size_class,
        randomization_seed=seed,
        recording_date=dt.date.today().isoformat(),
        hash=digest,
    )
    return {
        "episode_id":         episode_id,
        "source":             source,
        "n_steps":            T,
        "success":            success_val,
        "box_size_class":     box_size_class,
        "randomization_seed": seed,
        "recording_date":     dt.date.today().isoformat(),
        "hash":               digest,
        "path":               out_path.name,
    }


def generate(
    out_root: Path,
    n_robot: int,
    n_human: int,
    seed: int,
    T_range: tuple[int, int] = (60, 180),
) -> None:
    rng_master = np.random.default_rng(seed)
    for source, n in (("robot", n_robot), ("human", n_human)):
        ds_dir = out_root / source
        ds_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for i in tqdm(range(n), desc=f"gen {source}"):
            ep_id = f"{source}_{i:06d}"
            ep_seed = int(rng_master.integers(0, 2**31 - 1))
            row = _write_episode(
                ds_dir / f"ep_{i:06d}.h5",
                source=source,  # type: ignore[arg-type]
                episode_id=ep_id,
                seed=ep_seed,
                T_range=T_range,
            )
            rows.append(row)
        write_manifest(rows, ds_dir)
        print(f"  -> {ds_dir} ({len(rows)} episodes)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("datasets"))
    p.add_argument("--robot-episodes", type=int, default=10)
    p.add_argument("--human-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-min", type=int, default=60)
    p.add_argument("--t-max", type=int, default=180)
    args = p.parse_args()
    generate(args.out, args.robot_episodes, args.human_episodes, args.seed,
             T_range=(args.t_min, args.t_max))


if __name__ == "__main__":
    main()
