"""Pre-training dataset validator (PROJECT_PLAN_1.md §1.3).

Asserted invariants — per the plan, training is blocked until these all hold:

1. Every required array exists for the episode's `source`, with the correct
   leading dim T and the per-step shape/dtype declared in `ARRAY_SPEC`.
2. All per-timestep arrays in an episode share the same T.
3. `phase` values are a subset of {0,1,2,3,4}.
4. `contact_lift` values lie in {0, 1}.
5. `box_state` translation lies in `BOX_STATE_TRANS_RANGE` and the quaternion
   has unit magnitude (within `BOX_STATE_QUAT_TOL`). This is the cross-source
   sanity check called out in §1.3 — divergent box-state ranges between robot
   and human break the shared bridge.
6. `meta/episode_id`, `meta/source`, `meta/success` are present.
7. The manifest's `hash` column matches the actual file md5.

CLI usage:
    python -m cotrain.scripts.validate_dataset --root datasets

Exits non-zero with a stack of error messages on the first failed file.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from cotrain.data.schemas import (
    ARRAY_SPEC,
    BOX_STATE_QUAT_TOL,
    BOX_STATE_TRANS_RANGE,
    CONTACT_VALUES,
    MANIFEST_FILENAME,
    PHASE_VALUES,
    Source,
    array_dtype,
    array_shape_per_step,
    file_md5,
    read_manifest,
    required_for,
)


class ValidationError(Exception):
    pass


@dataclass
class EpisodeReport:
    path: Path
    source: Source | None = None
    n_steps: int | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _read_meta_str(group: h5py.Group, key: str) -> str:
    raw = group[key][()]
    return raw.decode() if isinstance(raw, (bytes, bytearray, np.bytes_)) else str(raw)


def _check_array(
    rep: EpisodeReport,
    f: h5py.File,
    name: str,
    expected_T: int | None,
) -> int | None:
    """Returns this array's leading dim T (or None if checks fail catastrophically)."""
    if name not in f:
        rep.errors.append(f"missing required dataset '{name}'")
        return None
    ds = f[name]
    arr_shape: tuple[int, ...] = tuple(ds.shape)
    if not arr_shape:
        rep.errors.append(f"'{name}' is a scalar; expected at least 1-D with leading T")
        return None
    T = arr_shape[0]

    spec_per_step = array_shape_per_step(name)
    actual_per_step = arr_shape[1:]
    if len(actual_per_step) != len(spec_per_step):
        rep.errors.append(
            f"'{name}' rank mismatch: got {arr_shape}, expected leading T + {spec_per_step}"
        )
    else:
        for i, (got, want) in enumerate(zip(actual_per_step, spec_per_step)):
            if want is not None and got != want:
                rep.errors.append(
                    f"'{name}' axis {i + 1} mismatch: got {got}, expected {want}"
                )

    expected_dtype = array_dtype(name)
    if np.dtype(ds.dtype) != expected_dtype:
        rep.errors.append(f"'{name}' dtype mismatch: got {ds.dtype}, expected {expected_dtype}")

    if expected_T is not None and T != expected_T:
        rep.errors.append(f"'{name}' leading dim {T} != other arrays' T {expected_T}")
    return T


def validate_episode(path: Path, expected_hash: str | None = None) -> EpisodeReport:
    rep = EpisodeReport(path=path)
    try:
        with h5py.File(path, "r") as f:
            if "meta" not in f:
                rep.errors.append("missing 'meta' group")
                return rep
            meta = f["meta"]
            for k in ("episode_id", "source", "success"):
                if k not in meta:
                    rep.errors.append(f"missing 'meta/{k}'")
            if "source" in meta:
                src_str = _read_meta_str(meta, "source")
                if src_str not in ("robot", "human"):
                    rep.errors.append(f"meta/source = {src_str!r}, expected 'robot' or 'human'")
                else:
                    rep.source = src_str  # type: ignore[assignment]

            if rep.source is None:
                return rep

            T_seen: int | None = None
            for name in required_for(rep.source):
                T = _check_array(rep, f, name, T_seen)
                if T is not None and T_seen is None:
                    T_seen = T
            rep.n_steps = T_seen

            # Range checks on bridge slots.
            if "phase" in f:
                ph = f["phase"][()]
                bad = set(np.unique(ph).tolist()) - set(PHASE_VALUES)
                if bad:
                    rep.errors.append(f"'phase' contains illegal values {sorted(bad)}")
            if "contact_lift" in f:
                cl = f["contact_lift"][()]
                if not np.isin(cl, np.array(CONTACT_VALUES, dtype=cl.dtype)).all():
                    bad_vals = np.unique(cl[~np.isin(cl, np.array(CONTACT_VALUES, dtype=cl.dtype))])
                    rep.errors.append(
                        f"'contact_lift' must be in {CONTACT_VALUES}, got values like {bad_vals[:5].tolist()}"
                    )
            if "box_state" in f:
                bs = f["box_state"][()]
                trans = bs[..., :3]
                lo, hi = BOX_STATE_TRANS_RANGE
                if not (np.all(trans >= lo) and np.all(trans <= hi)):
                    rep.errors.append(
                        f"'box_state' translation outside {BOX_STATE_TRANS_RANGE}: "
                        f"min={trans.min():.3f} max={trans.max():.3f}"
                    )
                quat = bs[..., 3:]
                qnorm = np.linalg.norm(quat, axis=-1)
                if not np.all(np.abs(qnorm - 1.0) < BOX_STATE_QUAT_TOL):
                    rep.errors.append(
                        f"'box_state' quaternion non-unit: "
                        f"|q| range [{qnorm.min():.3f}, {qnorm.max():.3f}], tol={BOX_STATE_QUAT_TOL}"
                    )
    except OSError as e:
        rep.errors.append(f"failed to open HDF5: {e}")
        return rep

    if expected_hash is not None:
        actual_hash = file_md5(path)
        if actual_hash != expected_hash:
            rep.errors.append(
                f"manifest hash mismatch: file md5 {actual_hash} != manifest {expected_hash}"
            )
    return rep


def validate_dataset(root: Path, *, fail_fast: bool = True) -> list[EpisodeReport]:
    """Validate every episode under root/{robot,human}. Returns per-episode reports."""
    all_reports: list[EpisodeReport] = []
    for source in ("robot", "human"):
        ds_dir = root / source
        if not ds_dir.is_dir():
            raise ValidationError(f"missing dataset dir: {ds_dir}")
        manifest_path = ds_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ValidationError(f"missing manifest: {manifest_path}")
        manifest = read_manifest(manifest_path).to_pylist()
        for row in manifest:
            ep_path = ds_dir / row["path"]
            if not ep_path.is_file():
                raise ValidationError(f"manifest references missing file: {ep_path}")
            rep = validate_episode(ep_path, expected_hash=row["hash"])
            if row["source"] != source:
                rep.errors.append(
                    f"manifest source {row['source']!r} doesn't match dir {source!r}"
                )
            if rep.source is not None and rep.source != row["source"]:
                rep.errors.append(
                    f"meta/source {rep.source!r} != manifest source {row['source']!r}"
                )
            if rep.n_steps is not None and rep.n_steps != row["n_steps"]:
                rep.errors.append(
                    f"n_steps {rep.n_steps} != manifest {row['n_steps']}"
                )
            all_reports.append(rep)
            if fail_fast and not rep.ok:
                return all_reports
    return all_reports


def _format_failures(reports: list[EpisodeReport]) -> str:
    failed = [r for r in reports if not r.ok]
    if not failed:
        return ""
    lines = [f"{len(failed)} episode(s) failed validation:"]
    for r in failed:
        lines.append(f"  {r.path}")
        for e in r.errors:
            lines.append(f"    - {e}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("datasets"),
                    help="dataset root containing robot/ and human/ subdirs")
    ap.add_argument("--no-fail-fast", action="store_true",
                    help="report all failures instead of stopping at the first")
    args = ap.parse_args()

    reports = validate_dataset(args.root, fail_fast=not args.no_fail_fast)
    msg = _format_failures(reports)
    if msg:
        print(msg, file=sys.stderr)
        return 1
    print(f"OK: validated {len(reports)} episodes under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
