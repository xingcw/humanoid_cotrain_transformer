"""HDF5 episodes → ArrayRecord shards (PROJECT_PLAN_1.md §4.1, §8 step 4a).

Walks `<root>/<source>/manifest.parquet`, generates a list of
`(episode, window_start)` examples per the supplied stride, and writes
them out as ArrayRecord shard files at `<root>/<source>/shards/`. Each
record is one window dict pickled via `cotrain.data.pipelines.shards`.

Why pre-tokenize at shard time (rather than slicing live in `__getitem__`):
TPU input pipelines are I/O-sensitive and `grain` works best with flat
shards of pre-extracted examples. The plan calls this out explicitly in
§4.1 — see the docstring there.

Usage:
    python -m cotrain.scripts.build_grain_shards \\
        --root datasets --T 16 --stride 8 --examples-per-shard 1024
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
from array_record.python.array_record_module import ArrayRecordWriter
from tqdm import tqdm

from cotrain.data.pipelines.shards import serialize_window
from cotrain.data.pipelines.window import read_window
from cotrain.data.schemas import (
    DEFAULT_D_A,
    DEFAULT_D_H,
    DEFAULT_D_P,
    MANIFEST_FILENAME,
    read_manifest,
)


def windows_for_episode(ep_path: Path, T: int, stride: int) -> list[int]:
    """Non-overlapping (or strided) start positions that fit one full T window."""
    with h5py.File(ep_path, "r") as f:
        ep_len = f["phase"].shape[0]
    if ep_len < T:
        return []
    return list(range(0, ep_len - T + 1, stride))


def build_shards_for_source(
    source_dir: Path,
    *,
    T: int,
    stride: int,
    examples_per_shard: int,
    D_p: int = DEFAULT_D_P,
    D_h: int = DEFAULT_D_H,
    D_a: int = DEFAULT_D_A,
) -> int:
    """Returns the total number of windows written under `source_dir/shards/`."""
    manifest_path = source_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    shards_dir = source_dir / "shards"
    if shards_dir.exists():
        # Always rebuild from scratch — safer than mixing stale + fresh shards.
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True)

    manifest = read_manifest(manifest_path).to_pylist()
    n_windows = 0
    shard_idx = 0
    writer: ArrayRecordWriter | None = None

    def _open_writer(idx: int) -> ArrayRecordWriter:
        path = shards_dir / f"{idx:05d}.array_record"
        return ArrayRecordWriter(str(path), "group_size:1")

    try:
        for row in tqdm(manifest, desc=f"shard {source_dir.name}"):
            ep_path = source_dir / row["path"]
            for start in windows_for_episode(ep_path, T, stride):
                if writer is None:
                    writer = _open_writer(shard_idx)
                window = read_window(ep_path, start, T, D_p=D_p, D_h=D_h, D_a=D_a)
                # Bake in window provenance so we can re-validate at sample time.
                window["episode_id"] = row["episode_id"]
                window["window_start"] = start
                writer.write(serialize_window(window))
                n_windows += 1
                if n_windows % examples_per_shard == 0:
                    writer.close()
                    writer = None
                    shard_idx += 1
    finally:
        if writer is not None:
            writer.close()

    return n_windows


def build_all(
    root: Path,
    *,
    T: int = 16,
    stride: int = 8,
    examples_per_shard: int = 1024,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in ("robot", "human"):
        src_dir = root / source
        if not src_dir.is_dir():
            print(f"  skip {src_dir} (does not exist)")
            continue
        counts[source] = build_shards_for_source(
            src_dir,
            T=T,
            stride=stride,
            examples_per_shard=examples_per_shard,
        )
        print(f"  {source}: {counts[source]} windows -> {src_dir / 'shards'}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("datasets"))
    ap.add_argument("--T", type=int, default=16, help="window length in timesteps")
    ap.add_argument("--stride", type=int, default=8,
                    help="stride between window starts within one episode")
    ap.add_argument("--examples-per-shard", type=int, default=1024)
    args = ap.parse_args()
    build_all(
        args.root, T=args.T, stride=args.stride,
        examples_per_shard=args.examples_per_shard,
    )


if __name__ == "__main__":
    main()
