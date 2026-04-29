"""Manifest spec — `manifest.parquet` lives next to the per-episode HDF5s.

PROJECT_PLAN_1.md §1.2 specifies the columns; the trainer uses this index to
build train/val splits without opening every HDF5.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_FILENAME = "manifest.parquet"

# Column name -> pyarrow type. Order matters for stable Parquet schemas.
MANIFEST_SCHEMA = pa.schema([
    pa.field("episode_id",         pa.string(), nullable=False),
    pa.field("source",             pa.string(), nullable=False),
    pa.field("n_steps",            pa.int32(),  nullable=False),
    pa.field("success",            pa.bool_(),  nullable=False),
    pa.field("box_size_class",     pa.string(), nullable=False),
    pa.field("randomization_seed", pa.int64(),  nullable=False),
    pa.field("recording_date",     pa.string(), nullable=False),
    pa.field("hash",               pa.string(), nullable=False),
    pa.field("path",               pa.string(), nullable=False),  # relative to manifest dir
])


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(rows: list[dict], out_dir: Path) -> Path:
    """Write a manifest.parquet from a list of row dicts. Returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)
    out_path = out_dir / MANIFEST_FILENAME
    pq.write_table(table, out_path)
    return out_path


def read_manifest(manifest_path: Path) -> pa.Table:
    return pq.read_table(manifest_path)
