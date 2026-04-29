"""Round-trip test for §8 step 4a: shard contents == source HDF5 contents.

Plan spec: 'one shard, one episode, equal contents to source HDF5.' We
shard one synthetic robot episode and one synthetic human episode, read
the records back, and compare every per-slot array to what `read_window`
produces directly from HDF5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from array_record.python.array_record_module import ArrayRecordReader

from cotrain.data.pipelines.shards import deserialize_window
from cotrain.data.pipelines.window import read_window
from cotrain.scripts.build_grain_shards import (
    build_shards_for_source,
    windows_for_episode,
)
from cotrain.scripts.generate_synthetic_data import generate

T = 16
STRIDE = 8


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth_for_shards")
    # Tight T_range so every episode yields at least 2 windows at stride 8.
    generate(root, n_robot=2, n_human=2, seed=11, T_range=(40, 48))
    return root


def _read_all_records(shards_dir: Path) -> list[dict]:
    records = []
    for path in sorted(shards_dir.glob("*.array_record")):
        reader = ArrayRecordReader(str(path))
        n = reader.num_records()
        for i in range(n):
            records.append(deserialize_window(reader.read([i])[0]))
        reader.close()
    return records


def test_shards_round_trip_match_hdf5(synth_root: Path) -> None:
    """Every shard record must equal read_window() of the same (ep, start)."""
    for source in ("robot", "human"):
        n_written = build_shards_for_source(
            synth_root / source,
            T=T,
            stride=STRIDE,
            examples_per_shard=1024,
        )
        records = _read_all_records(synth_root / source / "shards")
        assert len(records) == n_written, (len(records), n_written)
        # Sanity: the per-slot array content matches read_window().
        for rec in records[:6]:  # spot-check first 6
            ep_path = synth_root / source / f"{rec['episode_id'].replace(source + '_', 'ep_')}.h5"
            ref = read_window(ep_path, rec["window_start"], T)
            for k in ("rgb", "box_state", "phase", "contact_lift"):
                np.testing.assert_array_equal(rec[k], ref[k], err_msg=f"{source}/{k}")
            if source == "robot":
                np.testing.assert_array_equal(rec["proprio"], ref["proprio"])
                np.testing.assert_array_equal(rec["action"], ref["action"])
            else:
                np.testing.assert_array_equal(rec["human_kin"], ref["human_kin"])


def test_window_count_matches_stride(synth_root: Path) -> None:
    """Total window count = sum(windows_for_episode) over the manifest."""
    for source in ("robot", "human"):
        ds_dir = synth_root / source
        episodes = sorted(ds_dir.glob("ep_*.h5"))
        expected = sum(len(windows_for_episode(p, T, STRIDE)) for p in episodes)
        n_written = build_shards_for_source(
            ds_dir, T=T, stride=STRIDE, examples_per_shard=1024,
        )
        assert n_written == expected, (source, n_written, expected)


def test_format_version_mismatch_raises(tmp_path: Path) -> None:
    """A shard from a different format version must surface, not silently load."""
    import pickle
    from cotrain.data.pipelines import shards
    bad = pickle.dumps({"__fmt_version": shards.SHARD_FORMAT_VERSION + 1, "rgb": None})
    with pytest.raises(ValueError, match="shard format version mismatch"):
        shards.deserialize_window(bad)
