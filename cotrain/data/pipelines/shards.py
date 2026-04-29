"""ArrayRecord shard format and serialization (PROJECT_PLAN_1.md §4.1).

Each record holds one `(episode_id, window_start)` window with all per-slot
arrays pre-extracted at fixed T (default 16). Format is plain pickle of a
numpy dict — small enough that the per-record CPU cost is negligible
relative to the disk I/O. The pickle protocol is HIGHEST so it stays cheap
on Python 3.11+ and writes the array buffers as raw bytes.

The format is *internal*: only `serialize_window` and `deserialize_window`
should be calling pickle. Anything else that wants the data should call
`deserialize_window` and then walk the dict.
"""
from __future__ import annotations

import pickle
from typing import Any

# Bumping this whenever the §1.1 / window.py contract changes makes
# stale-shard mistakes obvious — the deserializer asserts on it.
SHARD_FORMAT_VERSION = 1

_HEADER_KEY = "__fmt_version"


def serialize_window(window: dict[str, Any]) -> bytes:
    payload = dict(window)
    payload[_HEADER_KEY] = SHARD_FORMAT_VERSION
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_window(record: bytes) -> dict[str, Any]:
    obj = pickle.loads(record)
    if not isinstance(obj, dict):
        raise ValueError(f"shard record is {type(obj).__name__}, not dict")
    version = obj.pop(_HEADER_KEY, None)
    if version != SHARD_FORMAT_VERSION:
        raise ValueError(
            f"shard format version mismatch: got {version!r}, expected "
            f"{SHARD_FORMAT_VERSION}. Rebuild with build_grain_shards.py."
        )
    return obj
