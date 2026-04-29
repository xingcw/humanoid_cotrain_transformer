"""Compatibility shims for third-party libraries that pin older Flax internals.

`jimmy-vision` v0.0.7 (the latest tag, used as our DINOv2 NNX encoder per
PROJECT_PLAN_1.md §2.2.1) imports `from flax.nnx.nnx.module import first_from`,
which was an internal path that disappeared when Flax flattened
`flax.nnx.nnx.*` to `flax.nnx.*` (somewhere between 0.8.x and 0.10.x).

We alias the old path back to the new one in `sys.modules` *before* jimmy's
own modules try to import. The alias is a no-op if jimmy ever updates
upstream; we will drop it then.
"""
from __future__ import annotations

import sys


def install_jimmy_flax_shim() -> None:
    """Idempotent. Safe to call multiple times."""
    if "flax.nnx.nnx" in sys.modules:
        return
    import flax.nnx as _nnx
    import flax.nnx.module as _nnx_module
    sys.modules["flax.nnx.nnx"] = _nnx
    sys.modules["flax.nnx.nnx.module"] = _nnx_module


# Apply at import time so anything pulling in `cotrain` is protected.
install_jimmy_flax_shim()
