"""SPMD mesh + partition specs (PROJECT_PLAN_1.md §5.1).

For v1 we run with a **1D data-only mesh**. Tensor parallelism is overkill
at 110M params and adds debugging surface (§5.1). The
nnx.with_partitioning(..., (None, "model")) annotations on the Linears /
Embeds are inert under this mesh; promoting to a 2D ('data', 'model')
mesh is a config change, not a rewrite (§3.2 #1).
"""
from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def make_mesh(*, axis_names: tuple[str, ...] = ("data",)) -> Mesh:
    """Build the SPMD mesh from the local devices.

    Default is a flat ('data',) mesh — every device is one slice of the
    batch. Pass `axis_names=("data", "model")` once we promote to FSDP +
    TP (§5.1)."""
    devices = jax.devices()
    if len(axis_names) == 1:
        return Mesh(np.array(devices), axis_names=axis_names)
    if len(axis_names) == 2:
        return Mesh(np.array(devices).reshape(-1, 1), axis_names=axis_names)
    raise ValueError(f"unsupported axis_names: {axis_names}")


def data_sharding(mesh: Mesh) -> NamedSharding:
    """Shard the leading (batch) axis across the data axis; replicate the rest."""
    return NamedSharding(mesh, P("data"))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Replicate fully — used for scalar metrics and small constants."""
    return NamedSharding(mesh, P())


def shard_batch(batch, sharding: NamedSharding):
    """Apply `data_sharding` to every leaf of a batch dict.

    Falls back to replicated for 0-D leaves (e.g. count scalars) since
    you can't shard a scalar."""
    def _put(x):
        try:
            if hasattr(x, "ndim") and x.ndim >= 1:
                return jax.device_put(x, sharding)
        except Exception:
            pass
        return jax.device_put(x, NamedSharding(sharding.mesh, P()))
    return jax.tree.map(_put, batch)
