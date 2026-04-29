"""Orbax v1 checkpointing for the trainer (PROJECT_PLAN_1.md §5.4).

The plan calls out that the legacy `if process_index == 0: save(...)`
pattern hangs on >8 devices in multi-host setups (§5.4). Orbax handles
host coordination automatically — even on a single-host TPU we use the
same code path so promoting to multi-host is a config change.

We checkpoint:
  - the trainable model state (transformer Params + non-Param state),
  - the optimizer state,
  - a small step counter,
  - any extra user metadata (e.g. wandb run ID, the active w sweep point).

We *don't* checkpoint the encoder — it's frozen, immutable, and large
(86M params). Re-loading it from jimmy's hosted weights at restart is
fast and avoids 300+ MB of extra disk per checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp
from flax import nnx


def _ensure_dir(p: Path) -> Path:
    p = Path(p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(
    path: str | Path,
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save model + optimizer state under `path/{step:08d}/`.

    Returns the directory written. Atomic via Orbax's tmp-rename pattern."""
    root = _ensure_dir(Path(path))
    target_dir = root / f"{step:08d}"
    if target_dir.exists():
        # Orbax refuses to overwrite; remove the empty placeholder if any.
        if any(target_dir.iterdir()):
            raise FileExistsError(f"checkpoint already exists: {target_dir}")
        target_dir.rmdir()

    state = {
        "model_state":     nnx.state(model),
        "optimizer_state": nnx.state(optimizer),
        "step":            step,
    }
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(target_dir, state, force=False)

    if metadata is not None:
        (target_dir / "metadata.json").write_text(json.dumps(metadata, default=str))
    return target_dir


def load_checkpoint(
    path: str | Path,
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
) -> tuple[int, dict[str, Any] | None]:
    """Restore model + optimizer state in place from the checkpoint at `path`.

    `path` should be the *step* directory (e.g. `runs/foo/00001000/`).
    Returns (step, metadata)."""
    target_dir = Path(path).resolve()
    if not target_dir.is_dir():
        raise FileNotFoundError(target_dir)

    template = {
        "model_state":     nnx.state(model),
        "optimizer_state": nnx.state(optimizer),
        "step":            0,
    }
    ckptr = ocp.PyTreeCheckpointer()
    restored = ckptr.restore(target_dir, item=template)

    nnx.update(model, restored["model_state"])
    nnx.update(optimizer, restored["optimizer_state"])
    step = int(restored["step"])

    meta_path = target_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return step, metadata


def latest_checkpoint(path: str | Path) -> Path | None:
    """Find the highest-step checkpoint under `path/`. Returns None if empty."""
    root = Path(path)
    if not root.is_dir():
        return None
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    return candidates[-1] if candidates else None
