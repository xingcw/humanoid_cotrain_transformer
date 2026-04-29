"""Mixed-batch sampler (PROJECT_PLAN_1.md §4 / §8 step 8).

Per §4.2, every batch has a fixed `n_robot = round(w * batch_size)` robot
samples and `batch_size - n_robot` human samples. The mixing ratio `w`
itself is computed from the dataset sizes per Lei et al.'s guideline
(§4.2 again):

  w_n = N / (N + M)                            # natural / lower bound
  w_q = sqrt(N / M)        if M/N > 5
       N*q / ((1-q)*M + N*q)  otherwise        # q=0.8, optionally capped at 0.5

Helpers `lower_bound_w` and `upper_bound_w` expose those formulas; the
trainer picks the working `w` and sweeps within `[w_n, w_q]`.

The sampler reads our own ArrayRecord shards (built by §8.4a) and
deserializes each record via `cotrain.data.pipelines.shards`. It returns
batched numpy dicts whose keys match the model's input contract.

Determinism: every iteration uses a numpy.Generator seeded from the
constructor seed, so `make_mixed_loader(seed=k)` produces the same
sequence of batches across runs.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import grain.python as grain
import numpy as np

from cotrain.data.pipelines.shards import deserialize_window


# --- mixing-ratio bounds (§4.2) ------------------------------------------

def lower_bound_w(N_robot: int, M_human: int) -> float:
    """w_n = N / (N + M). The natural mixing ratio."""
    if N_robot + M_human == 0:
        raise ValueError("both datasets are empty")
    return N_robot / (N_robot + M_human)


def upper_bound_w(N_robot: int, M_human: int, *, q: float = 0.8,
                  cap_at_half: bool = True) -> float:
    """The plan's `w_q`. Pulls toward 0.5 when the human dataset dwarfs robot."""
    if N_robot <= 0 or M_human <= 0:
        raise ValueError("both datasets must be non-empty")
    if M_human / N_robot > 5:
        w = math.sqrt(N_robot / M_human)
    else:
        w = N_robot * q / ((1 - q) * M_human + N_robot * q)
    return min(w, 0.5) if cap_at_half else w


# --- sampler --------------------------------------------------------------

def _resolve_glob(pattern: str | Path) -> list[str]:
    """Expand a glob (or single path) into a sorted list of shard files.

    `grain.ArrayRecordDataSource` takes a list of file paths — it does NOT
    expand globs internally, despite what some examples suggest. Passing
    a literal '*.array_record' string ends up treating the asterisk as a
    filename and crashes at open time."""
    p = Path(pattern)
    if any(ch in str(p) for ch in "*?["):
        # It's a glob.
        parent = p.parent
        out = sorted(parent.glob(p.name))
    elif p.is_dir():
        out = sorted(p.glob("*.array_record"))
    else:
        out = [p]
    if not out:
        raise FileNotFoundError(f"no shard files match {pattern!r}")
    return [str(x) for x in out]


@dataclass
class MixedBatchSpec:
    """How a single batch is split between sources."""
    batch_size: int
    n_robot: int
    n_human: int

    @classmethod
    def from_w(cls, w: float, batch_size: int) -> "MixedBatchSpec":
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"w must be in [0, 1], got {w}")
        n_robot = round(w * batch_size)
        n_human = batch_size - n_robot
        return cls(batch_size=batch_size, n_robot=n_robot, n_human=n_human)


class MixedShardSampler:
    """Deterministic mixed-batch iterator over robot + human ArrayRecord shards.

    Each batch contains `spec.n_robot` robot samples followed by
    `spec.n_human` human samples; the order *within* the batch is shuffled
    by the same seed so the model sees a heterogeneous source layout
    rather than a clean robot-then-human split.
    """

    def __init__(
        self,
        robot_glob: str | Path,
        human_glob: str | Path,
        spec: MixedBatchSpec,
        *,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.robot_source = grain.ArrayRecordDataSource(_resolve_glob(robot_glob))
        self.human_source = grain.ArrayRecordDataSource(_resolve_glob(human_glob))
        if len(self.robot_source) == 0 or len(self.human_source) == 0:
            raise ValueError("both shard sets must contain at least one record")
        self._rng = np.random.default_rng(seed)
        self._robot_perm: np.ndarray | None = None
        self._human_perm: np.ndarray | None = None
        self._robot_cursor = 0
        self._human_cursor = 0

    def _next_indices(self, source: str, n: int) -> np.ndarray:
        if source == "robot":
            full = len(self.robot_source)
            cursor = self._robot_cursor
            perm = self._robot_perm
        else:
            full = len(self.human_source)
            cursor = self._human_cursor
            perm = self._human_perm

        out = np.empty(n, dtype=np.int64)
        filled = 0
        while filled < n:
            if perm is None or cursor >= full:
                perm = self._rng.permutation(full)
                cursor = 0
            take = min(n - filled, full - cursor)
            out[filled:filled + take] = perm[cursor:cursor + take]
            cursor += take
            filled += take

        if source == "robot":
            self._robot_perm = perm
            self._robot_cursor = cursor
        else:
            self._human_perm = perm
            self._human_cursor = cursor
        return out

    def _read(self, source: str, indices: np.ndarray) -> list[dict]:
        src = self.robot_source if source == "robot" else self.human_source
        out = []
        for idx in indices:
            out.append(deserialize_window(src[int(idx)]))
        return out

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        r_idx = self._next_indices("robot", self.spec.n_robot)
        h_idx = self._next_indices("human", self.spec.n_human)
        windows = self._read("robot", r_idx) + self._read("human", h_idx)
        # In-batch shuffle so robot/human aren't always adjacent.
        order = self._rng.permutation(self.spec.batch_size)
        windows = [windows[i] for i in order]
        return collate_windows(windows)


# --- collation ------------------------------------------------------------

_STACKED_KEYS = (
    "rgb", "box_state", "phase", "contact_lift",
    "proprio", "human_kin", "action",
)


def collate_windows(windows: list[dict]) -> dict[str, np.ndarray]:
    """Stack a list of per-window dicts into one batch dict.

    Keys produced match what the model expects. We rename
    `source_is_robot` -> `source_mask` to align with the masking module
    and the projection heads' parameter name, and expose the per-sample
    provenance (episode_id, window_start) for debugging."""
    batch: dict[str, np.ndarray] = {
        k: np.stack([w[k] for w in windows], axis=0) for k in _STACKED_KEYS
    }
    batch["source_mask"] = np.array(
        [bool(w["source_is_robot"]) for w in windows], dtype=bool,
    )
    # The projection heads expect state_robot / state_human / box / contact
    # keys; map the schema names (proprio, human_kin, box_state,
    # contact_lift) here so the trainer can hand the dict straight to
    # ProjectionHeads.__call__ with no extra renaming.
    batch["state_robot"] = batch.pop("proprio")
    batch["state_human"] = batch.pop("human_kin")
    batch["contact"] = batch.pop("contact_lift")
    batch["box"] = batch.pop("box_state")
    batch["episode_ids"] = np.array([w["episode_id"] for w in windows], dtype=object)
    batch["window_starts"] = np.array([w["window_start"] for w in windows], dtype=np.int32)
    return batch


def make_mixed_loader(
    root: str | Path,
    *,
    w: float,
    batch_size: int,
    seed: int = 0,
) -> MixedShardSampler:
    """Convenience constructor for the standard `<root>/{robot,human}/shards/`
    layout produced by `cotrain.scripts.build_grain_shards`."""
    root = Path(root)
    spec = MixedBatchSpec.from_w(w, batch_size)
    return MixedShardSampler(
        robot_glob=str(root / "robot" / "shards" / "*.array_record"),
        human_glob=str(root / "human" / "shards" / "*.array_record"),
        spec=spec,
        seed=seed,
    )
