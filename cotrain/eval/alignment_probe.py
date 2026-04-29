"""§6.3 alignment probe — the safety net against silent failure.

Per Lei et al. and PROJECT_PLAN_1.md §3.1 / §6.3, the shared bridge wins
at co-training when robot and human bridge-slot representations are
*structured-aligned*: interleaved with locally distinct neighborhoods,
not collapsed onto each other (overlapping regime → negative transfer)
and not orbiting in disjoint clusters (disjoint regime → no transfer).

Three diagnostics, all on the deep-layer hidden states at the BOX,
PHASE, and CONTACT slot positions:

  1. **UMAP** (2D, color by source) — visualization only. Healthy:
     interleaved with local neighborhoods that recognize source.
  2. **Sliced Wasserstein distance** between robot and human distributions
     in the d_model-dim feature space. Moderate and decreasing during
     training is healthy; collapsing toward 0 is the overlapping regime.
  3. **Discriminator probe** — train a 2-layer MLP for `n_steps` steps to
     classify source from a single bridge-token feature. Lei et al.: ~75-95%
     accuracy is healthy ('discernibility preserved'). ~50% is collapse;
     ~100% with poor sim rollouts is disjoint.

The trainer logs these to wandb every `eval_cadence` steps (§5.3). The
plan calls `eval/plots/umap_step_{N}.png` for the figure path; we honor
that.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from cotrain.models.transformer import CoTrainTransformer

# Bridge slot indices into SLOT_ORDER = ("vis","state","box","phase","contact","action").
_BRIDGE_SLOT_IDS = (2, 3, 4)


@dataclass(frozen=True)
class AlignmentMetrics:
    n_robot: int
    n_human: int
    sliced_wasserstein: float
    discriminator_accuracy: float
    umap_2d: np.ndarray | None         # (N_total, 2) or None if compute_umap=False
    source_labels: np.ndarray          # (N_total,) bool — True for robot

    def to_log_dict(self, prefix: str = "alignment") -> dict[str, float]:
        return {
            f"{prefix}/sliced_wasserstein":      float(self.sliced_wasserstein),
            f"{prefix}/discriminator_accuracy":  float(self.discriminator_accuracy),
            f"{prefix}/n_robot":                 int(self.n_robot),
            f"{prefix}/n_human":                 int(self.n_human),
        }


# --- feature extraction --------------------------------------------------

def extract_bridge_features(
    seq: jnp.ndarray,
    *,
    num_slots: int = 6,
) -> jnp.ndarray:
    """Pull the bridge-slot positions (BOX, PHASE, CONTACT) from a (B, 6T, D)
    sequence and flatten across (B, 3T) to give (B*3T, D) — every row is
    one bridge-slot token's hidden state."""
    if seq.ndim != 3:
        raise ValueError(f"expected (B, 6T, D); got {seq.shape}")
    B, L, D = seq.shape
    if L % num_slots:
        raise ValueError(f"L={L} not divisible by num_slots={num_slots}")
    bridges = jnp.stack(
        [seq[:, s::num_slots, :] for s in _BRIDGE_SLOT_IDS], axis=1,
    )                                             # (B, 3, T, D)
    return bridges.reshape(-1, D)                 # (B*3*T, D)


def collect_bridge_features(
    model: CoTrainTransformer,
    batches: Iterable[dict[str, jnp.ndarray]],
    *,
    n_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run `n_batches` of (already-vis-prepared) batches through `model.encode`
    and return (features, source_labels) flattened across (batch, bridge_token)
    so each row is one bridge-slot vector with its source label.

    The caller is responsible for filling in the `vis` field of each batch
    (typically by running the encoder beforehand). Keeping that out of this
    function lets the alignment probe run on cached features without
    re-encoding."""
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for i, batch in enumerate(batches):
        if i >= n_batches:
            break
        seq, _ = model.encode(batch, deterministic=True)
        bf = np.asarray(extract_bridge_features(seq))
        # Each sample in batch contributes 3*T bridge tokens; broadcast its
        # source label across them.
        T = batch["source_mask"].shape[0] and (seq.shape[1] // 6)
        per_sample = 3 * T
        src = np.repeat(np.asarray(batch["source_mask"]).astype(bool), per_sample)
        features.append(bf)
        labels.append(src)
    if not features:
        raise RuntimeError("collect_bridge_features got no batches")
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


# --- metrics --------------------------------------------------------------

def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_projections: int = 64,
    seed: int = 0,
) -> float:
    """W_2 between two empirical distributions in R^D, projected onto random
    1D directions and averaged. Cheap and well-defined for high-dim
    features."""
    import ot
    # POT 0.9 wants an int seed (it builds its own RandomState internally);
    # passing a np.random.Generator hits its "safe cast" guard with a TypeError.
    return float(ot.sliced_wasserstein_distance(
        x, y, n_projections=n_projections, seed=int(seed),
    ))


class _Discriminator(nnx.Module):
    """2-layer MLP, hidden=128 — matches the plan's '2-layer MLP for 200 steps'."""

    def __init__(self, d_in: int, *, rngs: nnx.Rngs, d_hidden: int = 128) -> None:
        self.fc1 = nnx.Linear(d_in, d_hidden, rngs=rngs)
        self.fc2 = nnx.Linear(d_hidden, 2, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.fc2(nnx.gelu(self.fc1(x)))


def discriminator_accuracy(
    features: np.ndarray,
    source_labels: np.ndarray,
    *,
    n_steps: int = 200,
    batch_size: int = 256,
    val_frac: float = 0.2,
    seed: int = 0,
) -> float:
    """Train a 2-layer MLP to classify source from a single feature row;
    return validation accuracy."""
    if features.ndim != 2 or features.shape[0] != source_labels.shape[0]:
        raise ValueError(f"shape mismatch: features {features.shape}, "
                         f"labels {source_labels.shape}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(features))
    n_val = max(int(val_frac * len(features)), 8)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_x = features[train_idx].astype(np.float32)
    train_y = source_labels[train_idx].astype(np.int32)
    val_x = features[val_idx].astype(np.float32)
    val_y = source_labels[val_idx].astype(np.int32)

    disc = _Discriminator(features.shape[-1], rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(disc, optax.adamw(1e-3, weight_decay=1e-4),
                              wrt=nnx.Param)

    @nnx.jit
    def step(disc, opt, x, y):
        def loss_fn(d):
            logits = d(x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        loss, grads = nnx.value_and_grad(loss_fn)(disc)
        opt.update(disc, grads)
        return loss

    n_train = len(train_x)
    for _ in range(n_steps):
        idx = rng.choice(n_train, size=min(batch_size, n_train), replace=True)
        step(disc,
             optimizer,
             jnp.asarray(train_x[idx]),
             jnp.asarray(train_y[idx]))

    val_logits = np.asarray(disc(jnp.asarray(val_x)))
    pred = np.argmax(val_logits, axis=-1)
    return float(np.mean(pred == val_y))


def umap_2d(
    features: np.ndarray,
    *,
    seed: int = 0,
    max_samples: int = 4000,
) -> np.ndarray:
    """UMAP-reduce features to 2D. Subsamples if there are too many points to
    keep the call cheap; the alignment probe is a periodic diagnostic, not a
    hot path."""
    import umap
    if len(features) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(features), size=max_samples, replace=False)
        features = features[idx]
    reducer = umap.UMAP(n_components=2, random_state=seed, n_jobs=1)
    return reducer.fit_transform(features)


# --- top-level orchestration ---------------------------------------------

def compute_metrics(
    features: np.ndarray,
    source_labels: np.ndarray,
    *,
    seed: int = 0,
    discriminator_steps: int = 200,
    compute_umap: bool = True,
) -> AlignmentMetrics:
    src = source_labels.astype(bool)
    robot = features[src]
    human = features[~src]
    if len(robot) == 0 or len(human) == 0:
        raise ValueError(f"need both sources; got n_robot={len(robot)}, "
                         f"n_human={len(human)}")
    sw = sliced_wasserstein(robot, human, seed=seed)
    da = discriminator_accuracy(features, src.astype(np.int32),
                                  n_steps=discriminator_steps, seed=seed)
    coords: np.ndarray | None = None
    if compute_umap:
        coords = umap_2d(features, seed=seed)
    return AlignmentMetrics(
        n_robot=int(src.sum()),
        n_human=int((~src).sum()),
        sliced_wasserstein=sw,
        discriminator_accuracy=da,
        umap_2d=coords,
        source_labels=src,
    )


# --- plotting -------------------------------------------------------------

def plot_umap(metrics: AlignmentMetrics, out_path: str | Path) -> Path:
    """Save a UMAP scatter colored by source. Plan format:
    `eval/plots/umap_step_{N}.png`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metrics.umap_2d is None:
        raise ValueError("metrics.umap_2d is None; recompute with compute_umap=True")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coords = metrics.umap_2d
    src = metrics.source_labels
    # UMAP may have been subsampled; clip the source-label vector to match.
    if len(src) != len(coords):
        src = src[:len(coords)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[~src, 0], coords[~src, 1], c="#1f77b4",
                label=f"human (n={int((~src).sum())})", alpha=0.4, s=8)
    ax.scatter(coords[src, 0], coords[src, 1], c="#d62728",
                label=f"robot (n={int(src.sum())})", alpha=0.4, s=8)
    ax.set_title(
        f"bridge-slot UMAP — sw={metrics.sliced_wasserstein:.3f} "
        f"disc_acc={metrics.discriminator_accuracy:.3f}"
    )
    ax.legend(loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
