# Project Plan: Shared-Bridge Multimodal Transformer for Human–Humanoid Co-Training (JAX/TPU)

**Audience:** coding agents implementing the policy described in the proposal "Vision-based human and humanoid co-training."
**Scope:** everything from raw data on disk → trained checkpoint → deployment-ready policy. Stages 0 (rollout generator) and 1 (human data collection) are upstream of this plan and assumed to produce the file layouts described in §1.
**Implementation target:** JAX + Flax NNX, training on TPU. Stack details in §0.

**Design anchors (read these first if uncertain):**
- Modality-aligned next-token prediction over interleaved sensorimotor tokens, with masked modalities filled in by the dataset that has them. (Radosavovic et al., *Humanoid Locomotion as Next Token Prediction*, arXiv:2402.19469.)
- Co-training works through **structured representation alignment**, not mixing ratio. Aim for the structured-aligned regime; verify with a UMAP probe. (Lei et al., *A Mechanistic Analysis of Sim-and-Real Co-Training*, arXiv:2604.13645.)
- Mixing ratio guideline: lower bound `w_n = N/(N+M)`, where N = robot rollouts, M = human demos. Sweep upward from there. Never blindly use 50/50.

**Repo layout the agent should create:**
```
cotrain/
  configs/                  # YAML, one per experiment
  data/
    pipelines/              # raw → tokenized → grain shards
    schemas/                # Pydantic models for each modality
  models/
    encoders/               # jimmy DINOv2 loader (frozen, in-graph)
    transformer/            # the shared backbone (Flax NNX)
    heads/                  # one prediction head per modality
  training/
    losses.py
    masking.py              # the heart of the bridge — see §3.4
    sampler.py              # mixed-batch logic, grain-based
    trainer.py              # nnx.jit train_step + Orbax checkpointing
    sharding.py              # mesh + partition specs
  eval/
    rollout.py              # closed-loop sim eval
    alignment_probe.py      # UMAP + Wasserstein, per Lei et al.
    parity_dino.py          # PyTorch DINOv2 vs jimmy NNX parity check (§2.2.1)
  scripts/
    preprocess_robot.py
    preprocess_human.py
    build_grain_shards.py    # HDF5 → ArrayRecord shards
    train.py
    eval.py
tests/                      # pytest, every module has at least smoke tests
```

---

## §0. Stack and library choices

| Concern              | Choice                                                       | Notes                                                                      |
|----------------------|--------------------------------------------------------------|----------------------------------------------------------------------------|
| Neural net library   | **Flax NNX** (not Linen)                                     | NNX is the recommended API for new projects. Linen still works but is not where new effort goes. |
| Optimizer            | **Optax** + `nnx.Optimizer`                                  | `optax.adamw` + `optax.warmup_cosine_decay_schedule` + `optax.clip_by_global_norm`. |
| Checkpointing        | **Orbax** (`orbax-checkpoint` v1 API)                        | Mandatory in multi-host. The legacy `if process_index == 0: save(...)` pattern hangs on >8 devices — do not use it. |
| Input pipeline       | **`grain`** (preferred) or `tf.data`                         | `grain` is JAX-native and host-agnostic; `tf.data` has the wider tutorial base. Pick one and stick to it. |
| Distributed config   | **SPMD via `jax.sharding`** + `nnx.with_partitioning`        | No `pmap`. No `xmap`. |
| Profiling            | TensorBoard profiler (`jax.profiler.start_trace`) + XProf    | Capture step traces every 5k steps during the first run to spot stragglers. |
| Container/runtime    | TPU VM (v4 / v5p / v6) with the matching `libtpu` wheel      | Pin `jax`, `jaxlib`, and `libtpu` versions together — mismatch is the #1 source of "everything compiles, nothing runs" failures. |

Avoid: `pmap` (legacy), `flax.linen.partitioning` (old API), any PyTorch-specific glue. Optax is fine to use straight.

---

## §1. Data: the contract between Stages 0/1 and this plan

The agent **does not collect data** — Stage 0 (mimic/RL rollout generator) and Stage 1 (Aria capture) produce it. This plan reads it.

### 1.1 On-disk format

Both datasets land as **per-episode HDF5 files**, one episode per file, plus a top-level `manifest.parquet` listing every episode with metadata.

```
datasets/
  robot/
    manifest.parquet
    ep_000001.h5
    ep_000002.h5
    ...
  human/
    manifest.parquet
    ep_000001.h5
    ...
```

**A second pass converts these to `grain` ArrayRecord shards** (see §4.1 for why). HDF5 is the authoritative source; the shards are a derived artifact rebuilt from `scripts/build_grain_shards.py` whenever the schema changes.

Each `ep_XXXXXX.h5` has the following groups, **all aligned on a common timeline at 30 Hz** (resample upstream if native rates differ):

| Group key            | Shape per timestep        | Robot? | Human? | Notes |
|----------------------|---------------------------|--------|--------|-------|
| `rgb`                | `(H, W, 3)` uint8         | ✓      | ✓      | 224×224 after preprocess. Robot RGB used for action grounding only — masked at training time (see §3.4). |
| `proprio`            | `(D_p,)` float32          | ✓      | —      | Concat of `q, qdot, root_quat, root_lin_vel, root_ang_vel`. Fixed `D_p` per embodiment; record in manifest. |
| `human_kin`          | `(D_h,)` float32          | —      | ✓      | Concat of head pose (SLAM), wrist 6-DoF poses, hand keypoints (21 per hand × 3), upper-body joint angles. Fixed `D_h`. |
| `box_state`          | `(7,)` float32            | ✓      | ✓      | `[x, y, z, qw, qx, qy, qz]` of box relative to **camera frame**. **The shared bridge.** Same definition for both datasets — this is non-negotiable. |
| `action`             | `(D_a,)` float32          | ✓      | —      | Robot joint position targets. |
| `phase`              | `(1,)` int8               | ✓      | ✓      | Enum: 0=approach, 1=reach, 2=contact, 3=lift, 4=hold. **Shared bridge.** |
| `contact_lift`       | `(3,)` float32            | ✓      | ✓      | `[left_contact, right_contact, lifted]` ∈ {0,1}. **Shared bridge.** |
| `meta/success`       | scalar bool               | ✓      | ✓      | Per episode, not per timestep. |
| `meta/episode_id`    | scalar str                | ✓      | ✓      | |
| `meta/source`        | scalar str                | ✓      | ✓      | `"robot"` or `"human"`. |

**Critical:** `box_state`, `phase`, and `contact_lift` are computed identically in both pipelines. The whole project depends on this — if a robot-side `box_state` uses the world frame and the human-side uses the camera frame, the bridge collapses and co-training fails per Lei et al.'s "disjoint" regime.

### 1.2 Manifest

`manifest.parquet` columns: `episode_id, source, n_steps, success, box_size_class, randomization_seed, recording_date, hash`. The trainer uses this to build splits without opening every HDF5.

### 1.3 Validation script (agent must implement)

`scripts/validate_dataset.py` runs **before any training** and asserts:
- All `box_state` values across both datasets fall in the same numeric range (sanity check on frame consistency).
- Phase enum values are a subset of `{0,1,2,3,4}`.
- For each episode, the per-timestep arrays have matching length.
- Manifest hashes match the actual files.

If this fails, the agent stops and reports — does not silently continue.

---

## §2. Tokenization: turning aligned timesteps into transformer-ready sequences

This is where most of the implementation work lives, and where most bugs will originate. Read §2 carefully.

### 2.1 Token slot definition

Per timestep `t`, the model consumes a fixed-order sequence of **6 token slots**:

```
[VIS_t] [STATE_t] [BOX_t] [PHASE_t] [CONTACT_t] [ACTION_t]
```

| Slot index | Slot name   | Robot fills with         | Human fills with          | Always shared? |
|------------|-------------|--------------------------|---------------------------|----------------|
| 0          | `VIS`       | `[VIS_MASK]` (see §3.4)  | DINO(rgb)                 | No             |
| 1          | `STATE`     | proprio                  | human_kin                 | No (different feature spaces) |
| 2          | `BOX`       | box_state                | box_state                 | **Yes — bridge** |
| 3          | `PHASE`     | phase                    | phase                     | **Yes — bridge** |
| 4          | `CONTACT`   | contact_lift             | contact_lift              | **Yes — bridge** |
| 5          | `ACTION`    | action                   | `[MASK_ACTION]`           | No             |

The **full input sequence** for a window of length `T` timesteps is the concatenation:
`[VIS_0][STATE_0][BOX_0][PHASE_0][CONTACT_0][ACTION_0][VIS_1]...[ACTION_{T-1}]`
→ total length `6T` tokens.

Default `T = 16` timesteps (~530 ms at 30 Hz). Configurable.

### 2.2 Per-modality projection heads

Each slot has its own input projection into the transformer's `d_model` (default 768). Heads are tiny — these are **not** the place to add capacity.

| Slot       | Encoder                                                              | Output dim |
|------------|----------------------------------------------------------------------|------------|
| `VIS`      | Frozen DINO-v2 ViT-B/14 → mean-pool patch tokens → Linear(768→d_model) | d_model    |
| `STATE` (robot)  | LayerNorm(D_p) → MLP(D_p → 256 → d_model)                        | d_model    |
| `STATE` (human)  | LayerNorm(D_h) → MLP(D_h → 256 → d_model)                        | d_model    |
| `BOX`      | LayerNorm(7) → MLP(7 → 64 → d_model)                                 | d_model    |
| `PHASE`    | `nnx.Embed(5, d_model)`                                              | d_model    |
| `CONTACT`  | `nnx.Linear(3, d_model)`                                             | d_model    |
| `ACTION`   | LayerNorm(D_a) → MLP(D_a → 256 → d_model)                            | d_model    |

**Two STATE heads exist** because robot proprio and human kinematics live in different feature spaces. Both project into the same slot 1 of the same `d_model` space — the transformer sees them as the same token type, which is what allows cross-dataset attention to work. This mirrors the modality-aligned scheme of Radosavovic et al.

**Slot embeddings:** add a learned `nnx.Embed(6, d_model)` (the slot ID) to every token. **Time embeddings:** add a learned `nnx.Embed(T, d_model)` (the timestep within the window). RoPE optional and probably overkill at T=16 — skip it for v1.

**Mask tokens:** `[VIS_MASK]` and `[MASK_ACTION]` are two **separate learned vectors**, declared as `nnx.Param(jnp.zeros((d_model,)))` so they appear in the state tree and get optimizer updates. They are not the same as the slot embedding — they replace the projected modality content entirely.

#### 2.2.1 Visual encoder: `jimmy` DINOv2 in Flax NNX

The visual encoder is **DINOv2 ViT-B/14, loaded via the `jimmy` library** (`clementpoiret/jimmy`). It runs frozen, in-graph, on TPU — no precompute step, no host-side PyTorch dependency at train time.

```python
from jimmy.models import DINOV2_VITB14, load_model   # see note below if jimmy ships only ViT-S
from flax import nnx

rngs = nnx.Rngs(42)
vis_encoder = load_model(
    DINOV2_VITB14, rngs=rngs, pretrained=True,
    url="https://huggingface.co/poiretclement/dinov2_jax/resolve/main/dinov2_vitb14.jim",
)
# Forward pass during training:
#   features = vis_encoder(rgb_518)        # (B, 1370, 768) at 518×518
#   pooled   = features.mean(axis=1)       # (B, 768) -- mean-pool patch tokens
# Then pooled feeds the VIS projection head from §2.2.
```

**ViT size note:** the `jimmy` README explicitly demonstrates `DINOV2_VITS14` (ViT-S/14, 384-dim). ViT-B/14 (768-dim, what the rest of this plan assumes) is the standard size and the hosted-weights URL pattern follows from the S example, but the agent should verify the exact constant name and weight URL by inspecting `jimmy/models/__init__.py` before training. If only ViT-S/14 is available, drop `768 → 384` in the VIS projection head input dim (§2.2) and proceed; the rest of the plan is unaffected.

**Frozen forward, in `nnx.jit`:** call the encoder functionally inside the training step with `nnx.split` to separate the frozen `vis_encoder` state from the rest of the model — this prevents its parameters from showing up in the optimizer's gradient tree. Mean-pool the patch tokens (drop the `[CLS]` token) before the VIS projection head. The output of the projection head goes into slot 0 of the token sequence, exactly as §2.2 specifies.

**Mandatory parity check before any real training run.** `eval/parity_dino.py` runs 32 identical images through both the official PyTorch DINOv2 ViT-B/14 (loaded via `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`) and the `jimmy` JAX model, computes max-abs-error on the patch-token outputs, and asserts it's below 1e-3. This must pass before step 13 of §8 (the scaled training run). If parity fails, the fallback is to precompute features offline with PyTorch DINOv2 and load them at train time as a new HDF5 group `dino_features` — that path is straightforward enough to implement in a day if needed, but the agent should not pre-build it.

**Considered and rejected (for the record, so the agent doesn't relitigate the choice):**
- HuggingFace `FlaxDinov2Model` — `transformers` v5 (December 2025) sunset Flax support; the class was already pinned to `jax<=0.4.13` (#37262, closed as not planned) and has a known parity bug (#37246).
- `kylestach/dinov2-jax` — Flax **Linen**, not NNX; would require `nnx.bridge.ToNNX` wrapping at every step. One commit, self-disclaimed correctness, ViT-S only.
- DINOv3 (Meta, August 2025) — PyTorch-only release; would force the precompute path. Worth revisiting in v2 if the dense-feature improvements matter for this task.
- Train a ViT from scratch — defeats the point of using a frozen prior.

### 2.3 Output (prediction) heads

One per modality, mirroring the input heads:

| Target slot | Head architecture                       | Loss                                  |
|-------------|------------------------------------------|---------------------------------------|
| `VIS`       | (skipped — see §3.5)                     | —                                     |
| `STATE`     | MLP(d_model → 256 → D_p or D_h)          | MSE on next-step state                |
| `BOX`       | MLP(d_model → 64 → 7)                    | MSE on translation, geodesic on quat  |
| `PHASE`     | Linear(d_model → 5)                      | Cross-entropy                         |
| `CONTACT`   | Linear(d_model → 3) + sigmoid            | BCE                                   |
| `ACTION`    | MLP(d_model → 256 → D_a)                 | L1 (Huber as alternative)             |

Each output head reads from the corresponding slot's hidden state — i.e., the head for `BOX` only sees positions in the sequence where slot index = 2.

### 2.4 Causal masking and prediction targets

Use a standard **causal attention mask**: token at position `i` attends to all positions `≤ i`. Within a single timestep, the slot order matters: `VIS` is consumed before `STATE`, `STATE` before `BOX`, etc., so `ACTION_t` can attend to `VIS_t, STATE_t, BOX_t, PHASE_t, CONTACT_t` from the same step — this is what gives the action prediction full context of the current observation.

**Prediction targets** follow the modality-aligned rule: the output at position `i` predicts the **next token of the same modality** (i.e., next timestep's same-slot value). Concretely:
- The output at `ACTION_t` predicts `ACTION_{t+1}`.
- The output at `BOX_t` predicts `BOX_{t+1}`.
- Etc.

The very last step's predictions have no targets and are dropped.

---

## §3. The shared bridge: how the architecture forces alignment

This is the conceptual heart of the project. §1 and §2 are plumbing; §3 is the design.

### 3.1 What "shared bridge" means concretely

`BOX`, `PHASE`, and `CONTACT` are filled with **identical-format values** from both datasets. In every mixed batch, the transformer sees these tokens drawn from both distributions. Because:
- The same projection head processes both.
- The same prediction head produces targets for both.
- The same attention pattern routes information to/from them.

…gradients flowing through these slots **necessarily** entangle robot-side and human-side representations. Per Lei et al., this is exactly the structured representation alignment mechanism — and it explains ~50% of co-training loss variance, vs ~20% for mixing ratio. The agent does not need to add an explicit alignment loss (no MMD, no adversarial discriminator) for v1; the shared bridge does the work implicitly.

### 3.2 Transformer backbone — NNX module layout

NNX modules eagerly initialize parameters in `__init__`, so every shape is known at construction — preferable for sharding annotations. Sketch:

```python
from flax import nnx
import jax.numpy as jnp

class ProjectionHeads(nnx.Module):
    """One head per slot. Two STATE heads exist (robot vs human) -- routed by `source`."""
    def __init__(self, d_model: int, D_p: int, D_h: int, D_a: int, *, rngs: nnx.Rngs):
        self.vis = nnx.Linear(768, d_model, rngs=rngs,
                              kernel_init=nnx.with_partitioning(
                                  nnx.initializers.xavier_uniform(), (None, "model")))
        self.state_robot = MLP([D_p, 256, d_model], rngs=rngs)
        self.state_human = MLP([D_h, 256, d_model], rngs=rngs)
        self.box     = MLP([7, 64, d_model], rngs=rngs)
        self.phase   = nnx.Embed(num_embeddings=5, features=d_model, rngs=rngs)
        self.contact = nnx.Linear(3, d_model, rngs=rngs)
        self.action  = MLP([D_a, 256, d_model], rngs=rngs)

        # Learned mask vectors -- nnx.Param so they are tracked in the state tree.
        self.vis_mask    = nnx.Param(jnp.zeros((d_model,)))
        self.action_mask = nnx.Param(jnp.zeros((d_model,)))


class CoTrainTransformer(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.heads = ProjectionHeads(cfg.d_model, cfg.D_p, cfg.D_h, cfg.D_a, rngs=rngs)
        self.slot_emb = nnx.Embed(num_embeddings=6, features=cfg.d_model, rngs=rngs)
        self.time_emb = nnx.Embed(num_embeddings=cfg.T, features=cfg.d_model, rngs=rngs)
        self.blocks = [TransformerBlock(cfg, rngs=rngs) for _ in range(cfg.n_layers)]
        self.norm = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.out_heads = OutputHeads(cfg, rngs=rngs)

    def __call__(self, batch, *, deterministic: bool):
        # batch contains: rgb_or_dino, proprio_or_kin, box, phase, contact, action, source_mask
        # source_mask: (B,) bool, True for robot
        tokens = self._embed(batch)            # (B, 6T, d_model)
        tokens = self._apply_modality_masks(tokens, batch["source_mask"])
        for block in self.blocks:
            tokens = block(tokens, deterministic=deterministic)
        tokens = self.norm(tokens)
        return self.out_heads(tokens)          # dict of per-slot predictions
```

**Backbone:** decoder-only causal transformer, no cross-attention. FFN expansion 4×, GELU, pre-norm, dropout 0.1. Use `jax.nn.dot_product_attention(..., is_causal=True)` if you're on a recent JAX, else build the mask explicitly. Flash-attention-style implementations exist on TPU via `jax.experimental.pallas`; for v1 the standard implementation is fast enough and far less fragile.

**Three things the agent must get right here:**
1. **All parameter initializers carry `nnx.with_partitioning`** annotations. Without these, FSDP doesn't shard the parameters and you OOM on the first compile. With a 1D `("data",)` mesh the annotations are inert — that's fine; leave them in so promoting to 2D is a config change, not a rewrite.
2. **The `_apply_modality_masks` function is JIT-compatible.** It takes `source_mask: (B,)` and uses `jnp.where(source_mask[:, None, None], vis_mask_token, vis_real_tokens)` to swap in the learned mask token. Do **not** use Python-level `if source == "robot"` — that doesn't trace.
3. **Slot identity is preserved by position alone** (see §3.3) — no operation may scramble the slot order.

Total params at the defaults: ~110M backbone + ~5M heads + 86M frozen DINOv2 ViT-B/14 (not in the optimizer's gradient tree, but resident on device). Fits comfortably on a single TPU v5p chip; FSDP across multiple chips is for throughput, not capacity.

### 3.3 Slot identity is preserved across the whole stack

At every layer, slot identity is implicit in the position. The agent must **not** introduce any operation that scrambles slot order (no gather/scatter that mixes positions arbitrarily, no global pooling before the heads). The output head reads from the exact same positional indices the input head wrote to.

Add a unit test (`tests/test_slot_integrity.py`) that asserts: for a batch where all `BOX` tokens are zeroed and everything else is random, the output `BOX` head's gradients are zero on every non-`BOX` input — i.e., information about box state can only flow through box-slot positions on the input side. This is a strict test and might fail due to attention; the relaxed version is to check that the output heads are wired to the correct slots.

### 3.4 Masking — the implementation that must be exactly right

The masking module (`training/masking.py`) is where most subtle bugs will hide. **This is the single function most likely to break when an agent ports from PyTorch** — Python-level branching on `source == "robot"` works in eager PyTorch but silently breaks under `nnx.jit`. Use `jnp.where`. Spec:

```python
import jax.numpy as jnp
from flax import nnx

def apply_modality_masks(
    tokens: dict[str, jnp.ndarray],   # slot_name -> (B, T, d_model), already projected
    source_mask: jnp.ndarray,         # (B,) bool, True for robot, False for human
    vis_mask_token: jnp.ndarray,      # (d_model,) learned, from ProjectionHeads.vis_mask
    action_mask_token: jnp.ndarray,   # (d_model,) learned, from ProjectionHeads.action_mask
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """
    Returns (masked_tokens, loss_masks).
    loss_masks[slot]: (B, T) float in {0, 1}, multiplied into the per-token loss.
    """
    B = source_mask.shape[0]
    is_robot = source_mask[:, None, None]                 # (B, 1, 1)
    vis_replaced = jnp.broadcast_to(vis_mask_token, tokens["vis"].shape)
    act_replaced = jnp.broadcast_to(action_mask_token, tokens["action"].shape)

    masked = dict(tokens)
    masked["vis"]    = jnp.where(is_robot,  vis_replaced, tokens["vis"])
    masked["action"] = jnp.where(is_robot,  tokens["action"], act_replaced)

    T = tokens["vis"].shape[1]
    ones = jnp.ones((B, T))
    loss_masks = {
        "vis":     jnp.zeros((B, T)),                     # always skip (see §3.5)
        "state":   ones,
        "box":     ones,
        "phase":   ones,
        "contact": ones,
        "action":  jnp.where(source_mask[:, None], ones, 0.0),  # robot only
    }
    return masked, loss_masks
```

Rules in plain language:
1. **Robot samples**: replace `VIS` token with the learned `[VIS_MASK]` parameter. Reason: the robot-side RGB comes from sim and is visually dissimilar to the deployment-time real RGB. We do not want the policy to learn shortcuts from sim RGB. Loss contribution from robot `VIS` predictions = 0.
2. **Human samples**: replace `ACTION` token with the learned `[MASK_ACTION]` parameter. Loss contribution from human `ACTION` predictions = 0.
3. All other slots are unmasked. Their loss masks = 1.

**This is the "filling in missing modalities" mechanism.** The transformer learns that when slot 0 is `[VIS_MASK]`, the action prediction must rely on slots 1–4 (state + bridge). When slot 5 is `[MASK_ACTION]`, the box / phase / contact predictions must explain themselves through slots 0–4. Over many mixed batches, the latent representations of bridge tokens get aligned across both domains because both datasets supervise them.

### 3.5 Why VIS prediction is skipped

Predicting next-frame RGB pixels is expensive and unnecessary for our task. We rely on DINO features being stable enough that skipping the visual prediction loss does not hurt action grounding. If later experiments show this is wrong, add a **DINO-feature MSE** prediction head (predict next step's DINO features, not raw pixels) — but defer this to v2.

### 3.6 Loss

```
L_total = λ_action * L_action + λ_state * L_state +
          λ_box * L_box + λ_phase * L_phase + λ_contact * L_contact
```

With per-modality loss masks applied. Default λ:
- `λ_action = 1.0` (only contributes from robot samples)
- `λ_state = 0.2`
- `λ_box = 1.0`   (bridge — pull this hard)
- `λ_phase = 0.5` (bridge)
- `λ_contact = 0.5` (bridge)

**Do not down-weight bridge losses** — they're the alignment mechanism. If anything, up-weight them in early epochs to lock in alignment before actions start dominating.

---

## §4. Mixed-batch sampling and the input pipeline

### 4.1 Pre-tokenize at shard creation time

This is a deliberate choice for TPU. PyTorch dataloaders happily slice windows and pad tensors inside `__getitem__` because that's parallel CPU work; on TPU the host-side pipeline is more I/O-sensitive, and `grain` works much better with flat shards of pre-tokenized examples than with thousands of small HDF5 files.

`scripts/build_grain_shards.py` produces an `ArrayRecord` shard set per dataset. Each example in the shard is one `(window_start, episode_id, source)` triple, with all per-window arrays pre-extracted and pre-padded to the fixed `T = 16`. Rebuild whenever the schema in §1.1 changes.

```
datasets/
  robot/shards/00000.array_record, 00001.array_record, ...
  human/shards/00000.array_record, 00001.array_record, ...
```

### 4.2 Mixed batching with grain

Two `grain` data sources, one per dataset, multiplexed by a deterministic interleave per the `w` config (Lei et al. mixing-ratio guideline).

```python
import grain.python as grain

def make_loader(w: float, batch_size: int, seed: int):
    n_robot = round(w * batch_size)
    n_human = batch_size - n_robot

    robot = grain.MapDataset.source(
        grain.ArrayRecordDataSource("datasets/robot/shards/*.array_record"))
    human = grain.MapDataset.source(
        grain.ArrayRecordDataSource("datasets/human/shards/*.array_record"))
    # Each iteration draws n_robot from robot, n_human from human, concatenates,
    # shuffles within the global batch, and emits one (B, ...) batch.
    ...
```

`w` is computed from the dataset sizes per Lei et al.'s guideline:
- Lower bound: `w_n = N / (N + M)` (natural mixing ratio).
- Upper bound: if `M / N > 5` use `w_q = sqrt(N / M)`; else `w_q = N*q / ((1-q)*M + N*q)` with `q = 0.8`. Optionally cap at 0.5.

**Sweep schedule for v1:** sweep `w` across the `[w_n, w_q]` range with three points (e.g. `w_n`, midpoint, `w_q`). Pick the best on validation rollout success rate. Lei et al. report 0.1–0.3 is often optimal when human data is plentiful; 0.5+ when robot rollouts are scarce.

**Determinism note:** `grain` is deterministic by default. Keep it that way — when sweeping `w`, you want each sweep point to see the same windows in the same order modulo the ratio.

### 4.3 What lives on host vs device

| Artifact                          | Host | Device |
|-----------------------------------|------|--------|
| Raw HDF5                          | ✓    |        |
| ArrayRecord shards                | ✓    |        |
| Decoded batch (NumPy)             | ✓    |        |
| Sharded batch (after `device_put`)|      | ✓      |
| Model params, optimizer state     |      | ✓      |
| Frozen `jimmy` DINOv2 encoder     |      | ✓      |

---

## §5. Training loop

`scripts/train.py`. Uses Flax NNX, Optax, and a JIT-compiled `train_step`. Sharding via `jax.sharding`.

### 5.1 Mesh and sharding

For a 110M-param model, **FSDP alone is sufficient**. Tensor parallelism is overkill at this scale and adds debugging surface.

```python
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = jax.devices()                      # all local TPU cores on this VM, or a slice
mesh = Mesh(np.array(devices).reshape(-1, 1), axis_names=("data", "model"))
data_sharding = NamedSharding(mesh, P("data"))
```

Default to a **1D data-only mesh** for v1 (`axis_names=("data",)`). Promote to 2D only if you need it. Parameter `nnx.with_partitioning` annotations from §3.2 are inert under a 1D mesh — that's fine, leave them in.

Each device sees a different slice of the batch (`P("data")`), all-gathers parameters as needed during forward, reduce-scatters gradients on backward. This is FSDP via the standard JAX SPMD model.

### 5.2 The training step

```python
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        preds = model(batch, deterministic=False)
        return compute_loss(preds, batch)            # returns (loss, aux_dict)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, aux
```

`nnx.jit` handles state propagation in-place. Don't return `model` — that's a Linen-ism. For multi-host runs, `grain` handles cross-host coordination automatically; in single-host runs `jax.device_put(batch, data_sharding)` is enough.

### 5.3 Hyperparameter defaults

| Hyperparameter      | Value                                                      |
|---------------------|------------------------------------------------------------|
| `d_model`           | 768                                                        |
| `n_layers`          | 12                                                         |
| `n_heads`           | 12                                                         |
| `T` (window)        | 16                                                         |
| Batch size          | 256 windows / chip × n_chips (e.g. v5p-8 → 2048 effective) |
| LR                  | 3e-4 × √(batch_ratio); cosine decay; 2k linear warmup      |
| Optimizer           | `optax.adamw` β=(0.9, 0.95), weight_decay=0.1              |
| Grad clip           | `optax.clip_by_global_norm(1.0)`                           |
| Precision           | **bf16 native** (TPUs are native bf16; no GradScaler)      |
| Steps               | 200k at single-chip batch; reduce inversely if batch grew  |
| Eval cadence        | every 5k steps: alignment probe + offline action MSE       |
| Rollout cadence     | every 25k steps: 50 sim rollouts, log success rate         |
| Compile cache       | `jax.experimental.compilation_cache.set_cache_dir(...)`    |

**First-compile time:** expect 5–10 min on first JIT. The compile cache makes subsequent runs near-instant. **Eval uses a separate JIT** — different shapes, different caching, don't share the train_step compile.

### 5.4 Checkpointing

Use **Orbax** (`orbax-checkpoint` v1 API). Mandatory in multi-host. The legacy `if process_index == 0: save(...)` pattern hangs on >8 devices — Orbax handles host coordination automatically.

Checkpoint every 5k steps; keep last 3 + best-by-rollout-success.

### 5.5 JAX gotchas the agent should know up front

These are pitfalls, not blockers — but if the agent isn't expecting them, debugging will be slow.

1. **No in-place mutation inside JIT.** The masking module from §3.4 must use `jnp.where`, not `tokens[:, vis_positions] = mask`.
2. **Random keys are explicit.** Dropout, masking, and any sampling all need PRNG keys threaded through. NNX hides some of this with `nnx.Rngs`, but the trainer still needs to split keys per step.
3. **Shape polymorphism is limited.** Variable sequence length (e.g., `T` differing across batches) recompiles every time — pad to a fixed `T = 16` always, mask out padding in the loss.
4. **`jax.debug.print` works inside JIT;** `print` does not. Use the former for debugging; remove before merging.
5. **`nan_to_num` on gradients is NOT free.** If the agent finds itself reaching for it, the underlying issue (probably a mask producing zero-norm rows in a softmax) needs fixing instead.
6. **Multi-host coordination.** On a TPU pod with multiple hosts, `jax.process_index()` matters for logging. Logs need explicit `if jax.process_index() == 0:` gating; checkpointing does not (Orbax handles it).

---

## §6. Evaluation

Three things, in priority order:

### 6.1 Sim rollout success (the only metric that ultimately matters)

`eval/rollout.py` loads the policy, runs `N=50` randomized box-pickup episodes in the same Stage-0 simulator, reports success rate (lifted-and-held-for-1s). This is the headline number.

### 6.2 Offline prediction quality

Per-modality validation losses on held-out episodes from both datasets. Useful for catching regressions; don't over-index on these.

### 6.3 Alignment probe (the diagnostic that prevents silent failure)

`eval/alignment_probe.py` is the agent's safety net against the **overlapping** and **disjoint** regimes from Lei et al.

For a sample of validation windows from both sources, extract the **deep-layer hidden states** at the bridge slots (`BOX`, `PHASE`, `CONTACT` positions) and:
1. **UMAP** to 2D, color by source (robot vs human). Save as `eval/plots/umap_step_{N}.png`.
2. **Wasserstein distance** between the robot and human distributions in feature space. Log to TensorBoard.
3. **Discriminator probe**: train a 2-layer MLP for 200 steps to classify source from the hidden state. Report accuracy. Lei et al.: ~100% accuracy is *good* (discernibility preserved); ~50% accuracy is *bad* (collapsed/overlapping representations → negative transfer).

Healthy training: UMAP shows robot and human points **interleaved with locally distinct neighborhoods** (structured aligned regime), Wasserstein moderate and decreasing, discriminator accuracy plateaus around 75–95%. If discriminator drops to ~50% AND rollout success drops, you're in the overlapping regime → reduce mixing ratio toward `w_n` or add the CFG-style domain conditioning hinted at in Lei et al. §"A Unified View."

---

## §7. Deployment (Stage 4 from the proposal)

`eval/deploy.py` runs the trained checkpoint against real robot observations.

At inference, the model receives:
- Real `rgb` from the robot's egocentric camera → DINO → `VIS` slot.
- Real `proprio` → robot STATE head → `STATE` slot.
- `BOX`, `PHASE`, `CONTACT`: **estimated** at runtime, not ground truth. Two options for v1:
  - **Option A (recommended):** the model predicts these as part of its output; feed predictions back as the next-step input (autoregressive, the standard NTP setup).
  - **Option B:** wire in lightweight runtime estimators (a small box detector + a hand-coded phase classifier) that produce these tokens from observation. Slower to build; safer in early experiments.
- `ACTION` slot is `[MASK_ACTION]` (we don't know the action; we're predicting it).

The model produces an `ACTION` prediction for the next step → robot executes → loop. Run at the same 30 Hz the data was collected at; if the model can't keep up, drop `T` to 8.

**Sanity gates before any real-robot run:**
1. Rollout success in sim ≥ 80% over 100 episodes.
2. Alignment probe in the structured-aligned regime.
3. A short open-loop replay test: feed a recorded robot episode token-by-token and verify predicted actions track within tolerance.

**Export for robot hardware.** The robot won't have a TPU. Two routes: (a) run JAX on the robot's GPU (`jax[cuda]` with the model loaded from Orbax); (b) export to ONNX or convert weights to PyTorch via `transformers`-compatible state dict for inference on whatever runtime the robot uses. Route (a) is simpler if a GPU is available; route (b) is more portable. Decide before training so the architecture stays exportable (e.g., avoid `pallas` kernels that don't have a non-TPU equivalent — for v1 this is already the case).

---

## §8. Order of operations for the coding agent

Build in this order. Each step has a passing test before moving on.

0. **TPU environment smoke test.** Set up the TPU VM, pin `jax`/`jaxlib`/`libtpu` versions together, run `jax.devices()` and confirm the topology. Then run a 10-line "linear regression on TPU" smoke test before importing any project code. This rules out ~80% of environment issues upfront.
1. **§1 schemas + validator** — Pydantic models, `validate_dataset.py`, run on a sample of 10 episodes from each side. *Test:* validator catches an intentionally-broken episode.
2. **§2.2 projection heads** as standalone NNX modules. *Test:* shape & dtype on dummy inputs; `nnx.split(model)` produces a non-empty state tree.
3. **§2.2.1 visual encoder.** Install `jimmy`, load DINOv2 ViT-B/14, write `eval/parity_dino.py`, and run it. Must pass before step 13. *Test:* max-abs-error on patch tokens vs PyTorch reference < 1e-3 over 32 images.
4. **§2 tokenizer** that reads HDF5 → produces a `(B, 6T, d_model)` array. *Test:* round-trip a known episode and check no dimension shuffling.
4a. **`build_grain_shards.py`** converts HDF5 → ArrayRecord. *Test:* one shard, one episode, equal contents to source HDF5.
5. **§3.2 transformer backbone** with random heads. *Tests:* (a) forward pass on CPU first (`JAX_PLATFORMS=cpu`), (b) then on a 1-chip TPU, (c) then on the full mesh. Each step catches a different class of bug.
6. **§3.4 masking module** in isolation. *Test:* given a batch with mixed `source_mask`, the right tokens are replaced and the right loss masks emerge. Run *under* `jax.jit` to catch traceability bugs. **This is the test that catches the most bugs — write it carefully.**
7. **§3.6 loss module**. *Test:* zero loss on a batch where predictions equal targets; nonzero otherwise; loss masks zero out the right contributions.
8. **§4 mixed sampler with grain**. *Test:* over 1000 batches, ratio of robot samples ≈ `w` ± 1%; deterministic given the same seed.
9. **Synthetic-data training loop.** Run a 10-step `train_step` on synthetic batches with the real `nnx.jit` + sharding. Shape and sharding bugs surface here, where the iteration is fast.
10. **§5 training loop** end-to-end on 100 episodes per side, 1k steps, single host. *Test:* loss decreases; no NaNs; Orbax checkpoint saves and loads correctly.
11. **§6.3 alignment probe** runs as part of eval. *Test:* on a batch where robot and human samples are identical (synthetic), Wasserstein ≈ 0 and discriminator accuracy ≈ 50%.
12. **§6.1 sim rollout** harness. *Test:* runs deterministically with a fixed seed.
13. **Scaled training run.** First at 10% of full step count and full per-chip batch size, on a single TPU host. If that converges, scale to multi-host. **Do not** debut multi-host on the full run.
14. **Full training run** on real Stage-0 + Stage-1 data, full step count, 3-way `w` sweep.
15. **§7 deployment** glue. Real-robot test happens **after** the §7 sanity gates pass.

---

## §9. What's deferred to v2 (do not build now)

- VIS-feature prediction loss (mentioned in §3.5).
- Explicit alignment objectives (CFG-ADDA from Lei et al.) — only add if §6.3 shows we're stuck in overlapping or disjoint regime.
- Multi-task heads (e.g., other manipulation tasks beyond box pickup).
- Larger backbones (>200M params).
- Action chunking / diffusion action head — start with single-step regression.

These are all real follow-ups, but each one introduces failure modes. Get the simple version working and probed first.

---

## §10. References (per-decision mapping)

| Decision in this plan | Source |
|------------------------|--------|
| Modality-aligned next-token prediction; missing-modality masking (§2, §3.4) | Radosavovic et al., arXiv:2402.19469 |
| Shared bridge as alignment mechanism; UMAP + discriminator probe; mixing ratio bounds (§3.1, §4, §6.3) | Lei et al., arXiv:2604.13645 |
| Retargeted reference motion in Stage 0 (upstream of this plan) | Yang et al., *OmniRetarget*, arXiv:2509.26633 |
| Aria + IMU rig for Stage 1 human data (upstream of this plan) | Mao et al., *RoSHI*, arXiv:2604.07331 |
| Transformer-as-implicit-optimizer perspective informing the choice of a decoder-only causal architecture | von Oswald et al., arXiv:2212.07677 |
| NNX over Linen for new projects (§0, §3.2)                                  | Flax docs (`flax.readthedocs.io`); Google Cloud "JAX for PyTorch developers" (2025) |
| FSDP via `nnx.with_partitioning` (§3.2, §5.1)                               | "Train a GPT2 model with JAX on TPU" (Google Developers Blog, 2025) |
| `shard_map` is reserved for tensor-parallel; FSDP sufficient at 110M (§5.1) | JAX scaling book ("How to Parallelize a Transformer for Training") |
| HuggingFace `transformers` v5 dropped Flax (§2.2.1)                          | "Transformers v5: Simple model definitions" (HF blog, December 2025); transformers#37262 (closed as not planned) |
| `jimmy` library for native NNX DINOv2 (§2.2.1)                              | `clementpoiret/jimmy` GitHub repo                                    |
| Don't use `FlaxDinov2Model` (§2.2.1)                                         | HF transformers#37246 (parity bug); pinned to `jax<=0.4.13`         |
| Orbax v1 for checkpointing (§5.4)                                           | Orbax docs; Flax NNX checkpointing guide                            |
| `grain` for input pipelines, deterministic by default (§4.2)                | JAX training cookbook                                                |
