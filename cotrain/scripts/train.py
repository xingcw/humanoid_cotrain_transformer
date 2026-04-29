"""End-to-end training entrypoint (PROJECT_PLAN_1.md §8 step 10).

Wires loader → encoder → transformer → loss → optimizer → checkpoint and
streams metrics to wandb. Auto-resumes from the most recent Orbax
checkpoint under `--ckpt-dir` if one exists.

Typical usage (the §8.10 manual milestone):

    python -m cotrain.scripts.train \\
        --root datasets \\
        --w 0.25 --batch-size 8 --steps 1000 \\
        --ckpt-dir runs/dev \\
        --wandb-project humanoid_cotrain --wandb-run-name dev_smoke

For a CI-friendly synthetic-data smoke (10 episodes, 50 steps), pass
`--steps 50 --no-pretrained-encoder`.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import cotrain  # ensures the jimmy compat shim runs first
from cotrain.models.encoders import DinoV2Encoder
from cotrain.models.transformer import CoTrainTransformer, TransformerConfig
from cotrain.training.checkpointing import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from cotrain.training.sampler import make_mixed_loader
from cotrain.training.sharding import data_sharding, make_mesh, shard_batch
from cotrain.training.trainer import (
    OptimizerConfig,
    make_optimizer,
    train_step_with_encoder,
)


_MODEL_KEYS = {
    "rgb", "state_robot", "state_human", "box", "phase", "contact", "action",
    "source_mask",
}


def _to_jax(batch: dict) -> dict[str, jnp.ndarray]:
    return {k: jnp.asarray(batch[k]) for k in _MODEL_KEYS}


def _maybe_init_wandb(args: argparse.Namespace, cfg_dict: dict):
    if args.wandb_project is None:
        return None
    import wandb
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=cfg_dict,
        resume="allow",
    )
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("datasets"))
    ap.add_argument("--w", type=float, default=0.25,
                    help="mixing ratio (round(w*B) robot rows per batch)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--T", type=int, default=16, help="window length")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("runs/dev"))
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--no-pretrained-encoder", action="store_true",
                    help="Use random-init DINO (offline; for smoke tests).")
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--lr-peak", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)
    args = ap.parse_args()

    is_main = jax.process_index() == 0

    # --- model, optimizer, encoder ---------------------------------------
    cfg = TransformerConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        T=args.T,
    )
    model = CoTrainTransformer(cfg, rngs=nnx.Rngs(args.seed))
    optimizer = make_optimizer(model, OptimizerConfig(
        lr_peak=args.lr_peak,
        warmup_steps=args.warmup_steps,
        decay_steps=args.steps,
    ))
    encoder = DinoV2Encoder(rngs=nnx.Rngs(args.seed + 1),
                            pretrained=not args.no_pretrained_encoder)

    # --- resume if checkpoint present ------------------------------------
    start_step = 0
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    if (latest := latest_checkpoint(args.ckpt_dir)) is not None:
        start_step, _ = load_checkpoint(latest, model=model, optimizer=optimizer)
        if is_main:
            print(f"resumed from {latest} at step {start_step}")

    # --- wandb -----------------------------------------------------------
    run = _maybe_init_wandb(args, cfg_dict=vars(args)) if is_main else None

    # --- mesh + loader ---------------------------------------------------
    mesh = make_mesh()
    sharding = data_sharding(mesh)
    loader = make_mixed_loader(
        args.root, w=args.w, batch_size=args.batch_size, seed=args.seed,
    )

    # --- train loop ------------------------------------------------------
    if is_main:
        print(f"training for {args.steps - start_step} steps "
              f"(batch={args.batch_size}, T={args.T}, w={args.w})")
    t0 = time.perf_counter()
    for step in range(start_step, args.steps):
        batch = shard_batch(_to_jax(next(loader)), sharding)
        loss, aux = train_step_with_encoder(model, optimizer, encoder, batch)
        loss_v = float(loss)
        if not np.isfinite(loss_v):
            raise RuntimeError(f"non-finite loss at step {step}: {loss_v}")

        if is_main and (step % args.log_every == 0 or step == args.steps - 1):
            elapsed = time.perf_counter() - t0
            steps_done = step - start_step + 1
            rate = steps_done / max(elapsed, 1e-6)
            print(f"step {step:>6d} | loss {loss_v:.4f} | "
                  f"action {float(aux['loss/action']):.4f} | "
                  f"box {float(aux['loss/box']):.4f} | "
                  f"phase {float(aux['loss/phase']):.4f} | "
                  f"{rate:.2f} step/s")
            if run is not None:
                run.log({
                    "step": step,
                    "loss/total":   loss_v,
                    "loss/action":  float(aux["loss/action"]),
                    "loss/state":   float(aux["loss/state"]),
                    "loss/box":     float(aux["loss/box"]),
                    "loss/phase":   float(aux["loss/phase"]),
                    "loss/contact": float(aux["loss/contact"]),
                    "throughput/step_per_sec": rate,
                })

        if step > start_step and step % args.ckpt_every == 0:
            save_checkpoint(args.ckpt_dir, model=model, optimizer=optimizer,
                            step=step, metadata={"w": args.w, "seed": args.seed})

    # Always checkpoint at the end.
    save_checkpoint(args.ckpt_dir, model=model, optimizer=optimizer,
                    step=args.steps, metadata={"w": args.w, "seed": args.seed,
                                                "final": True})
    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
