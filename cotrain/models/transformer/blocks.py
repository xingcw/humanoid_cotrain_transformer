"""Decoder transformer block (PROJECT_PLAN_1.md §3.2).

Pre-norm decoder block, FFN expansion 4×, GELU, dropout 0.1, causal
self-attention. We use `nnx.MultiHeadAttention` with the `decode=False`
training-time path. The plan calls out that `jax.nn.dot_product_attention
(..., is_causal=True)` is acceptable on recent JAX; under the hood that's
what NNX's MHA delegates to. Sticking to NNX keeps parameter sharding
annotations consistent with the rest of the model.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx


def _kernel_init():
    """Xavier init with FSDP-ready partitioning on the output (model) axis."""
    return nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, "model"))


class TransformerBlock(nnx.Module):
    """One pre-norm decoder block: Attn(LN(x)) → x + FFN(LN(x)) → x."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        ffn_expansion: int = 4,
        dropout_rate: float = 0.1,
        rngs: nnx.Rngs,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            qkv_features=d_model,
            decode=False,
            kernel_init=_kernel_init(),
            rngs=rngs,
        )
        self.drop1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        d_ff = ffn_expansion * d_model
        self.fc1 = nnx.Linear(d_model, d_ff, rngs=rngs, kernel_init=_kernel_init())
        self.fc2 = nnx.Linear(d_ff, d_model, rngs=rngs, kernel_init=_kernel_init())
        self.drop2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        # Pre-norm self-attention with causal mask. NNX builds the mask from
        # `mask=` or accepts the `is_causal` kw on newer JAX; we pass mask=None
        # and rely on the explicit `decode=False` causal path being applied via
        # the boolean lower-triangular mask we compute here. This avoids any
        # version drift in is_causal kw plumbing.
        L = x.shape[1]
        causal = nnx.make_causal_mask(jnp.ones((1, L)), dtype=jnp.bool_)
        h = self.norm1(x)
        h = self.attn(h, h, h, mask=causal, deterministic=deterministic)
        h = self.drop1(h, deterministic=deterministic)
        x = x + h
        # Pre-norm FFN with GELU.
        h = self.norm2(x)
        h = nnx.gelu(self.fc1(h))
        h = self.fc2(h)
        h = self.drop2(h, deterministic=deterministic)
        return x + h
