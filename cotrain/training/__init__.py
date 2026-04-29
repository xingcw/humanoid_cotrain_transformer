"""Training utilities. Subpackages are imported lazily to avoid circulars.

The transformer backbone (`cotrain.models.transformer.backbone`) imports
`apply_modality_masks` from `cotrain.training.masking`. If we eagerly
import `trainer` here, that pulls the transformer back, and Python's
half-initialized `cotrain.models.transformer` package raises an ImportError.
So we keep this __init__ skinny and let callers import from submodules
directly when they need the heavier pieces.
"""
from cotrain.training.losses import LossWeights, compute_loss
from cotrain.training.masking import apply_modality_masks

__all__ = [
    "LossWeights",
    "apply_modality_masks",
    "compute_loss",
]
