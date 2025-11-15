import torch
from typing import Tuple

from .base import BaseQuantizer
from ..config import QuantizerConfig


class DynamicQuantizer(BaseQuantizer):
    """Dynamic Quantizer

    Example:
        >>> from quantvsr.core.config import ActQuantConfig
        >>> config = ActQuantConfig(bits=8, dynamic=True)
        >>> quantizer = DynamicQuantizer(config)
        >>> act = torch.randn(1, 128, 32, 32)
        >>> a_quant = quantizer(act)
    """

    def __init__(self, config: QuantizerConfig, device: str | torch.device = "cuda"):
        super().__init__(config, device)
        self.initialized = True  # Dynamic quantizer is always initialized

    def compute_qparams(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute quantization parameters dynamically"""
        if self.config.granularity == "channel":  # Per-channel quantization
            x_min = self._reduce_per_channel(x, reduce_fn="min")
            x_max = self._reduce_per_channel(x, reduce_fn="max")
        else:
            x_min = x.min()
            x_max = x.max()
        if self.config.symmetric:
            abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
            scale = abs_max / (2 ** (self.config.bits - 1) - 1)
            scale = scale.clamp(min=1e-4)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (x_max - x_min) / (2 ** (self.config.bits) - 1)
            scale = scale.clamp(min=1e-4)
            zero_point = self._round_ste(-x_min / scale)

        return scale.detach(), zero_point.detach()

    def _reduce_per_channel(self, x: torch.Tensor, reduce_fn: str) -> torch.Tensor:
        """Per-channel reduce (same as StaticQuantizer)"""
        dims = list(range(1, x.dim()))
        result = x

        for dim in reversed(dims):
            if reduce_fn == "min":
                result = result.min(dim=dim, keepdim=True)[0]
            else:
                result = result.max(dim=dim, keepdim=True)[0]

        return result
