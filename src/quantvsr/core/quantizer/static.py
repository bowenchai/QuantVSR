import torch
import torch.nn as nn
from typing import Tuple

from .base import BaseQuantizer
from ..config import QuantizerConfig


class StaticQuantizer(BaseQuantizer):
    """Static Quantizer

    Example:
        >>> from quantvsr.core.config import WeightQuantConfig
        >>> config = WeightQuantConfig(bits=8, symmetric=True, granularity="channel")
        >>> quantizer = StaticQuantizer(config)
        >>> weight = torch.randn(64, 128, 3, 3)
        >>> w_quant = quantizer(weight)
    """

    def __init__(self, config: QuantizerConfig, device: str | torch.device = "cuda"):
        super().__init__(config, device)

        self.x_min = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32, device=device))
        self.x_max = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=device))

        self.running_stat = False
        self.momentum = 0.95

    def compute_qparams(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute quantization parameters (scale, zero_point)."""
        if not self.initialized:
            self._init_from_data(x)
            self.initialized = True

        if self.running_stat and self.training:
            self._momentum_update(x)

        if self.config.symmetric:
            abs_max = torch.max(torch.abs(self.x_min), torch.abs(self.x_max))
            scale = abs_max / (2 ** (self.config.bits - 1) - 1)
            scale = scale.clamp(min=1e-4)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (self.x_max - self.x_min) / (2 ** (self.config.bits) - 1)
            scale = scale.clamp(min=1e-4)
            zero_point = self._round_ste(-self.x_min / scale)

        return scale, zero_point

    def _init_from_data(self, x: torch.Tensor):
        with torch.no_grad():
            if self.config.granularity == "channel":  # per-channel quantization
                x_min = self._reduce_per_channel(x, reduce_fn="min")
                x_max = self._reduce_per_channel(x, reduce_fn="max")
            else:  # per-tensor quantization
                x_min = x.min()
                x_max = x.max()

            self.x_min.data = x_min
            self.x_max.data = x_max

    def _momentum_update(self, x: torch.Tensor):
        """momentum update"""
        with torch.no_grad():
            if self.config.granularity == "channel":
                x_min = self._reduce_per_channel(x, reduce_fn="min")
                x_max = self._reduce_per_channel(x, reduce_fn="max")
            else:
                x_min = x.min()
                x_max = x.max()

            self.x_min.data = self.x_min.data * self.momentum + x_min * (1 - self.momentum)
            self.x_max.data = self.x_max.data * self.momentum + x_max * (1 - self.momentum)

    def _reduce_per_channel(self, x: torch.Tensor, reduce_fn: str) -> torch.Tensor:
        """Per-channel reduce"""
        dims = list(range(1, x.dim()))
        result = x

        for dim in reversed(dims):
            if reduce_fn == "min":
                result = result.min(dim=dim, keepdim=True)[0]
            else:
                result = result.max(dim=dim, keepdim=True)[0]

        return result

    def set_running_stat(self, running_stat: bool):
        """set running stat"""
        self.running_stat = running_stat

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Reshape x_min and x_max before loading ckpt.
        x_min_key = prefix + "x_min"
        x_max_key = prefix + "x_max"
        if x_min_key in state_dict:
            self.x_min.data = torch.zeros_like(state_dict[x_min_key])
            self.x_max.data = torch.zeros_like(state_dict[x_max_key])

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self.initialized = True
