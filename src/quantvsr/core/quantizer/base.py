from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple


from ..config.quantizer import QuantizerConfig


class BaseQuantizer(ABC, nn.Module):
    def __init__(self, config: QuantizerConfig, device: str | torch.device = "cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.initialized = False

    @abstractmethod
    def compute_qparams(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """compute qparams: (scale, zero_point)"""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward: quantization + dequantization"""
        scale, zero_point = self.compute_qparams(x)
        x_q = self._quantize(x, scale, zero_point)
        x_dq = self._dequantize(x_q, scale, zero_point)
        return x_dq

    def _quantize(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        x_scaled = x / scale
        x_rounded = self._round_ste(x_scaled)

        if not self.config.symmetric:
            x_rounded = x_rounded + zero_point
        x_clamped = torch.clamp(x_rounded, self.config.qmin, self.config.qmax)
        return x_clamped

    def _dequantize(
        self, x_q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        if not self.config.symmetric:
            x_q = x_q - zero_point
        return x_q * scale

    def _round_ste(self, x: torch.Tensor) -> torch.Tensor:
        """STE round"""
        return (x.round() - x).detach() + x

    def extra_repr(self) -> str:
        return f"bits={self.config.bits}, symmetric={self.config.symmetric}"
