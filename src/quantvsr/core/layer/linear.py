import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Callable, Any


from .base import BaseQuantLayer
from ..config import QuantLayerConfig


class QuantLinear(BaseQuantLayer):
    def __init__(self, fp_module: nn.Linear, config: QuantLayerConfig):
        super().__init__(fp_module, config)

    @property
    def _fp_forward(self) -> Callable:
        return F.linear

    def _channel_split(
        self, w: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w0, w1 = torch.chunk(w, 2, dim=1)
        x0, x1 = torch.chunk(x, 2, dim=-1)
        return w0, w1, x0, x1

    def _channel_cat(
        self, w0: torch.Tensor, w1: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.cat((w0, w1), dim=1)
        x = torch.cat((x0, x1), dim=-1)
        return w, x
