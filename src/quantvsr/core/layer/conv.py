import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Callable, Any, Union

from .base import BaseQuantLayer
from ..config import QuantLayerConfig


class QuantConv(BaseQuantLayer):
    def __init__(self, fp_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], config: QuantLayerConfig):
        super().__init__(fp_module, config)

    def _channel_split(
        self, w: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w0, w1 = torch.chunk(w, 2, dim=1)
        x0, x1 = torch.chunk(x, 2, dim=1)
        return w0, w1, x0, x1

    def _channel_cat(
        self, w0: torch.Tensor, w1: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.cat((w0, w1), dim=1)
        x = torch.cat((x0, x1), dim=1)
        return w, x


class QuantConv1d(QuantConv):
    def __init__(self, fp_module: nn.Conv1d, config: QuantLayerConfig):
        super().__init__(fp_module, config)

    @property
    def _fp_forward(self) -> Callable:
        return F.conv1d


class QuantConv2d(QuantConv):
    def __init__(self, fp_module: nn.Conv2d, config: QuantLayerConfig):
        super().__init__(fp_module, config)

    @property
    def _fp_forward(self) -> Callable:
        return F.conv2d


class QuantConv3d(QuantConv):
    def __init__(self, fp_module: nn.Conv3d, config: QuantLayerConfig):
        super().__init__(fp_module, config)

    @property
    def _fp_forward(self) -> Callable:
        return F.conv3d
