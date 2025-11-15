import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Union, Callable, Optional, Any
from abc import ABC, abstractmethod


from ..config import QuantLayerConfig
from ..quantizer import StaticQuantizer, DynamicQuantizer
from ..component import STCAComponent, RotationComponent, LBAComponent


class BaseQuantLayer(ABC, nn.Module):
    """BaseQuantLayer

    Subclass must define _fp_forward, _fp_kwargs, _channel_split(...) and _channel_cat(...)
    """

    def __init__(
        self,
        fp_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear],
        config: QuantLayerConfig,
    ):
        super().__init__()
        self.config = config
        self.device = next(fp_module.parameters()).device
        self.fp_weight = fp_module.weight.data
        self.fp_bias = fp_module.bias.data if fp_module.bias is not None else None
        self._fp_kwargs = {
            k: getattr(fp_module, k)
            for k in ("stride", "padding", "dilation", "groups")
            if hasattr(fp_module, k)
        }
        self.use_fp = False

        # Quantizer
        self.wq = StaticQuantizer(self.config.weight_quant, self.device)
        self.aq = DynamicQuantizer(self.config.act_quant, self.device)
        # Component
        self.stca = STCAComponent(self.config.components.stca, fp_module, self.device)
        self.rot = RotationComponent(self.config.components.rotation, fp_module, self.device)
        self.lba = LBAComponent(self.config.components.lba, fp_module, self.device)

        # Channel split
        if self.config.channel_split:
            self.wq1 = StaticQuantizer(self.config.weight_quant, self.device)
            self.aq1 = DynamicQuantizer(self.config.act_quant, self.device)
            self.rot1 = RotationComponent(self.config.components.rotation, fp_module, self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.use_fp:
            return self._fp_forward(x, self.fp_weight, self.fp_bias, **self._fp_kwargs)
        w = self.fp_weight
        w_skip, w = self.stca(w)

        # FP branch
        out_fp = self._fp_forward(x, w_skip, None, **self._fp_kwargs)

        # Quantization branch
        if self.config.channel_split:
            w0, w1, x0, x1 = self._channel_split(w, x)

            # Rotation
            w0, x0 = self.rot(w0, x0)
            w1, x1 = self.rot1(w1, x1)

            # Fake quantization
            w0, x0 = self.wq(w0), self.aq(x0)
            w1, x1 = self.wq1(w1), self.aq1(x1)

            w, x = self._channel_cat(w0, w1, x0, x1)
        else:
            w, x = self.rot(w, x)
            w, x = self.wq(w), self.aq(x)
        out_q = self._fp_forward(x, w, self.fp_bias, **self._fp_kwargs)

        # Add
        out = out_q + out_fp
        out = self.lba(out)
        return out

    def check_init(self) -> bool:
        check_components = [self.wq, self.aq]

        if self.config.components.stca.enabled:
            check_components.append(self.stca)
        if self.config.components.rotation.enabled:
            check_components.append(self.rot)

        if self.config.channel_split:
            check_components.extend([self.wq1, self.aq1])
            if self.config.components.rotation.enabled:
                check_components.append(self.rot1)

        return all(c.initialized for c in check_components)

    @property
    @abstractmethod
    def _fp_forward(self) -> Callable:
        pass

    @abstractmethod
    def _channel_split(
        self, w: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def _channel_cat(
        self, w0: torch.Tensor, w1: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
