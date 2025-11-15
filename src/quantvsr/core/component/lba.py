import torch
import torch.nn as nn
from typing import Union

from ..config import LBAConfig


class LBAComponent(nn.Module):
    def __init__(
        self,
        config: LBAConfig,
        fp_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear],
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.config = config
        shape = (1, fp_module.weight.shape[0]) + (1,) * (fp_module.weight.dim() - 2)
        self.lba = nn.Parameter(torch.zeros(shape, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.enabled:
            return x
        return x + self.lba
