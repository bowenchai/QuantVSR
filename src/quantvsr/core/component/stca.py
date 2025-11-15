import torch
import torch.nn as nn
from typing import Tuple, Union

from ..config import STCAConfig


class STCAComponent(nn.Module):
    def __init__(
        self,
        config: STCAConfig,
        fp_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear],
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.config = config
        self._fp_weight = fp_module.weight.data
        self.register_buffer("_rank", torch.tensor(self.config.min_rank, device=device))
        self.rank = int(self._rank.item())
        self.device = device
        self.initialized = False

    def forward(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self._init_svd(self._fp_weight)
        if not self.config.enabled:
            return torch.zeros_like(w), w
        assert w.allclose(self._fp_weight), "fp weight is changed."
        w_skip = (self.L1 @ self.L2).view_as(w)
        return w_skip, w - w_skip

    @property
    def rank(self) -> int:
        return int(self._rank.item())

    @rank.setter
    def rank(self, value: int):
        self._rank.fill_(value)
        m = self._fp_weight.shape[0]
        n = self._fp_weight.numel() // m
        L1 = torch.zeros((m, self.rank), dtype=self._fp_weight.dtype, device=self._fp_weight.device)
        L2 = torch.zeros((self.rank, n), dtype=self._fp_weight.dtype, device=self._fp_weight.device)
        if hasattr(self, "L1"):
            self.L1.data = L1
            self.L2.data = L2
        else:
            self.L1 = nn.Parameter(L1)
            self.L2 = nn.Parameter(L2)
        self.initialized = False

    def _init_svd(self, fp_weight):
        r = self._rank
        W = fp_weight.view(fp_weight.shape[0], -1).to(torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        L1 = (U[:, :r] @ torch.diag(S[:r])).to(self._fp_weight).detach()
        L2 = Vh[:r, :].to(self._fp_weight).detach()

        if hasattr(self, "L1"):
            self.L1.data = L1
            self.L2.data = L2
        else:
            self.L1 = nn.Parameter(L1)
            self.L2 = nn.Parameter(L2)

        self.initialized = True

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Reshape L1, L2 according to rank before loading ckpt.
        rank_key = prefix + "_rank"
        if rank_key in state_dict:
            self.rank = state_dict[rank_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self.initialized = True
