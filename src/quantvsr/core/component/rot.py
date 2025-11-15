import torch
import torch.nn as nn
from typing import Union
from ..config.component import RotationConfig
from .utils.hadamard import random_hadamard_matrix


class RotationComponent(nn.Module):
    def __init__(
        self,
        config: RotationConfig,
        fp_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear],
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.config = config
        self.initialized = False
        self.register_buffer("rot_mat", torch.empty(0))
        self.device = device
        if type(fp_module) in [nn.Linear]:
            self.layer_type = "linear"
        elif type(fp_module) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            self.layer_type = "conv"

    def forward(self, w, x):
        if not self.config.enabled:
            return w, x
        if not self.initialized:
            self._init_rotation_matrix(w.shape[1], self.device)

        w_rotated = self._rotate_weight(w)
        a_rotated = self._rotate_activation(x)
        return w_rotated, a_rotated

    def _init_rotation_matrix(self, dim: int, device: str | torch.device = "cpu"):
        """Initialize Hadamard rotation matrix"""
        self.rot_mat.data = random_hadamard_matrix(dim, device)
        self.rot_mat: torch.Tensor
        self.initialized = True

    def _rotate_weight(self, w: torch.Tensor) -> torch.Tensor:
        layer_type = self.layer_type
        if layer_type == "linear":
            return w @ self.rot_mat
        elif layer_type == "conv":
            return torch.einsum("o i ... , i j -> o j ...", w, self.rot_mat.T)
        else:
            raise NotImplementedError(f"Rotation not implemented for {layer_type}")

    def _rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        layer_type = self.layer_type
        if layer_type == "linear":
            return x @ self.rot_mat
        elif layer_type == "conv":
            return torch.einsum("b c ... , j c -> b j ...", x, self.rot_mat)
        else:
            raise NotImplementedError(f"Rotation not implemented for {layer_type}")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Reshape rot_mat before loading ckpt.
        rot_mat_key = prefix + "rot_mat"
        if rot_mat_key in state_dict:
            self.rot_mat.data = torch.zeros_like(state_dict[rot_mat_key])
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self.initialized = True
