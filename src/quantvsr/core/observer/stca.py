from typing import Dict, Tuple
from einops import rearrange
import torch
from ..layer import BaseQuantLayer
from ..model import QuantVSRModel


class STCAObserver:
    """Observer for collecting STCA statistics of each QuantLayer."""

    def __init__(self, model: QuantVSRModel):
        self.model = model
        self.statistics = {}
        self.hooks = []

    def _hook_fn(self, name: str):
        def hook(module: BaseQuantLayer, input: Tuple[torch.Tensor]):
            x = input[0]  # (T, C, H, W) for common
            if "emb_layers" in name or "time_embed" in name:
                return
            else:
                if "temporal_conv" in name:  # (B, C, T, H, W)
                    x = rearrange(x, "b c t h w -> (b t) c h w")
                value = self._compute_statistics(x)

            if name not in self.statistics:
                self.statistics[name] = []
            self.statistics[name].append(value)

        return hook

    def _compute_statistics(self, x: torch.Tensor) -> Dict:
        return {
            "c_spatial": compute_c_space(x),
            "c_temporal": compute_c_temporal(x),
        }

    def aggregate_statistics(self):
        aggregated = {}
        for name, stats_list in self.statistics.items():
            aggregated[name] = {
                "c_spatial": [s["c_spatial"] for s in stats_list],
                "c_temporal": [s["c_temporal"] for s in stats_list],
            }
        self.statistics = aggregated

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, BaseQuantLayer):
                hook_handle = module.register_forward_pre_hook(self._hook_fn(name))
                self.hooks.append(hook_handle)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def reset(self):
        self.statistics.clear()


# Input shape is [T, C, H, W]
def compute_c_space(x: torch.Tensor):
    x = x.view(x.shape[0], x.shape[1], -1)  # (T, C, -1)
    return x.std(dim=-1).mean((1, 0))


def compute_c_temporal(x: torch.Tensor):
    T = x.shape[0]
    rest_dims = x.shape[1:]

    frame_diff = x[1:] - x[:-1]

    diff_flat = frame_diff.reshape(T - 1, -1)
    diff_energy = torch.sum(diff_flat**2, dim=1)

    num_spatial_elements = torch.prod(torch.tensor(rest_dims)).item()
    normalized_energy = diff_energy / num_spatial_elements

    return torch.mean(normalized_energy)
