import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Literal, Optional, Callable, List, Tuple, Type

from ..config import QuantModelConfig
from ..layer import QuantConv1d, QuantConv2d, QuantConv3d, QuantLinear, BaseQuantLayer
from ...utils.logger import get_logger


class QuantVSRModel(nn.Module):
    """QuantVSR Model"""

    # Mapping from layer type name to (FP class, Quant class)
    _LAYER_TYPE_MAP: Dict[str, Tuple[Type[nn.Module], Type[nn.Module]]] = {
        "Conv1d": (nn.Conv1d, QuantConv1d),
        "Conv2d": (nn.Conv2d, QuantConv2d),
        "Conv3d": (nn.Conv3d, QuantConv3d),
        "Linear": (nn.Linear, QuantLinear),
    }

    def __init__(self, fp_model: nn.Module, config: QuantModelConfig):
        super().__init__()
        self.model = fp_model
        self.config = config
        self.device = next(fp_model.parameters()).device
        self.logger = get_logger("quantvsr")
        self._quantize_model()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def set_precision(self, precision: Literal["fp", "low_bit"]):
        for m in self.modules():
            if isinstance(m, BaseQuantLayer):
                m.use_fp = True if precision == "fp" else False

    def _quantize_model(self):
        self._replace_layers(self.model, "")

    def _replace_layers(self, module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # exclude the first and the last layers.
            if any(full_name.startswith(ex) for ex in self.config.exclude_layers):
                continue

            replaced = False
            for layer_type in self.config.layer_types:
                fp_class, quant_class = self._LAYER_TYPE_MAP[layer_type]
                if isinstance(child, fp_class):
                    quant_layer = quant_class(child, self.config.layer_config)
                    setattr(module, name, quant_layer)
                    replaced = True
                    break

            # Recursively replace in child modules if not replaced
            if not replaced and len(list(child.children())) > 0:
                self._replace_layers(child, full_name)

    def save_qparams(self, save_path: str | Path):
        """Save quantization parameters (scales, zero_points, STCA factors)"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, BaseQuantLayer):
                assert module.check_init(), "There are uninitialized QuantLayer."
                state_dict[name] = module.state_dict()
        torch.save(state_dict, save_path)

        self.logger.info(f"Quantization parameters saved to {save_path}")

    def load_qparams(self, load_path: str | Path):
        """Load quantization parameters"""
        load_path = Path(load_path)
        state_dict = torch.load(load_path, map_location=self.device)

        for name, module in self.named_modules():
            if isinstance(module, BaseQuantLayer) and name in state_dict:
                params = state_dict[name]
                module.load_state_dict(params)
                assert module.check_init(), "Load qparams failed."

        self.logger.info(f"Quantization parameters loaded from {load_path}")
