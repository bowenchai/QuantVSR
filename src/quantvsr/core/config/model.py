"""Model-level Quantization Configuration"""

from dataclasses import dataclass, field, asdict
from typing import List

from .layer import QuantLayerConfig


@dataclass
class QuantModelConfig:
    method_name: str = "QuantVSR"
    layer_types: List[str] = field(default_factory=lambda: ["Conv1d", "Conv2d", "Conv3d", "Linear"])
    layer_config: QuantLayerConfig = field(default_factory=QuantLayerConfig)
    exclude_layers: List[str] = field(
        default_factory=lambda: ["time_embed.0", "input_blocks.0.0", "out.2"]
    )

    def to_dict(self):
        return asdict(self)
