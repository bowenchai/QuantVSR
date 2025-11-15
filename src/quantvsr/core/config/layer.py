from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal
from pathlib import Path
import yaml

from .quantizer import WeightQuantConfig, ActQuantConfig
from .component import ComponentConfig


@dataclass
class QuantLayerConfig:
    channel_split: bool = True
    weight_quant: WeightQuantConfig = field(default_factory=WeightQuantConfig)
    act_quant: ActQuantConfig = field(default_factory=ActQuantConfig)
    components: ComponentConfig = field(default_factory=ComponentConfig)
