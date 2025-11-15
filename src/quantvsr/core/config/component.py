"""Component Configuration for Quantization Components"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RotationConfig:
    """Configuration for Rotation Component

    Args:
        enabled: Whether to use rotation
        layer_type: FP layer type (linear or conv)
    """

    enabled: bool = True


@dataclass
class LBAConfig:
    """Configuration for Learnable Bias Alignment

    Args:
        enabled: Whether to use LBA
    """

    enabled: bool = True


@dataclass
class STCAConfig:
    """Configuration for Spatial-Temporal Complexity Aware

    Args:
        enabled: Whether to use STCA
    """

    enabled: bool = True
    min_rank: int = 16
    max_rank: int = 64
    percentile: float = 0.25


@dataclass
class ComponentConfig:
    """Unified Component Configuration"""

    rotation: RotationConfig = field(default_factory=RotationConfig)
    lba: LBAConfig = field(default_factory=LBAConfig)
    stca: STCAConfig = field(default_factory=STCAConfig)
