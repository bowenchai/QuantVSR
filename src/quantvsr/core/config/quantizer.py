from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class QuantizerConfig:
    bits: int = 4
    symmetric: bool = False
    dynamic: bool = False
    granularity: Literal["tensor", "channel"] = "tensor"

    def __post_init__(self):
        if not (2 <= self.bits <= 32):
            raise ValueError(f"bits must be in [2, 32], got {self.bits}")

    @property
    def qmin(self) -> int:
        if self.symmetric:
            return -(2 ** (self.bits - 1) - 1)
        return 0

    @property
    def qmax(self) -> int:
        if self.symmetric:
            return 2 ** (self.bits - 1) - 1
        return 2**self.bits - 1


@dataclass
class WeightQuantConfig(QuantizerConfig):
    """Weight Quantization Config

    default: 4-bit, asymmetric, per-channel, static
    """

    bits: int = 4
    symmetric: bool = False
    dynamic: bool = False
    granularity: Literal["tensor", "channel"] = "channel"


@dataclass
class ActQuantConfig(QuantizerConfig):
    """Activation Quantization Config

    default: 4-bit, asymmetric, layer-wise, static
    """

    bits: int = 4
    symmetric: bool = False
    dynamic: bool = True
    granularity: Literal["tensor", "channel"] = "tensor"
