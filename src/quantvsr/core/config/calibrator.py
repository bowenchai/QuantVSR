from dataclasses import dataclass, field, asdict
from typing import Dict, Literal
from .model import QuantModelConfig


@dataclass
class CalibratorConfig:
    # model config
    fp_model_name: str = "mgldvsr"
    quant_model_config: QuantModelConfig = field(default_factory=QuantModelConfig)
    qparam_save_path: str = "./weights/qparams.pth"
    # data config
    cali_data_path: str = "./data/REDS30/LQ"  # or "./data/cali_data.pt"
    cali_sample_nums: int = 600
    upscale: int = 4
    # train settings
    num_iterations: int = 1800
    learning_rate: float = 1e-3
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0
    report_to: Literal["", "wandb"] = "wandb"

    def to_dict(self) -> Dict:
        return asdict(self)
