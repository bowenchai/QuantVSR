from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseVSRModel(ABC, nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def load_pretrained(self, ckpt_paths: Dict[str, str], **kwargs):
        """Load pretrained weights.
        Args:
            ckpt_paths: Dictionary of checkpoint paths (e.g., unet_ckpt_path, vae_ckpt_path)
            **kwargs: Additional arguments
        """
        pass

    @abstractmethod
    def inference(
        self, lq_frames: torch.Tensor, save_path: Optional[str | Path] = None, **kwargs
    ) -> torch.Tensor:
        """Inference interface.
        Args:
            lq_frames: [B, T, C, H, W]
        Returns:
            hq_frames: [B, T, C, H, W]
        """
        pass

    @abstractmethod
    def get_quantizable_modules(self) -> Dict[str, nn.Module]:
        """Return a dictionary of quantizable modules for PTQ.
        Returns:
            Dict[str, nn.Module], for example：
            {
                'unet': self.unet,
                'first_stage_model': self.first_stage_model,
                'cond_stage_model': self.cond_stage_model
            }
        """
        pass

    @abstractmethod
    def generate_calibration_data(
        self, dataloader: DataLoader, sample_num: int = 600, save_path: str = "./data/cali_data.pt"
    ):
        """Generate calibration data for quantization (saved as .pt).
        Args:
            dataloader: DataLoader for the calibration data
        """
        pass
