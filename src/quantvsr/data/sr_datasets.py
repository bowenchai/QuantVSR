import random
from typing import List
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F

from ..utils.logger import get_logger


def read_video_frames(video_folder_path: Path, min_frames=None) -> List[np.ndarray]:
    """Read video frames from an image folder.

    Args:
        video_folder_path: Path to folder containing image frames
        min_frames: Minimum number of frames to read. If fewer frames exist, pad with last frame.

    Returns:
        List of (H, W, C) numpy arrays, uint8
    """
    # Get all image files sorted by name
    image_files = sorted(
        [
            f
            for f in video_folder_path.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
        ]
    )

    frames = []
    for img_path in image_files:
        frame = np.array(Image.open(img_path).convert("RGB"))
        frames.append(frame)

    # Pad with last frame if needed
    if min_frames is not None and len(frames) < min_frames:
        last_frame = frames[-1]
        while len(frames) < min_frames:
            frames.append(last_frame.copy())

    return frames


class VSRDataset(Dataset):
    """Dataset for VSR tasks."""

    def __init__(
        self,
        data_root: str,  # Path to LQ video folders
        device: str | torch.device,
        upscale: int = 4,  # Upscale factor
        video_mode: str = "image_folder",  # 'image_folder' or 'video'
        cali_resolution: str = "5x512x512",
    ):
        super().__init__()
        self.logger = get_logger("quantvsr")
        self.device = device
        self.lq_vdieos = None
        self.video_mode = video_mode
        self.upscale = upscale

        self.frames, self.height, self.width = map(int, cali_resolution.split("x"))

        lq_dir = Path(data_root)

        if video_mode == "image_folder":
            self.lq_video_paths = sorted([f for f in lq_dir.iterdir() if f.is_dir()])
            self.logger.debug(f"Found {len(self.lq_video_paths)} videos.")
        else:
            raise NotImplementedError("Currently we only support image_folder.")

        self._frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]  # -1, 1
        )

    def __len__(self):
        return len(self.lq_video_paths)

    def __getitem__(self, index):
        """
        Return vdieo shape: (F, H, W, C), float32 in [-1, 1]
        """
        # lq_video shape: List of (H, W, C), uint8
        lq_video = read_video_frames(self.lq_video_paths[index], self.frames)
        lq_video = self.preprocess(lq_video)
        video_name = self.lq_video_paths[index].name
        return {
            "lq_video": lq_video,
            "name": video_name,
        }

    def preprocess(self, lq_video: List[np.ndarray]) -> torch.Tensor:
        """Preprocess video. Upscale first, then random crop.
        Args:
            lq_video: List of (H, W, C), uint8

        Returns:
            torch.Tensor: (F, H, W, C), float32 in [-1, 1]
        """
        F_lq, H_lq, W_lq, _ = len(lq_video), *lq_video[0].shape

        beg_frame = random.randint(0, F_lq - self.frames) if F_lq > self.frames else 0

        lq_frames = [
            self._frame_transform(torch.tensor(lq_video[i]))
            for i in range(beg_frame, beg_frame + self.frames)
        ]
        lq_tensor = torch.stack(lq_frames, dim=0)

        # Upscale LQ by upscale factor (F, H, W, C)
        lq_tensor = lq_tensor.permute(0, 3, 1, 2)  # (F, C, H, W)
        upscaled_h = H_lq * self.upscale
        upscaled_w = W_lq * self.upscale
        lq_tensor = F.interpolate(
            lq_tensor, size=(upscaled_h, upscaled_w), mode="bicubic", align_corners=False
        )
        lq_tensor = lq_tensor.permute(0, 2, 3, 1)  # (F, H, W, C)
        lq_tensor = torch.clamp(lq_tensor, -1.0, 1.0)

        # Random crop to target size
        top = random.randint(0, upscaled_h - self.height) if upscaled_h > self.height else 0
        left = random.randint(0, upscaled_w - self.width) if upscaled_w > self.width else 0

        ch = min(self.height, upscaled_h)
        cw = min(self.width, upscaled_w)

        lq_cropped = lq_tensor[:, top : top + ch, left : left + cw, :]

        return lq_cropped


class VSRValDataset(VSRDataset):
    def __init__(
        self,
        data_root: str,  # Path to LQ video folders
        device: str | torch.device,
        upscale: int = 4,  # Upscale factor
        video_mode: str = "image_folder",  # 'image_folder' or 'video'
    ):
        super().__init__(
            data_root, device, upscale=upscale, video_mode=video_mode, cali_resolution="1x1x1"
        )

    def preprocess(self, lq_video: List[np.ndarray]) -> torch.Tensor:
        """Preprocess video without cropping, with upscaling.
        Args:
            lq_video: List of (H, W, C), uint8

        Returns:
            torch.Tensor: (F, H, W, C), float32 in [-1, 1]
        """
        H_lq, W_lq, _ = lq_video[0].shape

        lq = torch.stack([self._frame_transform(torch.tensor(f)) for f in lq_video], dim=0)

        # Upscale LQ by upscale factor
        lq = lq.permute(0, 3, 1, 2)  # (F, C, H, W)
        upscaled_h = H_lq * self.upscale
        upscaled_w = W_lq * self.upscale
        lq = F.interpolate(lq, size=(upscaled_h, upscaled_w), mode="bicubic", align_corners=False)
        lq = lq.permute(0, 2, 3, 1)  # (F, H, W, C)
        lq = torch.clamp(lq, -1.0, 1.0)

        return lq


def custom_collate_fn(batch):
    lq_videos = []
    names = []

    for sample in batch:
        lq_videos.append(sample["lq_video"])
        names.append(sample["name"])

    lq_videos = torch.stack(lq_videos, dim=0)

    return {
        "lq_video": lq_videos,
        "name": names,
    }
