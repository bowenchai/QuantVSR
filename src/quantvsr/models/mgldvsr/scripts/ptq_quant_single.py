import argparse, os, sys, glob
import yaml
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(base_path)

from quantization.methods import load_config
from quantization.apply_quant import apply_quant
from quantization.load_quant import load_Quantmodel
from quantization.calibration import calibration
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
import torch.nn.functional as F
import cv2
from util_image import ImageSpliterTh
from pathlib import Path
from basicsr.archs.arch_util import resize_flow
from scripts.util_flow import forward_backward_consistency_check
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    return 2.*image - 1.


def read_image(im_path):
    im = np.array(Image.open(im_path).convert("RGB"))
    im = im.astype(np.float32) / 255.0
    im = im[None].transpose(0, 3, 1, 2)
    im = (torch.from_numpy(im) - 0.5) / 0.5

    return im


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seqs-path",
        type=str,
        nargs="?",
        help="path to the input image",
        default="inputs/user_upload"
    )
    # parser.add_argument(
    #     "--init-img",
    #     type=str,
    #     nargs="?",
    #     help="path to the input image",
    #     default="inputs/user_upload"
    # )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/user_upload"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device",
        default="cuda"
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=1000,
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=5,
        help="number of frames to perform inference",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/stablevsr_025.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        default="checkpoints/vqgan_cfw_00011.ckpt",
        help="path to checkpoint of VQGAN model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--select_idx",
        type=int,
        default=0,
        help="selected sequence index",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="number of gpu for testing",
    )
    parser.add_argument(
        "--dec_w",
        type=float,
        default=0.5,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=32,
        help="tile overlap size (in latent)",
    )
    parser.add_argument(
        "--upscale",
        type=float,
        default=4.0,
        help="upsample scale",
    )
    parser.add_argument(
        "--colorfix_type",
        type=str,
        default="nofix",
        help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
    )
    parser.add_argument(
        "--vqgantile_stride",
        type=int,
        default=750,
        help="the stride for tile operation before VQGAN decoder (in pixel)",
    )
    parser.add_argument(
        "--vqgantile_size",
        type=int,
        default=960,
        help="the size for tile operation before VQGAN decoder (in pixel)",
    )
    parser.add_argument(
        "--qvsr_options",
        type=str,
        default="configs/quant_config.yaml",
        help="the size for tile operation before VQGAN decoder (in pixel)",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    print('>>>>>>>>>>color correction>>>>>>>>>>>')
    if opt.colorfix_type == 'adain':
        print('Use adain color correction')
    elif opt.colorfix_type == 'wavelet':
        print('Use wavelet color correction')
    else:
        print('No color correction')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # Testing
    select_idx = opt.select_idx
    num_gpu_test = opt.n_gpus

    # Model
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    model.configs = config
    quant_config = load_config(opt.qvsr_options)["quantization_options"]

    calibration(model, quant_config, device)

if __name__ == "__main__":
    main()
