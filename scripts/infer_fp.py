from pathlib import Path
import torch
import argparse
import pyfiglet
from colorama import Fore, init
from tqdm.rich import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from quantvsr.models import get_model, list_models
from quantvsr.utils.logger import setup_logger
from quantvsr.data.sr_datasets import VSRDataset, VSRValDataset, custom_collate_fn

# Prevent disrupting the tqdm progress bar
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*")

if __name__ == "__main__":
    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="the device to use (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/SPMCS/LQ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments/infer_fp/SPMCS",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    # -------------------- Preparation --------------------
    logger = setup_logger("quantvsr")
    seed_everything(args.seed)
    device = args.device

    # -------------------- Model --------------------
    ckpt_path = {
        "unet_ckpt_path": "weights/mgldvsr/mgldvsr_unet.ckpt",
        "vae_ckpt_path": "weights/mgldvsr/video_vae_cfw.ckpt",
    }
    model = get_model("mgldvsr", pretrained_path=ckpt_path, device=device)

    # -------------------- Data --------------------
    datasets = VSRValDataset(args.input_dir, device=device, upscale=args.upscale)
    dataloader = DataLoader(datasets, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # -------------------- Logging  --------------------
    init(autoreset=True)
    project_name = "QuantVSR"
    ascii_art = pyfiglet.figlet_format(project_name)
    print(Fore.CYAN + ascii_art)

    logger.info("Currently in full-precision inference")
    logger.info(f"Model: {type(model)}")
    logger.info(f"Video: {Path(args.input_dir).absolute()} with {len(datasets)} videos")
    logger.info(f"Output: {Path(args.output_dir).absolute()}")
    logger.info(f"VSR upscale factor: {args.upscale}")
    logger.info("Begin inference")

    # -------------------- Inference  --------------------
    for batch in tqdm(
        dataloader, desc="[bold cyan]Video SR[/bold cyan]", leave=True, dynamic_ncols=True
    ):
        lq = batch["lq_video"]
        name = batch["name"]
        model.inference(lq, save_path=Path(args.output_dir) / name[0])

    logger.info("All videos has been processed.")
