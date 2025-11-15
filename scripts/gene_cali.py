from pathlib import Path
import torch
import argparse
import pyfiglet
from colorama import Fore, init
from tqdm.rich import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from quantvsr.models import get_model
from quantvsr.utils.logger import setup_logger
from quantvsr.data.sr_datasets import VSRDataset, custom_collate_fn

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
        "--save_path",
        type=str,
        default="./data/cali_data.pt",
        help="the path to save the calibration data",
    )
    parser.add_argument(
        "--sample_nums",
        type=int,
        default=600,
        help="the number of samples to generate for calibration",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/REDS30/LQ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    # ----------------- Preparation --------------------
    logger = setup_logger("quantvsr")
    seed_everything(args.seed)
    device = args.device

    # ------------------ Model --------------------
    ckpt_path = {
        "unet_ckpt_path": "weights/mgldvsr/mgldvsr_unet.ckpt",
        "vae_ckpt_path": "weights/mgldvsr/video_vae_cfw.ckpt",
    }
    model = get_model("mgldvsr", pretrained_path=ckpt_path, device=device)

    # -------------------- Data --------------------
    datasets = VSRDataset(
        args.input_dir, device=device, upscale=args.upscale, cali_resolution="5x512x512"
    )
    dataloader = DataLoader(datasets, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    # ----------------- Logging  --------------------
    init(autoreset=True)
    project_name = "QuantVSR"
    ascii_art = pyfiglet.figlet_format(project_name)
    print(Fore.CYAN + ascii_art)

    logger.info("Currently in generating calibration data process.")
    logger.info(f"Model: {type(model)}")
    logger.info(f"Calibration Video: {Path(args.input_dir).absolute()} with {len(datasets)} videos")
    logger.info(f"VSR upscale factor: {args.upscale}")
    quantized_model_dict = {k: type(v) for k, v in model.get_quantizable_modules().items()}
    logger.info(f"Quantized model dict: {quantized_model_dict}")

    # ----------- Generating Calibration Data -------------
    model.generate_calibration_data(
        dataloader, sample_num=args.sample_nums, save_path=args.save_path
    )
