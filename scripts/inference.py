import argparse
from pathlib import Path
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
import pyfiglet
from colorama import Fore, init


from quantvsr.data.sr_datasets import VSRDataset, custom_collate_fn, VSRValDataset
from quantvsr.core.model.quantvsr import QuantVSRModel
from quantvsr.models import get_model
from quantvsr.utils.logger import setup_logger
from quantvsr.core.config import (
    QuantModelConfig,
    QuantLayerConfig,
    WeightQuantConfig,
    ActQuantConfig,
    ComponentConfig,
    RotationConfig,
    STCAConfig,
    LBAConfig,
)

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
        default="./experiments/quant/SPMCS",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    # -------------------- Preparation --------------------
    logger = setup_logger("quantvsr")
    seed_everything(42)
    device = "cuda"

    # -------------------- Model --------------------

    config = QuantModelConfig(
        method_name="QuantVSR",
        layer_types=["Conv1d", "Conv2d", "Conv3d", "Linear"],
        layer_config=QuantLayerConfig(
            channel_split=True,
            weight_quant=WeightQuantConfig(
                bits=args.bits,
                symmetric=False,
                dynamic=False,
                granularity="channel",
            ),
            act_quant=ActQuantConfig(
                bits=args.bits,
                symmetric=False,
                dynamic=True,
                granularity="tensor",
            ),
            components=ComponentConfig(
                stca=STCAConfig(
                    enabled=True,
                    min_rank=16,
                    max_rank=64,
                    percentile=0.25,
                ),
                rotation=RotationConfig(enabled=True),
                lba=LBAConfig(enabled=True),
            ),
        ),
        exclude_layers=["time_embed.0", "input_blocks.0.0", "out.2"],
    )

    ckpt_path = {
        "unet_ckpt_path": "weights/mgldvsr/mgldvsr_unet.ckpt",
        "vae_ckpt_path": "weights/mgldvsr/video_vae_cfw.ckpt",
    }
    model = get_model("mgldvsr", pretrained_path=ckpt_path, device=device)
    unet = model.get_quantizable_modules()["unet"]

    unet = QuantVSRModel(unet, config)
    unet.load_qparams(f"./weights/qparams/w{args.bits}a{args.bits}.pth")
    unet.to(device)

    # -------------------- Data --------------------
    datasets = VSRValDataset(args.input_dir, device=device, upscale=args.upscale)
    dataloader = DataLoader(datasets, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # -------------------- Logging  --------------------
    init(autoreset=True)
    project_name = "QuantVSR"
    ascii_art = pyfiglet.figlet_format(project_name)
    print(Fore.CYAN + ascii_art)

    logger.info("Currently in quantized inference")
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
