import argparse
from pytorch_lightning import seed_everything
import pyfiglet
from colorama import Fore, init

from quantvsr.core.calibrator import Calibrator
from quantvsr.utils.logger import setup_logger
from quantvsr.core.config import (
    CalibratorConfig,
    QuantModelConfig,
    QuantLayerConfig,
    WeightQuantConfig,
    ActQuantConfig,
    ComponentConfig,
    STCAConfig,
    RotationConfig,
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
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="the device to use (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    # -------------------- Config --------------------
    config = CalibratorConfig(
        fp_model_name="mgldvsr",
        quant_model_config=QuantModelConfig(
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
        ),
        qparam_save_path=f"./weights/qparams/w{args.bits}a{args.bits}.pth",
        # data config
        cali_data_path="./data/REDS30/LQ",  # or "./data/cali_data.pt"
        cali_sample_nums=1800,
        upscale=4,
        # train settings
        num_iterations=900,
        learning_rate=1e-3,
        gradient_accumulation=4,
        max_grad_norm=1.0,
        report_to="wandb",
    )
    # Or just use the default config:
    # config = CalibratorConfig()

    # -------------------- Preparation  --------------------
    init(autoreset=True)
    project_name = "QuantVSR"
    ascii_art = pyfiglet.figlet_format(project_name)
    print(Fore.CYAN + ascii_art)

    logger = setup_logger("quantvsr")
    seed_everything(42)
    device = "cuda"
    logger.info("Currently in calibration process.")
    logger.info(f"Model: {config.fp_model_name}")
    logger.info(f"VSR upscale factor: {config.upscale}")
    logger.info("Begin calibration")

    # -------------------- Calibration  --------------------
    calibrator = Calibrator(config, device)
    calibrator.launch_task()
