import argparse
import os
import yaml
import subprocess
import sys

def parse_opt_arguments(opt_config):
    """Parse YAML configuration file for opt arguments"""
    with open(opt_config, 'r') as f:
        opt_args = yaml.safe_load(f)
    
    return opt_args

def run_command(args):
    """Run the command with parsed arguments"""
    # 解析opt配置
    opt = parse_opt_arguments(args.opt)
    mgld_options = opt["mgld_options"]
    quant_config = opt["quantization_options"]
    
    # 构建命令
    command = [
        "python", "scripts/vsr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py",
        "--config", mgld_options.get("config", "configs/mgldvsr/mgldvsr_512_realbasicvsr_deg.yaml"),
        "--ckpt", mgld_options.get("ckpt", "weights/mgldvsr_unet.ckpt"),
        "--vqgan_ckpt", mgld_options.get("vqgan_ckpt", "weights/video_vae_cfw.ckpt"),
        "--seqs-path", os.path.join(mgld_options.get("seqs_path_base", "data"), args.data),
        "--outdir", os.path.join(mgld_options.get("outdir", "experiments"), args.data.split("_")[0], quant_config['Unet']['method'], f"w{quant_config['Unet']['weight_quant_bits']}a{quant_config['Unet']['act_quant_bits']}"),
        "--ddpm_steps", str(mgld_options.get("ddpm_steps", 50)),
        "--dec_w", str(mgld_options.get("dec_w", 1.0)),
        "--colorfix_type", mgld_options.get("colorfix_type", "adain"),
        "--select_idx", str(args.select_idx),
        "--n_gpus", str(args.n_gpus),
        "--upscale", str(args.upscale),
        "--qvsr_options", args.opt
    ]
    
    print("Running command:")
    print(" ".join(command))
    
    # 运行命令
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run PTQ quantization script with configuration from YAML file")
    parser.add_argument("--opt", type=str, required=True,
                       help="Path to YAML configuration file")
    parser.add_argument("--data", type=str, required=True,
                       help="Dataset Name")
    parser.add_argument(
        "--n_gpus",
        type=int,
        default="1",
        help="Total GPU number.",
    )
    parser.add_argument(
        "--select_idx",
        type=int,
        default="0",
        help="Which GPU is selected.",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default="4",
        help="Scale factor",
    )
    
    args = parser.parse_args()
    
    # 检查opt配置文件是否存在
    import os
    if not os.path.exists(args.opt):
        print(f"Error: opt configuration file '{args.opt}' does not exist")
        sys.exit(1)
    
    run_command(args)

if __name__ == "__main__":
    main()
