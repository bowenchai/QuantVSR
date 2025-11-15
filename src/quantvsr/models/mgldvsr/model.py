import os
from pathlib import Path
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf
from typing import Dict, Any, Optional
from tqdm.rich import tqdm
from einops import rearrange, repeat
from PIL import Image
from ..base import BaseVSRModel
from ..registry import register_model
from ...utils.logger import get_logger

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.append(base_path)

from .ldm.util import instantiate_from_config
from .ldm.models.diffusion.ddpm import LatentDiffusionVSRTextWT
from .ldm.models.autoencoder import VideoAutoencoderKLResi
from .scripts.util_flow import forward_backward_consistency_check
from .scripts.wavelet_color_fix import adaptive_instance_normalization, wavelet_reconstruction
from .scripts.util_image import ImageSpliterTh


@register_model("mgldvsr")
class MGLDVSRModel(BaseVSRModel):
    def __init__(self, pretrained_path: Optional[Dict[str, str]] = None, device="cpu", **kwargs):
        """
        Args:
            config_path: Path of mgldvsr config file. If None, uses default config.
        """
        self.logger = get_logger("quantvsr")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        diffusion_model_config_path = os.path.join(
            current_dir, "configs/mgldvsr/mgldvsr_512_realbasicvsr_deg.yaml"
        )
        vae_config_path = os.path.join(
            current_dir, "configs/video_vae/video_autoencoder_kl_64x64x4_resi.yaml"
        )

        config = {
            "diffusion_model_config": OmegaConf.load(f"{diffusion_model_config_path}"),
            "vae_config": OmegaConf.load(f"{vae_config_path}"),
        }
        super().__init__(config)

        self.colorfix_type = "adain"

        self.diffusion_model = instantiate_from_config(config["diffusion_model_config"].model)
        self.diffusion_model.configs = config["diffusion_model_config"]
        self.vae = instantiate_from_config(config["vae_config"].model)
        self.logger.debug(f"Instantiate Diffusion Model: {type(self.diffusion_model)}")
        self.logger.debug(f"Instantiate VAE Model: {type(self.vae)}")

        if pretrained_path:
            self.load_pretrained(pretrained_path)
        self.diffusion_model.to(device)
        self.vae.to(device)
        self.device = device

        self.diffusion_model.register_schedule(
            given_betas=None,
            beta_schedule="linear",
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.0120,
            cosine_s=8e-3,
        )
        self.diffusion_model.num_timesteps = 1000

        self.sqrt_alphas_cumprod = copy.deepcopy(self.diffusion_model.sqrt_alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = copy.deepcopy(
            self.diffusion_model.sqrt_one_minus_alphas_cumprod
        )

        use_timesteps = set(space_timesteps(1000, [50]))
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(self.diffusion_model.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        new_betas = [beta.data.cpu().numpy() for beta in new_betas]
        self.diffusion_model.register_schedule(
            given_betas=np.array(new_betas), timesteps=len(new_betas)
        )
        self.diffusion_model.num_timesteps = 1000
        self.diffusion_model.ori_timesteps = list(use_timesteps)
        self.diffusion_model.ori_timesteps.sort()
        self.diffusion_model = self.diffusion_model.to(device)
        self.diffusion_model.cond_stage_model.device = device

        self.n_frames = 5
        self.n_samples = 1
        self.vqgantile_size = 960
        self.vqgantile_stride = 750
        self.ddpm_steps = 50
        self.tile_overlap = 32

    def load_pretrained(self, ckpt_paths: Dict[str, str], **kwargs):
        self.diffusion_model: LatentDiffusionVSRTextWT
        self.vae: VideoAutoencoderKLResi
        self.logger.debug(
            f"Loading pretrained weights from {ckpt_paths.get('unet_ckpt_path')} and {ckpt_paths.get('vae_ckpt_path')}"
        )
        unet_sd = torch.load(
            ckpt_paths.get("unet_ckpt_path", ""), map_location="cpu", weights_only=False
        )
        vae_sd = torch.load(
            ckpt_paths.get("vae_ckpt_path", ""), map_location="cpu", weights_only=False
        )
        m, u = self.diffusion_model.load_state_dict(unet_sd["state_dict"], strict=False)
        self.logger.debug(f"Loaded UNet weights: {len(m)} missing, {len(u)} unexpected")
        self.diffusion_model.eval()
        m, u = self.vae.load_state_dict(vae_sd["state_dict"], strict=False)
        self.logger.debug(f"Loaded VAE weights: {len(m)} missing, {len(u)} unexpected")
        self.vae.eval()

    def inference(
        self, lq_frames: torch.Tensor, save_path: Optional[str | Path] = None, **kwargs
    ) -> torch.Tensor:
        """Inference interface. (Copy from MGLD-VSR)
        Args:
            lq_frames: [B, T, H, W, C] (-1, 1)
        """
        B, T, H, W, C = lq_frames.shape
        assert B == self.n_samples, (
            "The MGLD‑VSR inference process currently only supports `batch_size=1`."
        )
        self.diffusion_model: LatentDiffusionVSRTextWT
        self.vae: VideoAutoencoderKLResi

        # Pad lq_frames to multiple of self.n_frames by repeating the last frame
        if T % self.n_frames != 0:
            num_pad_frames = self.n_frames - (T % self.n_frames)
            last_frame = lq_frames[:, -1:, :, :, :]
            pad_frames = last_frame.repeat(1, num_pad_frames, 1, 1, 1)
            lq_frames = torch.cat([lq_frames, pad_frames], dim=1)

        num_segments = lq_frames.shape[1] // self.n_frames
        lq_frames = lq_frames.reshape(B * num_segments, self.n_frames, H, W, C)
        lq_frames = lq_frames.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, C, H, W)
        self.logger.debug(f"lq_frames shape: {lq_frames.shape}")

        with torch.no_grad(), torch.autocast(self.device), self.diffusion_model.ema_scope():
            sr_save_index = 0
            for frames_segment in tqdm(
                lq_frames,
                desc=" " * 2 + "[bold green]Temporal chunks[/bold green]",
                leave=False,
                dynamic_ncols=True,
            ):
                frames_segment = frames_segment.to(self.device)
                ori_h, ori_w = frames_segment.shape[2:]
                if not (ori_h % 32 == 0 and ori_w % 32 == 0):
                    flag_pad = True
                    pad_h = ((ori_h // 32) + 1) * 32 - ori_h
                    pad_w = ((ori_w // 32) + 1) * 32 - ori_w
                    frames_segment = F.pad(frames_segment, pad=(0, pad_w, 0, pad_h), mode="reflect")
                else:
                    flag_pad = False

                im_lq_bs_0_1 = torch.clamp((frames_segment + 1.0) / 2.0, min=0.0, max=1.0)
                _, _, im_h, im_w = im_lq_bs_0_1.shape

                # flow estimation
                im_lq_bs_0_1 = F.interpolate(
                    im_lq_bs_0_1, size=(im_h // 4, im_w // 4), mode="bicubic"
                )
                im_lq_bs_0_1 = rearrange(
                    im_lq_bs_0_1, "(b t) c h w -> b t c h w", t=frames_segment.size(0)
                )
                flows = self.diffusion_model.compute_flow(im_lq_bs_0_1)
                flows = [rearrange(flow, "b t c h w -> (b t) c h w") for flow in flows]
                flows = [
                    resize_flow(flow, size_type="shape", sizes=(im_h // 8, im_w // 8))
                    for flow in flows
                ]
                flows = [
                    rearrange(flow, "(b t) c h w -> b t c h w", t=frames_segment.size(0) - 1)
                    for flow in flows
                ]

                # occlusion mask estimation
                fwd_occ_list, bwd_occ_list = list(), list()
                for i in range(frames_segment.size(0) - 1):
                    fwd_flow, bwd_flow = flows[1][:, i, :, :, :], flows[0][:, i, :, :, :]
                    fwd_occ, bwd_occ = forward_backward_consistency_check(
                        fwd_flow, bwd_flow, alpha=0.01, beta=0.5
                    )
                    fwd_occ_list.append(fwd_occ.unsqueeze_(1))
                    bwd_occ_list.append(bwd_occ.unsqueeze_(1))
                fwd_occs = torch.stack(fwd_occ_list, dim=1)
                fwd_occs = rearrange(fwd_occs, "b t c h w -> (b t) c h w")
                bwd_occs = torch.stack(bwd_occ_list, dim=1)
                bwd_occs = rearrange(bwd_occs, "b t c h w -> (b t) c h w")
                # masks = [fwd_occ_list, bwd_occ_list]

                flows = [rearrange(flow, "b t c h w -> (b t) c h w") for flow in flows]

                if (
                    frames_segment.shape[2] > self.vqgantile_size
                    or frames_segment.shape[3] > self.vqgantile_size
                ):
                    imlq_spliter = ImageSpliterTh(
                        frames_segment, self.vqgantile_size, self.vqgantile_stride, sf=1
                    )
                    flow_spliter_f = ImageSpliterTh(
                        flows[0], self.vqgantile_size // 8, self.vqgantile_stride // 8, sf=1
                    )
                    flow_spliter_b = ImageSpliterTh(
                        flows[1], self.vqgantile_size // 8, self.vqgantile_stride // 8, sf=1
                    )
                    fwd_occ_spliter = ImageSpliterTh(
                        fwd_occs, self.vqgantile_size // 8, self.vqgantile_stride // 8, sf=1
                    )
                    bwd_occ_spliter = ImageSpliterTh(
                        bwd_occs, self.vqgantile_size // 8, self.vqgantile_stride // 8, sf=1
                    )
                    for (
                        (im_lq_pch, index_infos),
                        (flow_f, _),
                        (flow_b, _),
                        (fwd_occ, _),
                        (bwd_occ, _),
                    ) in tqdm(
                        zip(
                            imlq_spliter,
                            flow_spliter_f,
                            flow_spliter_b,
                            fwd_occ_spliter,
                            bwd_occ_spliter,
                        ),
                        desc=" " * 4 + "[bold yellow]Spatial chunks[/bold yellow]",
                        total=len(imlq_spliter),
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        init_latent = self.diffusion_model.get_first_stage_encoding(
                            self.diffusion_model.encode_first_stage(im_lq_pch)
                        )
                        text_init = [""] * self.n_samples
                        semantic_c = self.diffusion_model.cond_stage_model(text_init)
                        noise = torch.randn_like(init_latent)
                        # If you would like to start from the intermediate steps,
                        # you can add noise to LR to the specific steps.
                        t = repeat(torch.tensor([999]), "1 -> b", b=frames_segment.size(0))
                        t = t.to(self.device).long()
                        x_T = self.diffusion_model.q_sample_respace(
                            x_start=init_latent,
                            t=t,
                            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                            noise=noise,
                        )
                        # x_T = noise
                        # im_lq_pch_0_1 = torch.clamp((im_lq_pch + 1.0) / 2.0, min=0.0, max=1.0)
                        flow_f = rearrange(
                            flow_f, "(b t) c h w -> b t c h w", t=frames_segment.size(0) - 1
                        )
                        flow_b = rearrange(
                            flow_b, "(b t) c h w -> b t c h w", t=frames_segment.size(0) - 1
                        )
                        fwd_occ = rearrange(
                            fwd_occ, "(b t) c h w -> b t c h w", t=frames_segment.size(0) - 1
                        )
                        bwd_occ = rearrange(
                            bwd_occ, "(b t) c h w -> b t c h w", t=frames_segment.size(0) - 1
                        )
                        flows = (flow_f, flow_b)
                        masks = (fwd_occ, bwd_occ)
                        samples, _ = self.diffusion_model.sample_canvas(
                            cond=semantic_c,
                            struct_cond=init_latent,
                            guidance_scale=-10.0,
                            lr_images=None,
                            flows=flows,
                            masks=masks,
                            cond_flow=None,
                            batch_size=im_lq_pch.size(0),
                            timesteps=self.ddpm_steps,
                            time_replace=self.ddpm_steps,
                            x_T=x_T,
                            return_intermediates=True,
                            tile_size=64,
                            tile_overlap=self.tile_overlap,
                            batch_size_sample=self.n_samples,
                        )
                        del (
                            init_latent,
                            noise,
                            t,
                            flows,
                            masks,
                            flow_f,
                            flow_b,
                            fwd_occ,
                            bwd_occ,
                            x_T,
                            semantic_c,
                        )
                        torch.cuda.empty_cache()

                        _, enc_fea_lq = self.vae.encode(im_lq_pch)
                        x_samples = self.vae.decode(
                            samples * 1.0 / self.diffusion_model.scale_factor, enc_fea_lq
                        )
                        # x_samples = model.decode_first_stage(samples)

                        if self.colorfix_type == "adain":
                            x_samples = adaptive_instance_normalization(x_samples, im_lq_pch)
                        elif self.colorfix_type == "wavelet":
                            x_samples = wavelet_reconstruction(x_samples, im_lq_pch)
                        imlq_spliter.update(x_samples.cpu(), index_infos)
                        del x_samples, samples, enc_fea_lq
                        torch.cuda.empty_cache()

                    im_sr = imlq_spliter.gather()
                    im_sr = torch.clamp((im_sr + 1.0) / 2.0, min=0.0, max=1.0)
                else:
                    with tqdm(
                        total=1,
                        desc=" " * 4 + "[bold yellow]Spatial chunks[/bold yellow]",
                        leave=False,
                        dynamic_ncols=True,
                    ) as pbar:
                        init_latent = self.diffusion_model.get_first_stage_encoding(
                            self.diffusion_model.encode_first_stage(frames_segment)
                        )
                        text_init = [""] * self.n_samples
                        semantic_c = self.diffusion_model.cond_stage_model(text_init)
                        noise = torch.randn_like(init_latent)
                        # If you would like to start from the intermediate steps,
                        # you can add noise to LR to the specific steps.
                        t = repeat(torch.tensor([999]), "1 -> b", b=frames_segment.size(0))
                        t = t.to(self.device).long()
                        x_T = self.diffusion_model.q_sample_respace(
                            x_start=init_latent,
                            t=t,
                            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                            noise=noise,
                        )
                        # x_T = noise
                        flow_f, flow_b = flows[0], flows[1]
                        flow_f = rearrange(
                            flow_f, "(b t) c h w -> b t c h w", t=frames_segment.size(0) - 1
                        )
                        flow_b = rearrange(
                            flow_b, "(b t) c h w -> b t c h w", t=frames_segment.size(0) - 1
                        )
                        fwd_occ = rearrange(
                            fwd_occs, "(b t) c h w -> b t c h w", t=frames_segment.size(0) - 1
                        )
                        bwd_occ = rearrange(
                            bwd_occs, "(b t) c h w -> b t c h w", t=frames_segment.size(0) - 1
                        )
                        flows = (flow_f, flow_b)
                        masks = (fwd_occ, bwd_occ)
                        samples, _ = self.diffusion_model.sample_canvas(
                            cond=semantic_c,
                            struct_cond=init_latent,
                            guidance_scale=-10.0,
                            lr_images=None,
                            flows=flows,
                            masks=masks,
                            cond_flow=None,
                            batch_size=frames_segment.size(0),
                            timesteps=self.ddpm_steps,
                            time_replace=self.ddpm_steps,
                            x_T=x_T,
                            return_intermediates=True,
                            tile_size=64,
                            tile_overlap=self.tile_overlap,
                            batch_size_sample=self.n_samples,
                        )
                        _, enc_fea_lq = self.vae.encode(frames_segment)
                        x_samples = self.vae.decode(
                            samples * 1.0 / self.diffusion_model.scale_factor, enc_fea_lq
                        )
                        # x_samples = model.decode_first_stage(samples)

                        if self.colorfix_type == "adain":
                            x_samples = adaptive_instance_normalization(x_samples, frames_segment)
                        elif self.colorfix_type == "wavelet":
                            x_samples = wavelet_reconstruction(x_samples, frames_segment)
                        im_sr = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        pbar.update(1)

                im_sr = im_sr.cpu().numpy().transpose(0, 2, 3, 1) * 255  # (T, H, W, C)
                if flag_pad:
                    im_sr = im_sr[:, :ori_h, :ori_w]
                self.logger.debug(f"im_sr shape {im_sr.shape}")

                if save_path:
                    save_path = Path(save_path)
                    save_path.mkdir(exist_ok=True, parents=True)
                    for frame in im_sr:
                        if sr_save_index < T:
                            Image.fromarray(frame.astype(np.uint8)).save(
                                save_path / f"{sr_save_index:04d}.png"
                            )
                        sr_save_index += 1

        return lq_frames

    def get_quantizable_modules(self) -> Dict[str, nn.Module]:
        assert isinstance(self.diffusion_model, LatentDiffusionVSRTextWT), (
            "Wrong class type of diffusion model."
        )
        return {
            "unet": self.diffusion_model.model.diffusion_model,  # UNet
        }

    def generate_calibration_data(
        self, dataloader: DataLoader, sample_num: int = 600, save_path: str = "./data/cali_data.pt"
    ):
        self.diffusion_model: LatentDiffusionVSRTextWT
        self.diffusion_model.CALI = True
        pbar = tqdm(
            total=sample_num,
            desc="[bold cyan]Generating Calibration Data[/bold cyan]",
            leave=True,
            dynamic_ncols=True,
        )
        while len(self.diffusion_model.cali_datas) < sample_num:
            for batch in dataloader:
                if len(self.diffusion_model.cali_datas) >= sample_num:
                    break
                lq = batch["lq_video"]
                self.inference(lq, save_path=None)
                pbar.n = len(self.diffusion_model.cali_datas)
                pbar.refresh()
        pbar.close()
        if save_path:
            self.logger.info(f"Saving calibration data in {save_path}")
            torch.save(self.diffusion_model.cali_datas[:sample_num], save_path)
        self.diffusion_model.CALI = False

    def forward(self, *args, **kwargs):
        return self.inference(*args, **kwargs)


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
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]  # [250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
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


def resize_flow(flow, size_type, sizes, interp_mode="bilinear", align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == "ratio":
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == "shape":
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f"Size type should be ratio or shape, but got type {size_type}.")

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners
    )
    return resized_flow
