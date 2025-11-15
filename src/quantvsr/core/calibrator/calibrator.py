import wandb
import random
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.rich import tqdm

from ..config import CalibratorConfig
from ..observer import STCAObserver
from ..model import QuantVSRModel
from ..layer import BaseQuantLayer
from ...models import get_model
from ...data import VSRDataset, custom_collate_fn
from ...utils.logger import get_logger
from ...utils import recursive_to


class Calibrator:
    def __init__(self, config: CalibratorConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.logger = get_logger("quantvsr")
        ckpt_path = {
            "unet_ckpt_path": "weights/mgldvsr/mgldvsr_unet.ckpt",
            "vae_ckpt_path": "weights/mgldvsr/video_vae_cfw.ckpt",
        }
        self.fp_model = get_model(
            self.config.fp_model_name, pretrained_path=ckpt_path, device=device
        )

        # data
        if Path(self.config.cali_data_path).is_file():
            self.logger.info(f"Use pre-sampled cali_data {self.config.cali_data_path}")
            data_pt_path = self.config.cali_data_path
        else:
            data_pt_path = "./data/cali_data.pt"
            datasets = VSRDataset(
                self.config.cali_data_path,
                device=device,
                upscale=self.config.upscale,
                cali_resolution="5x512x512",
            )
            dataloader = DataLoader(
                datasets, batch_size=1, shuffle=True, collate_fn=custom_collate_fn
            )
            self.fp_model.generate_calibration_data(
                dataloader, self.config.cali_sample_nums, data_pt_path
            )

        self.cali_dataset = torch.load(data_pt_path)

        # quantization
        unet = self.fp_model.get_quantizable_modules()["unet"]
        self.quantized_model = QuantVSRModel(unet, self.config.quant_model_config)
        self.observer = STCAObserver(self.quantized_model)

    def launch_task(self):
        self.logger.info("Launching calibration task...")
        if self.config.report_to == "wandb":
            wandb.init(project="QuantVSR", name="Calibration", config=self.config.to_dict())
        self.LSRA_process()
        self.DBF_process()
        self.LBA_process()
        self.quantized_model.save_qparams(self.config.qparam_save_path)
        self.logger.info("Calibration completed.")

    @torch.no_grad()
    def LSRA_process(self):
        """Stage one: Layer-Specific Rank Allocation."""
        self.logger.info("Stage one: Layer-Specific Rank Allocation.")
        self.quantized_model.set_precision("fp")
        self.observer.register_hooks()

        for elem in tqdm(
            self.cali_dataset,
            desc="[bold cyan]Calibration stage one[/bold cyan]",
            leave=True,
            dynamic_ncols=True,
        ):
            input, fp_output = elem["input"], elem["fp_output"]
            input, fp_output = recursive_to((input, fp_output), self.device)
            _ = self.quantized_model(**input)
        self.observer.aggregate_statistics()
        stca_config = self.config.quant_model_config.layer_config.components.stca
        self._rank_allocation(stca_config.percentile, stca_config.min_rank, stca_config.max_rank)

        self.observer.reset()
        self.observer.remove_hooks()
        self.quantized_model.set_precision("low_bit")

    def DBF_process(self):
        """Stage two: Dual-Branch Refinement."""
        self.logger.info("Stage two: Dual-Branch Refinement.")
        trainable_params = [
            param
            for name, param in self.quantized_model.named_parameters()
            if "stca" in name and ("L1" in name or "L2" in name)
        ]
        self._train(trainable_params, desc="Calibration stage two")

    def LBA_process(self):
        self.logger.info("Stage three: Learnable Bias Alignment.")
        trainable_params = [
            param for name, param in self.quantized_model.named_parameters() if "lba.lba" in name
        ]
        self._train(
            trainable_params,
            iter_begin=self.config.num_iterations,
            desc="Calibration stage three",
        )

    def _train(self, trainable_params, iter_begin=0, desc=""):
        train_param_ids = {id(p) for p in trainable_params}
        for p in self.quantized_model.parameters():
            p.requires_grad = id(p) in train_param_ids
        self.logger.info(
            f"Trainable parameters: {sum(p.numel() for p in self.quantized_model.parameters() if p.requires_grad):,}"
        )

        loss_fn = F.mse_loss
        optimizer = AdamW(trainable_params, lr=self.config.learning_rate)
        scheduler = StepLR(optimizer, step_size=self.config.num_iterations // 2, gamma=0.2)

        pbar = tqdm(
            total=self.config.num_iterations,
            desc=f"[bold cyan]{desc}[/bold cyan]",
            dynamic_ncols=True,
        )

        iter_idx = 0
        accumulation_count = 0
        optimizer.zero_grad(set_to_none=True)
        while iter_idx < self.config.num_iterations:
            random.shuffle(self.cali_dataset)
            for elem in self.cali_dataset:
                input, fp_output = elem["input"], elem["fp_output"]
                input, fp_output = recursive_to((input, fp_output), self.device)

                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                    q_output = self.quantized_model(**input)
                    loss = loss_fn(q_output, fp_output) / self.config.gradient_accumulation

                loss.backward()

                accumulation_count += 1
                if accumulation_count % self.config.gradient_accumulation == 0:
                    if self.config.report_to == "wandb":
                        wandb.log(
                            {
                                "Mse Loss": loss.item() * self.config.gradient_accumulation,
                                "lr": optimizer.param_groups[0]["lr"],
                            },
                            step=iter_begin + iter_idx,
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    pbar.update()
                    iter_idx += 1
                    if iter_idx >= self.config.num_iterations:
                        break

    def _rank_allocation(
        self,
        percentile: float = 0.25,
        min_rank: int = 16,
        max_rank: int = 64,
    ):
        all_c_spatial, all_c_temporal = [], []

        for name, module in self.quantized_model.named_modules():
            if isinstance(module, BaseQuantLayer):
                stat = self.observer.statistics.get(name, {})
                all_c_spatial.extend(stat.get("c_spatial", []))
                all_c_temporal.extend(stat.get("c_temporal", []))

        all_c_spatial = torch.stack(all_c_spatial)
        all_c_temporal = torch.stack(all_c_temporal)

        l_spatial = torch.quantile(all_c_spatial, percentile)
        u_spatial = torch.quantile(all_c_spatial, 1 - percentile)
        l_temporal = torch.quantile(all_c_temporal, percentile)
        u_temporal = torch.quantile(all_c_temporal, 1 - percentile)

        for name, module in self.quantized_model.named_modules():
            if not isinstance(module, BaseQuantLayer):
                continue
            if name not in self.observer.statistics:
                continue

            c_spatial = self.observer.statistics[name]["c_spatial"]
            c_temporal = self.observer.statistics[name]["c_temporal"]
            assert len(c_spatial) == len(c_temporal)

            rank = min_rank
            for c_sp, c_tp in zip(c_spatial, c_temporal):
                if (c_sp <= l_spatial) and (c_tp <= l_temporal):
                    rank -= 1
                elif (c_sp >= u_spatial) and (c_tp >= u_temporal):
                    rank += 1

            rank = int(np.clip(rank, min_rank, max_rank))
            module.stca.rank = ((rank + 4) // 8) * 8
