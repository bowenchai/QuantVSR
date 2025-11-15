# 🚀 QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution

[Bowen Chai](https://github.com/bowenchai), [Zheng Chen](https://zhengchen1999.github.io/), [Libo Zhu](https://github.com/libozhu03), [Wenbo Li](https://fenglinglwb.github.io/), [Yong Guo](https://www.guoyongcs.com/), and [Yulun Zhang](http://yulunzhang.com/)

"QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution", AAAI 2026

[![page](https://img.shields.io/badge/Project-Page-blue?logo=github)](https://bowenchai.github.io/QuantVSR/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://www.arxiv.org/abs/2508.04485)
[![supp](https://img.shields.io/badge/Supplementary_material-Paper-orange.svg)](https://github.com/bowenchai/QuantVSR/releases/download/v1/Supplementary_Material.pdf)
[![releases](https://img.shields.io/github/downloads/bowenchai/QuantVSR/total?color=green&style=flat)](https://github.com/bowenchai/QuantVSR/releases)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=bowenchai/QuantVSR)](https://github.com/bowenchai/QuantVSR)
[![GitHub Stars](https://img.shields.io/github/stars/bowenchai/QuantVSR?style=social)](https://github.com/bowenchai/QuantVSR/stargazers)

---

## 📚 Table of Contents

- [🔥 News](#-news)
- [📘 Abstract](#-abstract)
- [📝 Structure Overview](#-structure-overview)
- [⚙️ Installation](#️-installation)
- [📁 Datasets](#-datasets)
- [📥 Download Pretrained Models](#-download-pretrained-models)
- [📈 Calibration](#-calibration)
- [🧪 Inference](#-inference)
- [📦 Measure](#-measure)
- [🔎 Results](#-results)
- [📌 Citation](#-citation)
- [📝 Acknowledgements](#-acknowledgements)

---

## 🔥 News

- 🎉 **[2025-11-15]** All code and pretrained weights are released.
- 🏆 **[2025-11-08]** QuantVSR is accepted by AAAI 2026.
- 🚩 **[2025-08-06]** This repo is released.

## 📘 Abstract

> Diffusion models have shown superior performance in real-world video super-resolution (VSR). However, the slow processing speeds and heavy resource consumption of diffusion models hinder their practical application and deployment. Quantization offers a potential solution for compressing the VSR model. Nevertheless, quantizing VSR models is challenging due to their temporal characteristics and high fidelity requirements. To address these issues, we propose QuantVSR, an effective low-bit quantization model for real-world VSR. We propose a spatio-temporal complexity aware (STCA) mechanism, where we first utilize the calibration dataset to measure both spatial and temporal complexities for each layer. Based on these statistics, we allocate layer-specific ranks to the low-rank full-precision (FP) auxiliary branch. Subsequently, we jointly refine the FP and low-bit branches to achieve simultaneous optimization. In addition, we propose a learnable bias alignment (LBA) module to reduce the biased quantization errors. Extensive experiments on synthetic and real-world datasets demonstrate that our method obtains comparable performance with the FP model and significantly outperforms recent leading low-bit quantization methods.

<p align="center">
  <img src="figs/intro_visual.png" width="800px">
</p>

## 📝 Structure Overview

<p align="center">
  <img src="figs/overview.png" width="800px">
</p>

## ⚙️ Installation
We recommend using [Pixi](https://pixi.sh/latest/) as the virtual environment manager and task runner, which has advantages in speed, simplicity, and reproducibility.

```bash
git clone https://github.com/bowenchai/QuantVSR.git
cd QuantVSR
pixi install -a
eval "$(pixi shell-hook)"
```

You can also manually install the dependencies based on the `pyproject.toml` file.

## 📁 Datasets

All datasets follow a consistent directory structure:

```bash
data/
  ├── [DatasetName]/
  │   ├── GT/         # Ground Truth: folder of high-quality frames.
  │   └── LQ/         # Low-quality Input: folder of degraded frames.
  └── [DatasetName]/
      └── ...
```

> ❗ Notice: Please make sure the dataset is placed in the `./data/`, otherwise you will need to modify some paths in the code manually.

### 🗳️ Calibration Datasets

| Dataset |    Type    | # Num |                           Download                           |
| :------ | :--------: | :---: | :----------------------------------------------------------: |
| REDS30  | Synthetic  |  30   | [OneDrive](https://1drv.ms/u/c/11b213ff6bce81e0/EcG0Wq3K1KlKieRs2dFg1AsBnLghD271-1tnas-WCyevGA?e=t6l2SK) |

### 🗳️ Test Datasets

| Dataset |    Type    | # Num |                           Download                           |
| :------ | :--------: | :---: | :----------------------------------------------------------: |
| REDS4   | Synthetic  |  4    | [OneDrive](https://1drv.ms/u/c/11b213ff6bce81e0/ESLilp3EnqtPiRl4adOw6uwBlCO7cvS6FXGbA2vvo-tngw?e=9UnITK) |
| UDM10   | Synthetic  |  10   | [OneDrive](https://1drv.ms/u/c/11b213ff6bce81e0/EQszwlrTFNJLrpVA7zk-CroBaFgqI6frbOxe1G_wsbQSrg) |
| SPMCS   | Synthetic  |  30   | [OneDrive](https://1drv.ms/u/c/11b213ff6bce81e0/EXd9UOEcmdtLop6oCRbO6_QBkaX6qYonZhO3N_iOvsIWlA?e=sX5Kaj) |
| MVSR4x  | Real-world |  15   | [OneDrive](https://1drv.ms/u/c/11b213ff6bce81e0/ETMhrZWbAe9Ng6KUJ_gN-JUBRMQtUQLNF2yB9VlgFeEHww?e=VJqxvT) |

## 📥 Download Pretrained Models

| Model     | Information     | Link                                |
| :-------- | :-------------: | :---------------------------------: |
| MGLD-VSR  | Model weights of MGLD-VSR    |  [OneDrive](https://1drv.ms/u/c/11b213ff6bce81e0/EXfiMzc8yWlGmUEL1c7989cBJ9k6IJYy-RC8O7jLpCNTAQ?e=bNUPgw) |
| QuantVSR  | The calibrated model weights |  [OneDrive](https://1drv.ms/u/c/11b213ff6bce81e0/EdNuEGVZG9VFscoWocYBXPUB2tZsmW5fOFdGmI66iwERGA?e=VjREsC) |

You need to store the weights in the following structure:

```bash
weights/
  ├── mgldvsr/
  │   ├── mgldvsr_unet.ckpt
  │   ├── open_clip_pytorch_model.bin
  │   ├── raft-things.pth
  │   ├── spynet_sintel_final-3d2a1287.pth
  │   ├── v2-1_512-ema-pruned.ckpt
  │   └── video_vae_cfw.ckpt
  └── qparams/
      ├── w4a4.pth
      └── ...
```

## 📈 Calibration
> 💡 Tips: Pixi is for simplicity. All corresponding Python commands can be found in the tasks section of the `pyproject.toml` file.

```bash
# {bits} is an integer
# for example: `pixi run calibration 4` represents the calibration of 4-bit quantization.
pixi run calibration {bits}
```

## 🧪 Inference
If you want to obtain the results for the test dataset：

```bash
# {datasets} represents the datasets to test.
# {upscale} is the upscaling factor.
# for example: `pixi run inference SPMCS 4 4` represents performing 4x upscaling on the SPMCS dataset with 4-bit quantization."
pixi run inference {datasets} {upscale} {bits}
```

If you want to test custom data：
```bash
python scripts/inference.py \
  --input_dir  {directory_of_input_videos}
  --output_dir {directory_to_save_outputs}
  --upscale    {upscaling_factor}
  --bits       {quantization_bits}
```

## 📦 Measure
> ❗ Notice: The computing environments of Dover and $E^*_{warp}$ are different from the default environment, but Pixi can automatically switch between environments. For more details, please refer to the [official documentation](https://pixi.sh/latest/tutorials/multi_environment/).

```bash
# {pred_dir} is a directory containing the videos to be measured.
# {gt_dir} is the corresponding directory containing the ground-truth videos.
# {metrics} represents the metrics to be measured, separated by commas, like `psnr,ssim,lpips,dists,clipiqa,musiqa,niqe,maniqa`.
# for example: `pixi run eval_metrics ./experiments/w4a4/SPMCS ./data/SPMCS/GT psnr,ssim`
pixi run eval_metrics {pred_dir} {get_dir} {metrics}
pixi run eval_dover {pred_dir}
pixi run eval_ewarp {pred_dir}
```

## 🔎 Results

QuantVSR significantly outperforms previous methods at the setting of W6A6 and W4A4.

Evaluation on synthetic and real-world datasets

<details>
<summary>Quantitative Results (click to expand)</summary>

- Results in Tab. 3 of the main paper

<p align="center">
  <img width="900" src="figs/quantitative_results.png">
</p>
</details>

<details>
<summary>Qualitative Results (click to expand)</summary>

- Results in Fig. 4 of the main paper

<p align="center">
  <img width="900" src="figs/qualitative_results.png">
</p>

</details>

## 📌 Citation  

If you find the code helpful in your research or work, please cite the following paper(s).

```
@article{chai2025quantvsr,
  title={QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution},
  author={Bowen Chai, Zheng Chen, Libo Zhu, Wenbo Li, Yong Guo, and Yulun Zhang},
  journal={arXiv preprint arXiv:2508.04485},
  year={2025}
}
```

## 📝 Acknowledgements

We thank the developers of [MGLD-VSR](https://github.com/IanYeung/MGLD-VSR), whose method provides a strong baseline for QuantVSR.

We also thank the open-source contributors of [PassionSR](https://github.com/libozhu03/PassionSR) and [ViDiT-Q](https://github.com/thu-nics/ViDiT-Q); their excellent code has greatly facilitated the research and development of QuantVSR.
