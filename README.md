<div align="center">

# CCGR: Cross-Covariate Gait Recognition

**Official project page for the CCGR benchmark and CCGR-3D.**

[![AAAI 2024](https://img.shields.io/badge/AAAI-2024-2f6fbb)](https://ojs.aaai.org/index.php/AAAI/article/view/28621)
[![Knowledge-Based Systems 2026](https://img.shields.io/badge/KBS-2026-8a2be2)](https://doi.org/10.1016/j.knosys.2026.115576)
[![Dataset](https://img.shields.io/badge/Dataset-CCGR%20%7C%20CCGR--Mini%20%7C%20CCGR--3D-green)](#dataset-access)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND-orange)](#license)

</div>

## Overview

Cross-covariate gait recognition aims to identify people under large appearance and acquisition changes, including clothing, carrying conditions, body-shape variations, viewpoints, and other practical covariates.

This project brings together the **CCGR benchmark**, **CCGR-Mini**, and **CCGR-3D** for cross-covariate gait recognition.

| Component | Description |
|---|---|
| **CCGR** | A large-scale cross-covariate gait benchmark with dense view and covariate annotations. |
| **CCGR-Mini** | A compact subset of CCGR for faster research iteration while preserving covariate diversity. |
| **CCGR-3D** | An extension of CCGR for studying robust cross-covariate gait recognition with 3D data. |

## News

- **2026**: **3D Gallery for Cross-Covariate Gait Recognition** was published in *Knowledge-Based Systems*.
- **2024**: **Cross-Covariate Gait Recognition: A Benchmark** was accepted by **AAAI 2024**.

## Papers

| Paper | Venue | Links |
|---|---|---|
| **3D Gallery for Cross-Covariate Gait Recognition** | *Knowledge-Based Systems*, 2026 | [DOI](https://doi.org/10.1016/j.knosys.2026.115576) |
| **Cross-Covariate Gait Recognition: A Benchmark** | AAAI 2024 | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28621) / [arXiv](https://arxiv.org/pdf/2312.14404.pdf) |

## Dataset Access

The released data are provided for non-commercial research use. Please follow the dataset license and cite the corresponding papers when using CCGR, CCGR-Mini, or CCGR-3D.

### CCGR

CCGR contains **970 subjects**, about **1.6 million sequences**, **33 views**, and **53 covariates**. It provides multiple gait modalities, including **RGB**, **silhouette**, **parsing**, and **pose**.

| Data type | Access |
|---|---|
| Derived data: silhouette, parsing, pose | [Baidu Netdisk](https://pan.baidu.com/s/1GUTdGRLHyqSHw0Fcc7iUEQ) `ngcw` / [OneDrive](https://1drv.ms/f/c/8464f220191191b1/Eov74XWuOi1Op_fdXDRzoAMBbJLrqSN1HoM4_WLNLUNm0Q?e=A8RQAJ) |
| Raw RGB data | Please sign the [RGB data usage agreement](https://github.com/ShinanZou/CCGR/blob/CCGR-Benchmark/output/CCGR_Dataset_RGB_Data_Usage_Agreement.pdf) and send it to [chengyulong@csu.edu.cn](mailto:chengyulong@csu.edu.cn). |

### CCGR-Mini

CCGR-Mini is a lightweight subset of CCGR. It keeps **970 subjects**, **47,884 sequences**, **53 covariates**, and **33 views**, while retaining one randomly selected view for each covariate of each subject.

| Data type | Access |
|---|---|
| Derived data: silhouette, parsing, pose | [Baidu Netdisk](https://pan.baidu.com/s/1h6auGcxWFqeUAws0PvSH8g) `ei8e` / [OneDrive](https://1drv.ms/f/c/8464f220191191b1/Ev18lg3FHJZCoyF_6z91JUUBDgBX7EZN0WHJKJnDEIzbWA?e=xon8em) |
| Raw RGB data | [Baidu Netdisk](https://pan.baidu.com/s/1qHJxbbMamgEPwp8fd2sfkQ) `oyqf` / [OneDrive](https://1drv.ms/f/c/8464f220191191b1/EjH2ZXqcl0tNnETfxOh2wZYBpgVsqx-6cZ7x8CVEADIjLA?e=YpbMVI) |

### CCGR-3D

CCGR-3D extends CCGR for evaluating cross-covariate gait recognition with 3D data. It provides multiple processed modalities, including **silhouette**, **pose**, **parsing**, **SMPL**, and **skeleton**. When using CCGR-3D, please cite both the 2026 KBS paper and the original CCGR benchmark paper.

| Data type | Access |
|---|---|
| Processed data: silhouette, pose, parsing, SMPL, skeleton, point cloud | [Baidu Netdisk](https://pan.baidu.com/s/1nFYf6lXI_GvZ5tT7BScIhw?pwd=dci3) `dci3` |
| Raw RGB data | [Baidu Netdisk](https://pan.baidu.com/s/1NcOaS2Znz3WAYmwzrxp7zw?pwd=bd3d) `bd3d`<br>Please sign the [RGB data usage agreement](https://github.com/ShinanZou/CCGR/blob/CCGR-Benchmark/output/CCGR_Dataset_RGB_Data_Usage_Agreement.pdf) and send it to [chengyulong@csu.edu.cn](mailto:chengyulong@csu.edu.cn). |

## Benchmark Results

### CCGR

**Silhouette input (%)**

| Method | R1<sup>hard</sup> | R1<sup>easy</sup> | R5<sup>hard</sup> | R5<sup>easy</sup> |
|---|---:|---:|---:|---:|
| GaitSet | 25.3 | 35.3 | 46.7 | 58.9 |
| GaitPart | 22.6 | 32.7 | 42.9 | 55.5 |
| GaitGL | 23.1 | 35.2 | 39.9 | 54.1 |
| GaitBase | 31.3 | 43.8 | 51.3 | 64.4 |
| DeepGaitV2 | 42.5 | 55.2 | 63.2 | 75.2 |

**Parsing input (%)**

| Method | R1<sup>hard</sup> | R1<sup>easy</sup> | R5<sup>hard</sup> | R5<sup>easy</sup> |
|---|---:|---:|---:|---:|
| GaitSet | 31.6 | 42.8 | 54.8 | 67.0 |
| GaitPart | 29.0 | 40.9 | 51.5 | 64.5 |
| GaitGL | 28.4 | 42.1 | 46.6 | 61.4 |
| GaitBase | 48.1 | 62.0 | 67.7 | 79.6 |
| DeepGaitV2 | 58.8 | 71.8 | 77.0 | 87.0 |

### CCGR-Mini

**Silhouette input (%)**

| Method | R1 | mAP | mINP |
|---|---:|---:|---:|
| GaitSet | 13.77 | 15.39 | 5.75 |
| GaitPart | 8.02 | 10.12 | 3.52 |
| GaitGL | 17.51 | 18.12 | 6.85 |
| GaitBase | 26.99 | 24.89 | 9.72 |
| DeepGaitV2 | 39.37 | 36.01 | 16.77 |

**Parsing input (%)**

| Method | R1 | mAP | mINP |
|---|---:|---:|---:|
| GaitSet | 18.09 | 19.18 | 7.38 |
| GaitPart | 10.60 | 12.29 | 4.25 |
| GaitGL | 22.53 | 22.58 | 9.06 |
| GaitBase | 38.96 | 35.48 | 16.08 |
| DeepGaitV2 | 50.43 | 46.53 | 24.43 |

## Citation

If this project is useful for your research, please cite the following papers.

```bibtex
@article{zou2026_3dgallery,
  title   = {3D Gallery for Cross-Covariate Gait Recognition},
  author  = {Zou, Shinan and Long, Chengyu and Guo, Fan and Tang, Jin},
  journal = {Knowledge-Based Systems},
  volume  = {339},
  pages   = {115576},
  year    = {2026},
  doi     = {10.1016/j.knosys.2026.115576}
}
```

```bibtex
@article{zou2024_ccgr,
  title   = {Cross-Covariate Gait Recognition: A Benchmark},
  author  = {Zou, Shinan and Fan, Chao and Xiong, Jianbo and Shen, Chuanfu and Yu, Shiqi and Tang, Jin},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {38},
  number  = {7},
  pages   = {7855--7863},
  year    = {2024},
  doi     = {10.1609/aaai.v38i7.28621}
}
```

## License

The dataset is released for non-commercial research use under **CC BY-NC-ND** unless otherwise specified. Raw RGB data requires a signed usage agreement.

## Notes

- In the AAAI 2024 version of the CCGR paper, the positions of R5<sup>easy</sup> and R5<sup>hard</sup> in Table 2 were reversed. The results shown above use the corrected order.
- In Table 3 of the AAAI 2024 version, the batch size of GaitBase and DeepGaitV2 was recorded incorrectly. The corrected results are shown above. These corrections do not affect the conclusions of the paper.

## Contact

For questions about the dataset or paper, please contact [zoushinan@csu.edu.cn](mailto:zoushinan@csu.edu.cn).
