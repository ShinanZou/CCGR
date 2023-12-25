# Cross-Covariate Gait Recognition: A Benchmark

Welcome to the official repository for the paper "Cross-Covariate Gait Recognition: A Benchmark," which has been accepted to AAAI 2024. In this work, we introduce a novel dataset and parsing-based gait recognition method. Below you'll find details about the paper, dataset, and accompanying resources.

## Paper Links
- **AAAI 2024 Version**: [Download paper (AAAI)](https://aaai.org/ojs/index.php/AAAI/article/view/XXXX)
- **ArXiv Preprint**: [Download preprint (ArXiv)](https://arxiv.org/pdf/2312.14404.pdf)

## Dataset Visualization
Here are some visualizations that depict the characteristics of the Cross-Covariate Gait Recognition (CCGR) dataset in comparison with other existing datasets, as well as the distribution of covariates, viewpoints, and data modalities within CCGR.

### CCGR Dataset vs. Other Datasets
![CCGR_vs_Others](imgs/F1.jpg)

### Covariates in the CCGR Dataset
![Covariates](imgs/F2.jpg)

### Views Distribution in the CCGR Dataset
![Viewpoints](imgs/F3.jpg)

### Data Modalities in the CCGR Dataset
![Modalities](imgs/F5.jpg)
#### We are one of the earliest teams to explore parsing-based gait recognition, and our work on parsing had been previously submitted to CVPR 2023.

## CCGR Dataset Download Guide
### Derived data (Silhouette, Parsing, Pose)
We are pleased to offer the derived data for use in your research projects. To ensure proper usage, we kindly ask that you adhere to the following terms:

1. **Research Purposes Only**: The dataset is strictly for academic and non-commercial research purposes only.
2. **No Modification or Redistribution**: Modifying the dataset or redistributing it in any form is not permitted.
3. **Internal Use**: The dataset is to be used within your organization or research group and is not to be shared outside of this group.

By downloading the dataset, you indicate your acceptance of these terms. We have facilitated easy access to the data via the links provided below, so no additional download request is necessary!

#### Dataset Links

You are invited to download the CCGR dataset from the links below. Before proceeding with the download, please ensure you have agreed to the above terms.

- **Baidu Netdisk**: [Download Dataset](https://pan.baidu.com/s/1GUTdGRLHyqSHw0Fcc7iUEQ) (Code:ngcw)
- **OneDrive**: [Download Dataset](https://1drv.ms/f/c/8464f220191191b1/Eov74XWuOi1Op_fdXDRzoAMBbJLrqSN1HoM4_WLNLUNm0Q?e=A8RQAJ)

We wish to support a collaborative and respectful research community and we believe that adherence to these terms will be mutually beneficial.

### Raw data (RGB)

We would like to inform you that the Raw data is scheduled to be released on March 1, 2024. Details regarding access and download instructions will be announced as the date approaches. Please stay tuned to our official communication channels for further updates.

Thank you for your attention and cooperation. We sincerely hope the dataset aids you in your research endeavors.

## Testing Code on OpenGait Platform
Specific adjustments are necessary to utilize the OpenGait framework testing the CCGR dataset effectively. For detailed instructions on how to adapt OpenGait, please refer to the guidelines provided at the beginning of the `CCGR_EVA.py` file. These modifications are designed to ensure seamless integration of the CCGR dataset with the existing capabilities of OpenGait. Additionally, we have included some yaml files within this document and some trained weights for silhouette. [Baidu Netdisk](https://pan.baidu.com/s/1BePp802Aimo5kNTzoO8-HQ)  (Code:4o92); [OneDrive](https://1drv.ms/f/c/8464f220191191b1/EghAuK8us5ROigkMkifohzMBG81qQQBBrVcLiGoPnCL_Tw?e=kBgiXG).
- [OpenGait Platform](https://github.com/ShiqiYu/OpenGait)

## Cite Us
If you find our dataset or paper useful for your research, please consider citing:

```bibtex
@inproceedings{Zou2024CCGR,
  title={Cross-Covariate Gait Recognition: A Benchmark},
  author={Zou, Shinan and Fan, Chao and Xiong, Jianbo and Shen, Chuanfu and Yu, Shiqi and Tang, Jin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
