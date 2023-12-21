# Cross-Covariate Gait Recognition: A Benchmark

Welcome to the official repository for the paper "Cross-Covariate Gait Recognition: A Benchmark," which has been accepted to AAAI 2024. In this work, we introduce a novel dataset and approach to gait recognition under various covariate conditions. Below you'll find details about the paper, dataset, and accompanying resources.

## Paper Links
- **AAAI 2024 Version**: [Download paper (AAAI)](https://aaai.org/ojs/index.php/AAAI/article/view/XXXX)
- **ArXiv Preprint**: [Download preprint (ArXiv)](https://arxiv.org/abs/XXXX.XXXXX)

## Dataset Visualization
Here are some visualizations that depict the characteristics of the Cross-Covariate Gait Recognition (CCGR) dataset in comparison with other existing datasets, as well as the distribution of covariates, viewpoints, and data modalities within CCGR.

### CCGR Dataset vs. Other Datasets
![CCGR_vs_Others](imgs/F1.jpg)

### Covariates in the CCGR Dataset
![Covariates](imgs/F2.jpg)

### Viewpoint Distribution in the CCGR Dataset
![Viewpoints](imgs/F3.jpg)

### Data Modalities in the CCGR Dataset
![Modalities](imgs/F5.jpg)

## Dataset Download Links
If you are interested in using the derived data from the CCGR dataset, such as parsing, silhouettes, and poses, please ensure that you agree to and comply with the following conditions:

1. The data may only be used for research purposes.
2. If you share the dataset with other researchers, you must convey the first condition regarding its use explicitly.

If you agree to the above terms, there is no need to send us any special download request! You can directly access the data via the links provided below:
You can download the CCGR dataset from the following links. Please note that you might require access permissions or to adhere to specific usage agreements.

- **Baidu Netdisk**: [Download Dataset (Baidu)](https://pan.baidu.com/s/XXXX)
- **Google Drive**: [Download Dataset (Google)](https://drive.google.com/drive/folders/XXXX)

We encourage open and responsible use of the dataset for research, and we appreciate your cooperation and understanding.

## Testing Code on OpenGait Platform
Specific adjustments are necessary to utilize the OpenGait framework for the CCGR dataset effectively. For detailed instructions on how to adapt OpenGait, please refer to the guidelines provided at the beginning of the `CCGR_EVA.py` file. These modifications are designed to ensure seamless integration of the CCGR dataset with the existing capabilities of OpenGait. Additionally, we have included the CCGR yaml file within this document and [some trained weights] (https://pan.baidu.com/s/XXXX).

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