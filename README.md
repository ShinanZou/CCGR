# Cross-Covariate Gait Recognition: A Benchmark

Welcome to the official repository for the paper "Cross-Covariate Gait Recognition: A Benchmark," which has been accepted to AAAI 2024.

## Paper Links
- **AAAI 2024 Version**: [Download paper (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/28621)
- **ArXiv Preprint**: [Download preprint (ArXiv)](https://arxiv.org/pdf/2312.14404.pdf)

## Dataset Download Guide
### CCGR Dataset
#### Derived data (Silhouette, Parsing, Pose)
We are pleased to offer the derived data for use in your research projects. You can download the data directly 
when you agree to comply with this Licence: CC BY-NC-ND (Creative Commons Attribution-NonCommercial-NoDerivatives).
- **Baidu Netdisk**: [Download Dataset](https://pan.baidu.com/s/1GUTdGRLHyqSHw0Fcc7iUEQ) (Code:ngcw)
- **OneDrive**: [Download Dataset](https://1drv.ms/f/c/8464f220191191b1/Eov74XWuOi1Op_fdXDRzoAMBbJLrqSN1HoM4_WLNLUNm0Q?e=A8RQAJ)

#### Raw data (RGB)
Please sign this agreement and send it to the email address (zoushinan@csu.edu.cn). 
We will process your request email as soon as possible. 
There are currently optional ways for applicants from mainland China to access the data; 
please let us know your choice in the email when you write.
1. Baidu Netdisk Link
2. OneDrive Link
3. Mailing Services. You can mail us a hard drive; we will copy the data to the hard drive and return it to you. 
(The hard drive needs to be larger than 2T, USB3.0 interface. Supported only in Mainland China)

Note: We're very sorry. Due to government network regulations in China (Internet Transmission Restrictions), 
we need more time to upload the RGB data to OneDrive, which means the OneDrive option is currently unavailable. 
(Because the CCGR raw data is too big)

### CCGR-Mini Dataset
CCGR-Mini is a subset of CCGR. The CCGR-Mini is smaller in size and can speed up your research.

We construct CCGR-Mini by extracting data from CCGR as follows. 
The 53 covariates for each human are retained, but of the 33 views under each covariate, 
one is randomly selected as data for the CCGR-Mini, and the remaining views are discarded. 
This way, each person still retains 53 covariates and enough views to maintain the original challenge. 
Moreover, with only 53 videos per person, data is significantly reduced. 

CCGR-MINI has 970 subjects, 47,884 sequences, 53 different covariates, and 33 different views.

We are pleased to offer the data for use in your research projects. You can download the data directly 
when you agree to comply with this Licence: CC BY-NC-ND (Creative Commons Attribution-NonCommercial-NoDerivatives).
#### Derived data (Silhouette, Parsing, Pose)
- **Baidu Netdisk**: [Download Dataset](https://pan.baidu.com/s/1h6auGcxWFqeUAws0PvSH8g) (Code:ei8e)
- **OneDrive**: [Download Dataset](https://1drv.ms/f/c/8464f220191191b1/Ev18lg3FHJZCoyF_6z91JUUBDgBX7EZN0WHJKJnDEIzbWA?e=xon8em)
#### Raw data (RGB)
- **Baidu Netdisk**: [Download Dataset](https://pan.baidu.com/s/1qHJxbbMamgEPwp8fd2sfkQ) (Code:oyqf)
- **OneDrive**: [Download Dataset]()

### Preview: Two new collected datasets CCGR-? and CCGR-? are coming.
## Code
We have uploaded the code, which is modified from OpenGait.
The main changes are listed below:
1. Compatible with CCGR and CCGR-Mini datasets. 
(The run.sh file contains commands to run all compatible algorithms.)
2. To be updated
## Results 
### CCGR 
#### Using silhouette (%)
| Methods    | R1^hard | R1^easy | R5^hard | R5^easy |
|------------|---------|---------|---------|---------|
 |GaitSet    |25.3  | 35.3 |46.7  |58.9|
 |GaitPart   | 22.6 |32.7  |42.9  |55.5|
 |GaitGL     |23.1  |35.2  |39.9  |54.1|
 |GaitBase   |31.3  |43.8  |51.3  |64.4|
 |DeepGaitV2 |42.5  |55.2  |63.2  |75.2|

  


#### Using parsing (%)

| Methods     | R1^hard | R1^easy | R5^hard | R5^easy |
|-------------|---------|---------|---------|---------|
 | GaitSet     | 31.6    | 42.8    | 54.8    | 67.0    |
 | GaitPart    | 29.0    | 40.9    | 51.5    | 64.5    |
 | GaitGL      | 28.4    | 42.1    | 46.6    | 61.4    |
 | GaitBase    | 48.1    | 62.0    | 67.7    | 79.6    |
 | DeepGaitV2  | 58.8    | 71.8    | 77.0    | 87.0    |



### CCGR-Mini 
#### Using silhouette (%)
| Methods    | R1    | mAP   | mINP | 
|------------|-------|-------|-----|
| GaitSet    | 13.77 | 15.39 | 5.75|
| GaitPart   | 8.02  | 10.12 | 3.52|
| GaitGL     | 17.51 | 18.12 | 6.85|
| GaitBase   | 26.99 | 24.89 |9.72 |
| DeepGaitV2 | 39.37 | 36.01 |16.77|

#### Using parsing (%)

| Methods    | R1     | mAP    | mINP  | 
|------------|--------|--------|-------|
| GaitSet    | 18.09  | 19.18  | 7.38  |    
| GaitPart   | 10.6   | 12.29  | 4.25  |    
| GaitGL     | 22.53  | 22.58  | 9.06  |       
| GaitBase   | 38.96  | 35.48  | 16.08 |    
| DeepGaitV2 | 50.43  | 46.53  | 24.43 |
## Cite Us
If you find our dataset or paper useful for your research, please consider citing:

```bibtex
@article{Zou_2024, 
title={Cross-Covariate Gait Recognition: A Benchmark}, 
volume={38}, number={7}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Zou, Shinan and Fan, Chao and Xiong, Jianbo and Shen, Chuanfu and Yu, Shiqi and Tang, Jin}, 
year={2024}, month={Mar.}, pages={7855-7863} }
```
## Contact 
If you have any questions, please contact (zoushinan@csu.edu.cn)
## Correct Some Mistakes
1. In the paper "Cross-Covariate Gait Recognition: A Benchmark" (AAAI2024 Version). 
The positions of R5^easy and R5^hard are reversed in Table 2. The rest of the data in the table is fine. 
We are sorry that this is our mistake.

2. In the paper "Cross-Covariate Gait Recognition: A Benchmark" (AAAI2024 Version). 
In Table 3, the experimental results of GaitBase and DeepGaitV2 are using batch size 8 X 8, 
which we mistakenly took as 8 X 16 when we recorded the data. 
The correct results have been placed in the results section of this note. 
The good thing is that this mistake does not affect the conclusion of this paper. 
Sorry, it was our mistake. Sincerely, please forgive us for our mistakes; 
it is not easy to deal with such a massive amount of data (raw, unprocessed data > 8TB).