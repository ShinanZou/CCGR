#交叉协变量步态识别:一个基准

欢迎访问“交叉协变量步态识别:基准”论文的官方知识库，该论文已被AAAI 2024接受。

##纸质链接
- **AAAI 2024版**: [下载论文(AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/28621)
- **ArXiv预印本**: [下载预印本(ArXiv)](https://arxiv.org/pdf/2312.14404.pdf)

##数据集下载指南
###CCGR数据集
####衍生数据(轮廓、解析、姿势)
我们很高兴为您的研究项目提供衍生数据。你可以直接下载数据
当您同意遵守本许可证时:CC BY-NC-ND(知识共享署名-非商业性使用-不可转让)。
- **百度网盘**: [下载数据集](https://pan.baidu.com/s/1GUTdGRLHyqSHw0Fcc7iUEQ)(代码:ngcw)
- **OneDrive**: [下载数据集](https://1drv.ms/f/c/8464f220191191b1/Eov74XWuOi1Op_fdXDRzoAMBbJLrqSN1HoM4_WLNLUNm0Q?e=A8RQAJ)

#### Raw data (RGB)
请在本[协议](https://github.com/ShinanZou/CCGR/blob/CCGR-Benchmark/output/CCGR_Dataset_RGB_Data_Usage_Agreement.pdf)上签名，并发至邮箱(zoushinan@csu.edu.cn)。
我们将尽快处理您的请求电子邮件。
目前，中国大陆的申请人可以通过多种方式获取数据；
请让我们知道你写邮件时的选择。
1. Baidu Netdisk Link
2. OneDrive Link
3. Mailing Services. You can mail us a hard drive; we will copy the data to the hard drive and return it to you. 
(硬盘需要大于2T，USB3.0接口。仅在Mainland China支持)

注意:我们非常抱歉。由于中国政府网络法规(互联网传输限制)，
我们需要更多时间将RGB数据上传到OneDrive，这意味着OneDrive选项当前不可用。
(因为CCGR原始数据太大)

### CCGR-Mini Dataset
CCGR迷你是CCGR的子集。迷你CCGR的体积更小，可以加快你的研究。

我们通过从CCGR提取如下数据来构建CCGR迷你模型。
每个人的53个协变量被保留，但是在每个协变量下的33个视图中，
随机选择一个作为CCGR迷你的数据，其余的视图被丢弃。
这样，每个人仍然保留53个协变量和足够的视图来维持原始挑战。
此外，每人只有53个视频，数据明显减少。

CCGR迷你有970名受试者，47，884个序列，53个不同的协变量和33个不同的视图。

我们很高兴为你的研究项目提供数据。你可以直接下载数据
当您同意遵守本许可证时:CC BY-NC-ND(知识共享署名-非商业性使用-不可转让)。
####衍生数据(轮廓、解析、姿势)
- **百度网盘**: [下载数据集](https://pan.baidu.com/s/1h6auGcxWFqeUAws0PvSH8g) (Code:ei8e)
- **OneDrive**: [下载数据集](https://1drv.ms/f/c/8464f220191191b1/Ev18lg3FHJZCoyF_6z91JUUBDgBX7EZN0WHJKJnDEIzbWA?e=xon8em)
#### Raw data (RGB)
- **百度网盘**: [下载数据集](https://pan.baidu.com/s/1qHJxbbMamgEPwp8fd2sfkQ) (Code:oyqf)
- **OneDrive**: [下载数据集]()

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
