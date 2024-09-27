import torch.nn.functional as F
import multiprocessing
import numpy as np
import torch
import os
from datetime import datetime
import pandas as pd
import argparse
'''
########################################################################################################################
Before OpenGait is compatible with the CCGR dataset, you can make some adjustments to test the CCGR dataset in advance. 
The modifications to OpenGait will not affect its existing functionality; they are merely adding compatibility for CCGR. 
Here are the specific changes:

1. In `opengait/data/dataset.py`, within the `__loader__` function.

Change from:
```
        for pth in paths:
            if pth.endswith('.pkl'):
```
To:
```
        for pth in paths:
          if 'sil' in pth: # If the path contains 'sil', indicating silhouette modality
          # if 'par' in pth: # Uncomment this if you want to use 'par' as an indicator instead
            if pth.endswith('.pkl'):
```
This is used to select the modality based on file naming.

2. Add the following function at the end of `opengait/evaluation/evaluator.py`.
```
def evaluate_CCGR(data, dataset, metric='euc'):
    get_msg_mgr().log_info("Evaluating CCGR")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    return feature, label, seq_type, view
```

3. In `opengait/base_model.py`.

Change from:
```
         self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])
```
To:
```
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])
        self.save_path1 = osp.join('output/', cfgs['data_cfg']['test_dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])
```

4. In `opengait/base_model.py`.

Change from:
```
            return eval_func(info_dict, dataset_name, **valid_args)
```
To:
```
            if 'CCGR' in dataset_name:
                feature, label, seq_type, view = eval_func(info_dict, dataset_name, **valid_args)
                if os.path.exists(model.save_path1):
                    pass
                else:
                    os.makedirs(model.save_path1)
                np.save(osp.join(model.save_path1, "{}.npy".format(1)), feature)
                np.save(osp.join(model.save_path1, "{}.npy".format(2)), label)
                np.save(osp.join(model.save_path1, "{}.npy".format(3)), seq_type)
                np.save(osp.join(model.save_path1, "{}.npy".format(4)), view)
            else:
                return eval_func(info_dict, dataset_name, **valid_args)
```
These changes are meant to adapt OpenGait for CCGR by modifying how modalities are selected, and evaluation results are stored.

########################################################################################################################
The steps to complete training and testing using CCGR are as follows (GaitSet example):
1. CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=50001 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR.yaml --phase train
2. CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=50001 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR.yaml --phase test
3. python CCGR_EVA.py --dataset CCGR --typedata sil --model GaitSet --savename GaitSet_SIL --dist_model 2 --cuda_num_all 01  


########################################################################################################################
Other:
The CCGR dataset is quite large, and completing all tests, including easy and hard metrics, is time-consuming. 
To increase efficiency, you can consider the following:

1. Priority on Easy Metrics Analysis:
   - Set the "easy" parameter to true. (python CCGR_EVA.py --easy)
   According to our experience, if the performance of algorithms is close, 
   the easy metrics do not fully reflect the performance difference under hard metrics. 
   However, this is useful for acceleration studies.

2. Implementation of Feature Dimension Pooling (https://github.com/ShinanZou/MSAFF):
   - Engage the feature dimension pooling by setting an "fdp" parameter to true. (python CCGR_EVA.py --fdp)
   While there is a potential for a minor dip in the performance of some models, this technique significantly reduces computation time.
   If feature dimension pooling is helpful for your research, please cite the following paper:
   ```
     @INPROCEEDINGS{ShinanZouMSAFF
    author={Zou, Shinan and Xiong, Jianbo and Fan, Chao and Yu, Shiqi and Tang, Jin},
    booktitle={2023 IEEE International Joint Conference on Biometrics (IJCB)}, 
    title={A Multi-Stage Adaptive Feature Fusion Neural Network for Multimodal Gait Recognition}, 
    year={2023}}
    ```
I recommend that you utilize these strategies to optimize the speed of your daily experiments. 
However, for final reports or when in-depth analysis is necessary, a full examination, including both easy and hard metrics, is essential. 
Please note that the results we have included in the published paper are based on complete tests, not utilizing these expedited methods.

'''
parser = argparse.ArgumentParser(description="Evaluate_CCGR")
parser.add_argument("--dataset", type=str, default='CCGR', help="dataset name.")
parser.add_argument("--typedata", type=str, default='sil', help="type of data.")
parser.add_argument("--model", type=str, default='Baseline', help="model name.")
parser.add_argument("--savename", type=str, default='GaitBase_SIL88', help="save name.")
parser.add_argument("--easy", action='store_true',
                    help="only easy metrics are tested.")
parser.add_argument("--fdp", action='store_true',
                    help="use feature dimension pooling (FDPool) to accelerate testing.")
parser.add_argument("--dist_model", type=int, default=1, help="1 or 2, GaitSet and  GaitGL use 2.")
parser.add_argument("--cuda_num_all", type=str, default="0123",
                    help="Which GPUs are used for testing, just enter the GPU number directly. "
                         "Usually you must have 2*3090 or higher GPUs to test.")
opt = parser.parse_args()
dataset = opt.dataset
typedata = opt.typedata
model = opt.model
savename = opt.savename
easy = opt.easy
fdp = opt.fdp
dist_model = opt.dist_model
cuda_num_all = opt.cuda_num_all
cuda_num_all = [int(i) for i in cuda_num_all]
cuda_id = 0
num_rank = 5


probe_seq_dict = {dataset: [['NM1'], ['NM2'], ['SU1'], ['UB1'], ['BK1'], ['PK1'], ['SD1'], ['BG1'], ['BGHV1'], ['BX1'], ['BXHV1'], ['BXHVBG1'], ['BXBGCL1'], ['CL1'],
 ['CLPK1'], ['CLUB1'], ['CLUBBG1'], ['CLUBBGSD1'], ['BGCRCL1'], ['BGCRCLSU1'], ['BGCRCLCVUB1'], ['BGCRCLCV1'], ['CV1'], ['CVBXHV1'], ['CVBXBG1'], ['BGCR1'], ['CR1'],
 ['UBBGSD1'], ['SF1'], ['SFUBBG1'], ['SFCL1'], ['SFUBBGCL1'], ['BMCL1'], ['BMCLBG1'], ['BMCLBGBX1'], ['BM1'], ['ASBGCLBXCV1'], ['ASBXHVCL1'], ['ASBX1'], ['AS1'],
 ['DN1'], ['DNBK1'], ['DNBKBG1'], ['DNBKBGCL1'], ['ATUBBGCLBG1'], ['ATBGCL1'], ['ATBG1'], ['AT1'], ['DT1'], ['DTBXHV1'], ['DTBXHVCL1'], ['DTBXCLBG1'], ['FD1'], ['CD1']]}
if easy:
    gallery_seq_dict = {dataset: [['NM1']]}
else:
    gallery_seq_dict = probe_seq_dict


# The following parameters control the accuracy print order.
print_seq = ['BK', 'BG', 'HVBG', 'BX', 'HVBX', 'TC', 'UB', 'CL', 'UTR', 'DTR', 'UTS', 'DTS', 'BM', 'CV', 'SF', 'FA', 'ST', 'NM1', 'NM2', 'CF', 'FD','MP', 'CL-UB',
 'HVBX-BG', 'BG-TC', 'SF-CL', 'UTR-BX', 'DTR-BK', 'DTS-HVBX', 'UTS-BG', 'BM-CL', 'CV-HVBX', 'CL-CF', 'CL-UB-BG', 'BX-BG-CL', 'BG-TC-CL', 'SF-UB-BG', 'UTR-HVBX-CL',
 'DTR-BK-BG', 'DTS-HVBX-CL', 'UTS-BG-CL', 'BM-CL-BG', 'CV-BX-BG', 'UB-BG-FA', 'CL-UB-BG-FA', 'BM-CL-BG-BX', 'BG-TC-CL-CV', 'DTR-BK-BG-CL',
 'DTS-BX-CL-BG', 'SF-UB-BG-CL', 'BG-TC-CL-ST', 'UTS-UB-BG-CL', 'BG-TC-CL-CV-UB', 'UTR-BG-CL-BX-CV']
print_id = [5, 8, 9, 10, 11, 27, 4, 14, 40, 41, 48, 49, 36, 23, 29, 7, 3, 1, 2, 6, 53, 54, 16, 12, 26, 31, 39, 42,
            50, 47, 33, 24, 15, 17, 13, 19, 30, 38, 43, 51, 46, 34, 25, 28, 18, 35, 22, 44, 52, 32, 20, 45, 21, 37]
print_view = ['0', '22_5', '45', '67_5', '90', '112_5', '135', '157_5', '180', '0', '22_5', '45', '67_5', '90', '112_5',
              '135', '157_5', '180', '0', '22_5', '45', '67_5', '90', '112_5', '135', '157_5', '180', '0', '45',  '90',
              '135', '180', 'overhead']
print_idv = [3, 21, 24, 28, 31, 7, 10, 14, 17, 2, 20, 23, 27, 30, 6, 9,
             13, 16, 1, 19, 22, 26, 29, 5, 8, 12, 15, 4, 25, 32, 11, 18,  33]


def cuda_dist_CCGR(x, y, cu):
  if dist_model == 1:
    metric = 'euc'
    x = torch.from_numpy(x).cuda(cu)
    y = torch.from_numpy(y).cuda(cu)
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda(cu)
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin
  elif dist_model == 2:
      x = torch.from_numpy(x).cuda(cu)
      y = torch.from_numpy(y).cuda(cu)
      dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
          1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
      dist = torch.sqrt(F.relu(dist))
      return dist


def de_diag_CCGR(acc, K, p, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), p) / K
    if not each_angle:
        result = np.mean(result)
    return result


def ac_CCGR(dataset, feature, view, seq_type, label, view_list, v1, probe_view, num_rank, acc, cu, ac_list):
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for (g, gallery_seq) in enumerate(gallery_seq_dict[dataset]):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist_CCGR(probe_x, gallery_x, cu)
                    idx = dist.sort(1)[1].cpu().numpy()
                    try:
                      acc[p, g, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)
                    except:
                        print(".............................")

    ac_list[0] = acc


def FDPool(test1, op=1):
    test1 = torch.from_numpy(test1).unsqueeze(3)
    # print('inputsize:', test1.size())
    test1 = torch.cat(torch.chunk(test1, op, 2), 3)
    # print(test1.size())
    feature = (test1.mean(2).squeeze(2))
    # feature = (test1.mean(2))
    print('outputsize:', feature.size())
    feature = feature.numpy()

    return feature


if __name__ == '__main__':
    savepath = os.path.join('output/', dataset, model, savename)
    print("dataset name:", dataset, "model name:", model, "save name:", savename, "easy:", easy, "FDPool:", fdp)
    _time1 = datetime.now()
    if os.path.exists(os.path.join(savepath, "{}.npy".format(5))):
        acc = np.load(os.path.join(savepath, "{}.npy".format(5)))
        view = np.load(os.path.join(savepath, "{}.npy".format(4)))
        view_list = sorted(list(set(view)))
        view_num = len(view_list)
        seq_type_num = len(probe_seq_dict[dataset])
    else:
        feature = None
        if dist_model == 1:
            feature = np.load(os.path.join(savepath, "{}.npy".format(1)))
        elif dist_model == 2:
            feature = np.load(os.path.join(savepath, "{}.npy".format(1)))
            n = np.size(feature, 0)
            feature = feature.swapaxes(1, 2)
            feature = feature.reshape(n, -1)
        label = np.load(os.path.join(savepath, "{}.npy".format(2)))
        seq_type = np.load(os.path.join(savepath, "{}.npy".format(3)))
        view = np.load(os.path.join(savepath, "{}.npy".format(4)))
        if fdp:
            feature = FDPool(feature)
            dist_model = 2

        print("feature shape:", feature.shape)
        print("label shape:", label.shape)
        label = np.array(label)
        view_list = sorted(list(set(view)))
        view_num = len(view_list)
        seq_type_num = len(probe_seq_dict[dataset])
        if easy:
            seq_type_num1 = 1
        else:
            seq_type_num1 = seq_type_num
        acc = np.zeros([seq_type_num, seq_type_num1, view_num, view_num, num_rank])
        manager = multiprocessing.Manager()
        acc_list = []
        for (v1, probe_view) in enumerate(view_list):
            qqq = manager.list()
            qqq.append(acc)
            acc_list.append(qqq)
        t_list = []
        for (v1, probe_view) in enumerate(view_list):
            if cuda_id == (len(cuda_num_all) - 1):
                cuda_id = 0
            else:
                cuda_id = cuda_id + 1
            cu = cuda_num_all[cuda_id]
            t = multiprocessing.Process(target=ac_CCGR,
                                        args=(dataset, feature, view, seq_type, label, view_list, v1, probe_view,
                                              num_rank, acc, cu, acc_list[v1]))
            t_list.append(t)
        for t in t_list:
            t.start()
        for t in t_list:
            t.join()
        for (v1, probe_view) in enumerate(view_list):
            acc = acc + acc_list[v1][0]
        np.save(os.path.join(savepath, "{}.npy".format(5)), acc)
    print(datetime.now() - _time1)

    acc_all = np.zeros([len(probe_seq_dict[dataset]), len(gallery_seq_dict[dataset])])
    acc_all_5 = np.zeros([len(probe_seq_dict[dataset]), len(gallery_seq_dict[dataset])])
    excl_data = []
    print('\n','\n','===Rank-1 33VIEW PROBE NM2 GALLERY NM1 ALL VIEWS (Exclude identical-view cases)===')
    P_NM2_G_NM1_33VIEW = de_diag_CCGR(acc[1, 0, :, :, 0], view_num - 1, 1, True)
    for (v1, probe_view) in enumerate(view_list):
        if v1 == 0:
            print('layer1 (pitch angle:5)', ":")
            excl_data.append(['layer1 (pitch angle:5)', 0])
        elif v1 == 9:
            print('layer2 (pitch angle:30)', ":")
            excl_data.append(['layer2 (pitch angle:30)', 0])
        elif v1 == 18:
            print('layer3 (pitch angle:55)', ":")
            excl_data.append(['layer3 (pitch angle:55)', 0])
        elif v1 == 27:
            print('layer4 (pitch angle:75)', ":")
            excl_data.append(['layer4 (pitch angle:75)', 0])
        elif v1 == 32:
            print('layer5 (pitch angle:90)', ":")
            excl_data.append(['layer5 (pitch angle:90)', 0])
        print(print_view[v1], ':', "%.2f" % P_NM2_G_NM1_33VIEW[print_idv[v1]-1], end='     ')
        excl_data.append([print_view[v1],  P_NM2_G_NM1_33VIEW[print_idv[v1]-1]])
        if (v1+1) % 9 == 0 or v1 ==31:
            print('\n')
    print('mean',np.mean(P_NM2_G_NM1_33VIEW))
    for i in range(len(probe_seq_dict[dataset])):
        for j in range(len(gallery_seq_dict[dataset])):
            acc_all[i, j] = de_diag_CCGR(acc[i, j, :, :, 0], view_num - 1, 1)
    if easy:
        pass
    else:
        print('\n','\n','===Rank-1 PROBE 53 walking conditions GALLERY 53 walking conditions   (Exclude identical cases)===')
        _walking_conditions = de_diag_CCGR(acc_all, seq_type_num - 1, 0, True)

    print('\n', '\n', '===easy_walking conditions===')
    excl_data.append(['easy_walking conditions', 0])
    for (i, case) in enumerate(probe_seq_dict[dataset]):
        print(print_seq[i], ':', "%.2f" % acc_all[(print_id[i]-1), 0], end='   ')
        excl_data.append([print_seq[i], acc_all[(print_id[i]-1), 0]])
        if (i+1) % 8 == 0:
            print('\n')
    if easy:
        pass
    else:
        print('\n', '\n', '===hard_walking conditions===')
        excl_data.append(['hard_walking conditions', 0])
        for (i, case) in enumerate(gallery_seq_dict[dataset]):
            print(print_seq[i], ':', "%.2f" % _walking_conditions[print_id[i] - 1], end='   ')
            excl_data.append([print_seq[i], _walking_conditions[print_id[i] - 1]])
            if (i + 1) % 8 == 0:
                print('\n'
)
    for i in range(len(probe_seq_dict[dataset])):
        for j in range(len(gallery_seq_dict[dataset])):
            acc_all_5[i, j] = de_diag_CCGR(acc[i, j, :, :, 4], view_num - 1, 1)
    _walking_conditions_5 = de_diag_CCGR(acc_all_5, seq_type_num - 1, 0, True)

    if easy:
        print('\n', '\n', '=============ONLY EASY OVER ALL RESULT==============')
        print('Rank-1 easy : ', "%.2f" % np.mean(acc_all), '    ',
              'Rank-5 easy : ', "%.2f" % np.mean(acc_all_5), '\n')
    else:
        print('\n', '\n', '=============OVER ALL RESULT==============')
        print('Rank-1 hard : ', "%.2f" % np.mean(_walking_conditions), '    ',
              'Rank-1 easy : ', "%.2f" % _walking_conditions[0], '    ',
              'Rank-5 hard : ', "%.2f" % np.mean(_walking_conditions_5), '    ',
              'Rank-5 easy : ', "%.2f" % _walking_conditions_5[0], '\n')
        excl_data.append(['Rank-1 hard', np.mean(_walking_conditions)])
        excl_data.append(['Rank-1 easy', _walking_conditions[0]])
        excl_data.append(['Rank-5 hard', np.mean(_walking_conditions_5)])
        excl_data.append(['Rank-5 easy', _walking_conditions_5[0]])
        df = pd.DataFrame(excl_data, columns=['acc', 'data'])
        writer = pd.ExcelWriter('%s.xlsx' % (dataset + savename), engine='openpyxl')
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.save()

    




