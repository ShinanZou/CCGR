import os
from time import strftime, localtime
import numpy as np
from utils import get_msg_mgr, mkdir
import multiprocessing
from .metric import mean_iou, cuda_dist, compute_ACC_mAP, evaluate_rank, evaluate_many
from .re_rank import re_ranking
import time

def de_diag(acc, each_angle=False):
    # Exclude identical-view cases
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result


def diag(acc, each_angle=False):
    result = np.diag(acc)
    if not each_angle:
        result = np.mean(result)
    return result


def cross_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    '''More details can be found: More details can be found in 
        [A Comprehensive Study on the Evaluation of Silhouette-based Gait Recognition](https://ieeexplore.ieee.org/document/9928336).
    '''
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-01'], 'BG': ['bg-01'], 'CL': ['cl-01']},
                      'OUMVLP': {'NM': ['00']}}

    gallery_seq_dict = {'CASIA-B': ['nm-02', 'bg-02', 'cl-02'],
                        'OUMVLP': ['01']}

    msg_mgr = get_msg_mgr()
    acc = {}
    mean_ap = {}
    view_list = sorted(np.unique(view))
    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros(len(view_list)) - 1.
        mean_ap[type_] = np.zeros(len(view_list)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]
            gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset])
            gallery_y = label[gseq_mask]
            gallery_x = feature[gseq_mask, :]
            dist = cuda_dist(probe_x, gallery_x, metric)
            eval_results = compute_ACC_mAP(
                dist.cpu().numpy(), probe_y, gallery_y, view[pseq_mask], view[gseq_mask])
            acc[type_][v1] = np.round(eval_results[0] * 100, 2)
            mean_ap[type_][v1] = np.round(eval_results[1] * 100, 2)

    result_dict = {}
    msg_mgr.log_info(
        '===Cross View Gallery Evaluation (Excluded identical-view cases)===')
    out_acc_str = "========= Rank@1 Acc =========\n"
    out_map_str = "============= mAP ============\n"
    for type_ in probe_seq_dict[dataset].keys():
        avg_acc = np.mean(acc[type_])
        avg_map = np.mean(mean_ap[type_])
        result_dict[f'scalar/test_accuracy/{type_}-Rank@1'] = avg_acc
        result_dict[f'scalar/test_accuracy/{type_}-mAP'] = avg_map
        out_acc_str += f"{type_}:\t{acc[type_]}, mean: {avg_acc:.2f}%\n"
        out_map_str += f"{type_}:\t{mean_ap[type_]}, mean: {avg_map:.2f}%\n"
    # msg_mgr.log_info(f'========= Rank@1 Acc =========')
    msg_mgr.log_info(f'{out_acc_str}')
    # msg_mgr.log_info(f'========= mAP =========')
    msg_mgr.log_info(f'{out_map_str}')
    return result_dict

# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py


def single_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-05', 'nm-06'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']},
                      'OUMVLP': {'NM': ['00']},
                      'CASIA-E': {'NM': ['H-scene2-nm-1', 'H-scene2-nm-2', 'L-scene2-nm-1', 'L-scene2-nm-2', 'H-scene3-nm-1', 'H-scene3-nm-2', 'L-scene3-nm-1', 'L-scene3-nm-2', 'H-scene3_s-nm-1', 'H-scene3_s-nm-2', 'L-scene3_s-nm-1', 'L-scene3_s-nm-2', ],
                                  'BG': ['H-scene2-bg-1', 'H-scene2-bg-2', 'L-scene2-bg-1', 'L-scene2-bg-2', 'H-scene3-bg-1', 'H-scene3-bg-2', 'L-scene3-bg-1', 'L-scene3-bg-2', 'H-scene3_s-bg-1', 'H-scene3_s-bg-2', 'L-scene3_s-bg-1', 'L-scene3_s-bg-2'],
                                  'CL': ['H-scene2-cl-1', 'H-scene2-cl-2', 'L-scene2-cl-1', 'L-scene2-cl-2', 'H-scene3-cl-1', 'H-scene3-cl-2', 'L-scene3-cl-1', 'L-scene3-cl-2', 'H-scene3_s-cl-1', 'H-scene3_s-cl-2', 'L-scene3_s-cl-1', 'L-scene3_s-cl-2']
                                  }

                      }
    gallery_seq_dict = {'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
                        'OUMVLP': ['01'],
                        'CASIA-E': ['H-scene1-nm-1', 'H-scene1-nm-2', 'L-scene1-nm-1', 'L-scene1-nm-2']}
    msg_mgr = get_msg_mgr()
    acc = {}
    view_list = sorted(np.unique(view))
    if dataset == 'CASIA-E':
        view_list.remove("270")
    view_num = len(view_list)
    num_rank = 1
    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros((view_num, view_num)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]

            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset]) & np.isin(
                    view, [gallery_view])
                gallery_y = label[gseq_mask]
                gallery_x = feature[gseq_mask, :]
                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                acc[type_][v1, v2] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx], 1) > 0,
                                                     0) * 100 / dist.shape[0], 2)

    result_dict = {}
    msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
    out_str = ""
    for type_ in probe_seq_dict[dataset].keys():
        sub_acc = de_diag(acc[type_], each_angle=True)
        msg_mgr.log_info(f'{type_}: {sub_acc}')
        result_dict[f'scalar/test_accuracy/{type_}'] = np.mean(sub_acc)
        out_str += f"{type_}: {np.mean(sub_acc):.2f}%\t"
    msg_mgr.log_info(out_str)
    return result_dict


def evaluate_indoor_dataset(data, dataset, metric='euc', cross_view_gallery=False):
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)

    if dataset not in ('CASIA-B', 'OUMVLP', 'CASIA-E'):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    if cross_view_gallery:
        return cross_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)
    else:
        return single_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)


def evaluate_CCGR(data, dataset, metric='euc'):
    get_msg_mgr().log_info("Evaluating CCGR")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    return feature, label, seq_type, view


def evaluate_CCGR_MINI(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    features, labels, cams, time_seqs = data['embeddings'], data['labels'], data['types'], data['views']

    import json
    gallery_sets = json.load(
        open('./datasets/CCGR-MINI/CCGR-MINI.json', 'rb'))['GALLERY_SET']
    probe_mask = []
    for id, ty, sq in zip(labels, cams, time_seqs):
        if '-'.join([id, ty, sq]) in gallery_sets:
            probe_mask.append(False)
        else:
            probe_mask.append(True)
    probe_mask = np.array(probe_mask)
    probe_features = features[probe_mask]
    gallery_features = features[~probe_mask]
    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[~probe_mask]

    results = {}
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['scalar/test_accuracy/Rank-{}'.format(r)] = cmc[r - 1] * 100
    results['scalar/test_accuracy/mAP'] = mAP * 100
    results['scalar/test_accuracy/mINP'] = mINP * 100

    msg_mgr.log_info(results)
    return results


def evaluate_CCGR_3D_view(data, dataset="CCGR-3D", metric='euc'):
        msg_mgr = get_msg_mgr()

        feature1, label, seq_type, view = data['embeddings-view'], data['labels'], data['types'], data['views']
        index_of_max_in_rows = np.argmax(feature1, axis=1)
        mapping = {"0_75.avi": 0, "90_0.avi": 1, "120_25.avi": 2, "150_60.avi": 3, "270_40.avi": 4,
                   "0_75": 0, "90_0": 1, "120_25": 2, "150_60": 3, "270_40": 4,}
        re_mapping = {0: "0_75", 1: "90_0", 2: "120_25", 3: "150_60", 4: "270_40"}

        view_label = []
        for vi in view:
            view_label.append(mapping[vi])
        view_emb_label = []
        for vi in index_of_max_in_rows:
            view_emb_label.append(re_mapping[vi])
        view_emb_label = np.array(view_emb_label)

        indices1 = np.where(np.char.find(seq_type, 'NM1') != -1)
        indices2 = np.where(np.char.find(seq_type, 'NM3DTO2D') != -1)
        if indices1[0].size == 0:
            indices = indices2
        if indices2[0].size == 0:
            indices = indices1

        index_of_max_in_rows = np.delete(index_of_max_in_rows, indices)
        view_label = np.delete(view_label, indices)
        seq_type = np.delete(seq_type, indices)
        view = np.delete(view, indices)

        accuracy = index_of_max_in_rows == view_label
        count_true = np.sum(accuracy)
        view_acc_array = count_true / len(view_label)
        msg_mgr.log_info("view accuracy:")
        msg_mgr.log_info(view_acc_array)

        # probe_seq_dict = {'CCGR-3D': [["NM2"], ["FA1"], ["FBCL1"], ["FBCL_SH1"], ["FBCL_SH_HAT1"],
        #                               ["FBCL_SH_HAT_BG1"], ["FBCL_SH_HAT_BG_FA1"],
        #                               ["FBCL_SH_HAT_BG2"], ["FBCL_SH_HAT_BG_FA2"]]}
        # view_list = list(set(view))
        # view_list.sort()
        # view_acc_array = np.zeros((len(probe_seq_dict[dataset]), len(view_list)))
        # for i in range(len(probe_seq_dict[dataset])):
        #     for j in range(len(view_list)):
        #         aa_indices = np.where(np.char.find(seq_type, probe_seq_dict[dataset][i][0]) != -1)[0]
        #         bb_indices = np.where(np.char.find(view, view_list[j]) != -1)[0]
        #         common_indices = np.intersect1d(aa_indices, bb_indices)
        #         selected_index_of_max = index_of_max_in_rows[common_indices]
        #         selected_view_label = view_label[common_indices]
        #         sub_ccuracy = selected_index_of_max == selected_view_label
        #         view_acc_array[i][j] = np.sum(sub_ccuracy)/len(selected_view_label)
        current_directory = os.getcwd()
        save_path = current_directory + "/output/CCGR-3D/vrn/vrn_SIL/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path1 = save_path + "view_acc_array.npy"
        save_path2 = save_path + "view_emb_label.npy"
        np.save(save_path1, view_acc_array)
        np.save(save_path2, view_emb_label)
        # return view_emb_label
        return view_acc_array, view_emb_label

def FDPool(test1, op=1):
    import torch
    test1 = torch.from_numpy(test1).unsqueeze(3)
    test1 = torch.cat(torch.chunk(test1, op, 1), 3)
    feature = test1.mean(1)
    feature = feature.swapaxes(1, 2)
    feature = feature.numpy()

    return feature

def evaluate_CCGR_3D(data, dataset='CCGR-3D', metric='euc', FDPool_F=False):
    start_time = time.time()

    msg_mgr = get_msg_mgr()
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']

    # 用于tsne可视化保存特征和id标签的
    # import os.path as osp
    # save_dir = '/12T/zsn5/CCGR-Benchmark/output/CCGR-3D/msaffgait6Lshiyan/MSAFF30/'
    # np.save(osp.join(save_dir, "{}.npy".format(1)), feature)
    # np.save(osp.join(save_dir, "{}.npy".format(2)), label)
    if FDPool_F:
       feature = FDPool(feature)

    label = np.array(label)
    for i in range(len(view)):
        if ".avi" in view[i]:
            view[i] = view[i][0:-4]

    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)

    probe_seq_dict = {'CCGR-3D': [["NM2"], ["FA1"], ["FBCL1"], ["FBCL_SH1"], ["FBCL_SH_HAT1"],
                                  ["FBCL_SH_HAT_BG1"], ["FBCL_SH_HAT_BG_FA1"],
                                  ["FBCL_SH_HAT_BG2"], ["FBCL_SH_HAT_BG_FA2"]]}

    gallery_seq_dict = {
        'CCGR-3D': [["NM1"], ["NM3DTO2D"]]}

    np.set_printoptions(precision=4, floatmode='fixed')

    gallery_seq = gallery_seq_dict[dataset][1]
    if gallery_seq[0] in seq_type:
        test_3D = True
        print(1)
    else:
        test_3D = False
    xxxx = True
    if xxxx:
      if test_3D:
       if len(data) == 5:
           view_acc_array, view_emb_label = evaluate_CCGR_3D_view(data)
           # view_emb_label = evaluate_CCGR_3D_view(data)
       else:
           current_directory = os.getcwd()
           save_path = current_directory + "/output/CCGR-3D/vrn/vrn_SIL/" + "view_acc_array.npy"
           save_path1 = current_directory + "/output/CCGR-3D/vrn/vrn_SIL/" + "view_emb_label.npy"
           view_acc_array = np.load(save_path)
           print("view_acc_array:", view_acc_array)
           view_emb_label = np.load(save_path1)

    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 25

    acc = np.zeros([len(probe_seq_dict[dataset]),
                   view_num, view_num, num_rank]) - 1.
    if not test_3D:
      for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][0]
        for (v1, probe_view) in enumerate(view_list):
            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                    view, [gallery_view])
                gallery_x = feature[gseq_mask, :]
                gallery_y = label[gseq_mask]

                pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                    view, [probe_view])
                probe_x = feature[pseq_mask, :]
                probe_y = label[pseq_mask]

                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.sort(1)[1].cpu().numpy()
                # print(p, v1, v2, "\n")
                acc[p, v1, v2, :] = np.round(
                    np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                           0) * 100 / dist.shape[0], 2)


    acc_s = np.zeros([len(probe_seq_dict[dataset]),
                   view_num, num_rank]) - 1.
    if test_3D:
      xxxx = True
      if xxxx:
        import random
        # random.shuffle(view_emb_label)
        for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
          gallery_seq = gallery_seq_dict[dataset][1]
          for (v1, probe_view) in enumerate(view_list):
                  pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                      view, [probe_view])
                  probe_x = feature[pseq_mask, :]
                  probe_y = label[pseq_mask]

                  gallery_view_list = view_emb_label[pseq_mask]
                  gallery_view_list_set = list(set(gallery_view_list))
                  gallery_view_list_set.sort()
                  idx_list = []
                  for gallery_view in gallery_view_list_set:
                     gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                      view, [gallery_view])
                     gallery_x = feature[gseq_mask, :]
                     gallery_y = label[gseq_mask]

                     dist = cuda_dist(probe_x, gallery_x, metric)
                     idx = dist.sort(1)[1].cpu().numpy()
                     idx_list.append(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]])
                  new_idx_list = []
                  for (v2, gallery_view) in enumerate(gallery_view_list):
                      new_idx_list.append(idx_list[gallery_view_list_set.index(gallery_view)][v2])
                  new_idx_list = np.array(new_idx_list)
                  acc_s[p, v1, :] = np.round(
                          np.sum(np.cumsum(new_idx_list, 1) > 0,
                                 0) * 100 / new_idx_list.shape[0], 2)
      else:
          for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
              gallery_seq = gallery_seq_dict[dataset][1]
              for (v1, probe_view) in enumerate(view_list):
                      gseq_mask = np.isin(seq_type, gallery_seq)
                      gallery_x = feature[gseq_mask, :]
                      gallery_y = label[gseq_mask]

                      pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                          view, [probe_view])
                      probe_x = feature[pseq_mask, :]
                      probe_y = label[pseq_mask]

                      dist = cuda_dist(probe_x, gallery_x, metric)
                      idx = dist.sort(1)[1].cpu().numpy()
                      # print(p, v1, v2, "\n")
                      acc_s[p, v1, :] = np.round(
                          np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                 0) * 100 / dist.shape[0], 2)


    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000
    # kkkk = [0,4,9,14]
    kkkk = [0]
    print(f"retrieval times: {elapsed_time:.2f} ms")
    if not test_3D:
      for i in kkkk:
        msg_mgr.log_info(
            '===Rank-%d (Include identical-view cases)===' % (i + 1))
        msg_mgr.log_info('NM2: %.3f,\tFA1: %.3f,\tFBCL1: %.3f,\tFBCL_SH1: %.3f,\tFBCL_SH_HAT1: %.3f,\tFBCL_SH_HAT_BG1: %.3f,'
                         '\tFBCL_SH_HAT_BG_FA1: %.3f,\tFBCL_SH_HAT_BG2: %.3f,\tFBCL_SH_HAT_BG_FA2: %.3f,\tMean: %.3f'
                         % (
            np.mean(acc[0, :, :, i]),
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i]),
            np.mean(acc[3, :, :, i]),
            np.mean(acc[4, :, :, i]),
            np.mean(acc[5, :, :, i]),
            np.mean(acc[6, :, :, i]),
            np.mean(acc[7, :, :, i]),
            np.mean(acc[8, :, :, i]),
           (np.mean(acc[0, :, :, i]) + np.mean(acc[3, :, :, i]) + np.mean(acc[6, :, :, i]) +
            np.mean(acc[1, :, :, i]) + np.mean(acc[4, :, :, i]) + np.mean(acc[7, :, :, i]) +
            np.mean(acc[2, :, :, i]) + np.mean(acc[5, :, :, i]) + np.mean(acc[8, :, :, i]))/9.0
                         ))
      for i in kkkk:
        msg_mgr.log_info(
            '===Rank-%d (Exclude identical-view cases. Gallery: NM1)===' % (i + 1))
        msg_mgr.log_info('NM2: %.3f,\tFA1: %.3f,\tFBCL1: %.3f,\tFBCL_SH1: %.3f,\tFBCL_SH_HAT1: %.3f,\tFBCL_SH_HAT_BG1: %.3f,'
                         '\tFBCL_SH_HAT_BG_FA1: %.3f,\tFBCL_SH_HAT_BG2: %.3f,\tFBCL_SH_HAT_BG_FA2: %.3f,\tMean: %.3f'
                         % (
            de_diag(acc[0, :, :, i]),
            de_diag(acc[1, :, :, i]),
            de_diag(acc[2, :, :, i]),
            de_diag(acc[3, :, :, i]),
            de_diag(acc[4, :, :, i]),
            de_diag(acc[5, :, :, i]),
            de_diag(acc[6, :, :, i]),
            de_diag(acc[7, :, :, i]),
            de_diag(acc[8, :, :, i]),
           (de_diag(acc[0, :, :, i]) + de_diag(acc[3, :, :, i]) + de_diag(acc[6, :, :, i]) +
            de_diag(acc[1, :, :, i]) + de_diag(acc[4, :, :, i]) + de_diag(acc[7, :, :, i]) +
            de_diag(acc[2, :, :, i]) + de_diag(acc[5, :, :, i]) + de_diag(acc[8, :, :, i]))/9.0
                         ))
      for i in kkkk:
        msg_mgr.log_info(
            '===Rank-%d (Only Same View Case. Gallery: NM1)===' % (i + 1))
        msg_mgr.log_info('NM2: %.3f,\tFA1: %.3f,\tFBCL1: %.3f,\tFBCL_SH1: %.3f,\tFBCL_SH_HAT1: %.3f,\tFBCL_SH_HAT_BG1: %.3f,'
                         '\tFBCL_SH_HAT_BG_FA1: %.3f,\tFBCL_SH_HAT_BG2: %.3f,\tFBCL_SH_HAT_BG_FA2: %.3f,\tMean: %.3f'
                         % (
            diag(acc[0, :, :, i]),
            diag(acc[1, :, :, i]),
            diag(acc[2, :, :, i]),
            diag(acc[3, :, :, i]),
            diag(acc[4, :, :, i]),
            diag(acc[5, :, :, i]),
            diag(acc[6, :, :, i]),
            diag(acc[7, :, :, i]),
            diag(acc[8, :, :, i]),
           (diag(acc[0, :, :, i]) + diag(acc[3, :, :, i]) + diag(acc[6, :, :, i]) +
            diag(acc[1, :, :, i]) + diag(acc[4, :, :, i]) + diag(acc[7, :, :, i]) +
            diag(acc[2, :, :, i]) + diag(acc[5, :, :, i]) + diag(acc[8, :, :, i]))/9.0
                         ))

    if test_3D:
      for i in kkkk:
        msg_mgr.log_info(
            '===Rank-%d (Cross View Accuracy in the 3DGait Framework. Gallery: NM3DTO2D)===' % (i + 1))
        msg_mgr.log_info('NM2: %.3f,\tFA1: %.3f,\tFBCL1: %.3f,\tFBCL_SH1: %.3f,\tFBCL_SH_HAT1: %.3f,\tFBCL_SH_HAT_BG1: %.3f,'
                         '\tFBCL_SH_HAT_BG_FA1: %.3f,\tFBCL_SH_HAT_BG2: %.3f,\tFBCL_SH_HAT_BG_FA2: %.3f,\tMean: %.3f'
                         % (
            np.mean(acc_s[0, :, i]),
            np.mean(acc_s[1, :, i]),
            np.mean(acc_s[2, :, i]),
            np.mean(acc_s[3, :, i]),
            np.mean(acc_s[4, :, i]),
            np.mean(acc_s[5, :, i]),
            np.mean(acc_s[6, :, i]),
            np.mean(acc_s[7, :, i]),
            np.mean(acc_s[8, :, i]),
            np.mean(acc_s[:, :, i]),
            ))


    msg_mgr.log_info("View List:")
    msg_mgr.log_info(view_list)

    if not test_3D:
      for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d of each angle (Only Same View Case.)===' % (i + 1))
        msg_mgr.log_info('NM2: {}'.format(diag(acc[0, :, :, i], True)))
        msg_mgr.log_info('FA1: {}'.format(diag(acc[1, :, :, i], True)))
        msg_mgr.log_info('FBCL1: {}'.format(diag(acc[2, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH1: {}'.format(diag(acc[3, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH_HAT1: {}'.format(diag(acc[4, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH_HAT_BG1: {}'.format(diag(acc[5, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH_HAT_BG_FA1: {}'.format(diag(acc[6, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH_HAT_BG2: {}'.format(diag(acc[7, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH_HAT_BG_FA2: {}'.format(diag(acc[8, :, :, i], True)))
      for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d of each angle. (Exclude identical-view cases.)===' % (i + 1))
        msg_mgr.log_info('NM2: {}'.format(de_diag(acc[0, :, :, i], True)))
        msg_mgr.log_info('FA1: {}'.format(de_diag(acc[1, :, :, i], True)))
        msg_mgr.log_info('FBCL1: {}'.format(de_diag(acc[2, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH1: {}'.format(de_diag(acc[3, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH_HAT1: {}'.format(de_diag(acc[4, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH_HAT_BG1: {}'.format(de_diag(acc[5, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH_HAT_BG_FA1: {}'.format(de_diag(acc[6, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH_HAT_BG2: {}'.format(de_diag(acc[7, :, :, i], True)))
        msg_mgr.log_info('FBCL_SH_HAT_BG_FA2: {}'.format(de_diag(acc[8, :, :, i], True)))

    if test_3D:
      for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d of each angle (Cross View Accuracy in the 3DGait Framework.)===' % (i + 1))
        msg_mgr.log_info('NM2: {}'.format(acc_s[0, :, i])),
        msg_mgr.log_info('FA1: {}'.format(acc_s[1, :, i])),
        msg_mgr.log_info('FBCL1: {}'.format(acc_s[2, :, i])),
        msg_mgr.log_info('FBCL_SH1: {}'.format(acc_s[3, :, i])),
        msg_mgr.log_info('FBCL_SH_HAT1: {}'.format(acc_s[4, :, i])),
        msg_mgr.log_info('FBCL_SH_HAT_BG1: {}'.format(acc_s[5, :, i])),
        msg_mgr.log_info('FBCL_SH_HAT_BG_FA1: {}'.format(acc_s[6, :, i])),
        msg_mgr.log_info('FBCL_SH_HAT_BG2: {}'.format(acc_s[7, :, i])),
        msg_mgr.log_info('FBCL_SH_HAT_BG_FA2: {}'.format(acc_s[8, :, i]))

    result_dict = {}

    return result_dict

if __name__ == '__main__':
    evaluate_CCGR_3D()