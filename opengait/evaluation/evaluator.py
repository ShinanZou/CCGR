import os
from time import strftime, localtime
import numpy as np
from utils import get_msg_mgr, mkdir
import multiprocessing
from .metric import mean_iou, cuda_dist, compute_ACC_mAP, evaluate_rank, evaluate_many
from .re_rank import re_ranking

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