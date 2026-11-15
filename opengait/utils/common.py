import copy
import os
import inspect
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import yaml
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict, namedtuple
from thop import profile
from ptflops import get_model_complexity_info
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op


class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v


def Ntuple(description, keys, values):
    if not is_list_or_tuple(keys):
        keys = [keys]
        values = [values]
    Tuple = namedtuple(description, keys)
    return Tuple._make(values)


def get_valid_args(obj, input_args, free_keys=[]):
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        if k in expected_keys:
            expected_args[k] = v
        elif k in free_keys:
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys != []:
        logging.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpect_keys), obj.__name__))
    return expected_args


def get_attr_from(sources, name):
    try:
        return getattr(sources[0], name)
    except:
        return get_attr_from(sources[1:], name) if len(sources) > 1 else getattr(sources[0], name)


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def is_bool(x):
    return isinstance(x, bool)


def is_str(x):
    return isinstance(x, str)


def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList)


def is_dict(x):
    return isinstance(x, dict) or isinstance(x, OrderedDict) or isinstance(x, Odict)


def is_tensor(x):
    return isinstance(x, torch.Tensor)


def is_array(x):
    return isinstance(x, np.ndarray)


def ts2np(x):
    return x.cpu().data.numpy()


def ts2var(x, **kwargs):
    return autograd.Variable(x, **kwargs).cuda()


def np2var(x, **kwargs):
    return ts2var(torch.from_numpy(x), **kwargs)


def list2var(x, **kwargs):
    return np2var(np.array(x), **kwargs)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def MergeCfgsDict(src, dst):
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    # with open("/12T/zsn5/CCGR-Benchmark/configs/default.yaml", 'r') as stream:
    with open("./configs/default.yaml", 'r') as stream:
        dst_cfgs = yaml.safe_load(stream)
    MergeCfgsDict(src_cfgs, dst_cfgs)
    return dst_cfgs


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def handler(signum, frame):
    logging.info('Ctrl+c/z pressed')
    os.system(
        "kill $(ps aux | grep main.py | grep -v grep | awk '{print $2}') ")
    logging.info('process group flush!')


def ddp_all_gather(features, dim=0, requires_grad=True):
    '''
        inputs: [n, ...]
    '''

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    feature_list = [torch.ones_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(feature_list, features.contiguous())

    if requires_grad:
        feature_list[rank] = features
    feature = torch.cat(feature_list, dim=dim)
    return feature


# https://github.com/pytorch/pytorch/issues/16885
class DDPPassthrough(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_ddp_module(module, find_unused_parameters=False, **kwargs):
    if len(list(module.parameters())) == 0:
        # for the case that loss module has not parameters.
        return module
    device = torch.cuda.current_device()
    module = DDPPassthrough(module, device_ids=[device], output_device=device,
                            find_unused_parameters=find_unused_parameters, **kwargs)
    return module


def params_count(net):
    n_parameters = sum(p.numel() for p in net.parameters())
    # print("ggg")
    #
    # def convert_flops_unit(value, is_macs=True, strict_conversion=True):
    #     """
    #     将 MACs 转换为 FLOPs 并格式化单位
    #
    #     参数：
    #         value: ptflops 输出的 MACs 数值（as_strings=False 时的原始数值，单位：MACs）
    #         is_macs: 是否为 MACs 输入（固定为 True）
    #         strict_conversion: 是否严格换算（1 MAC = 2 FLOPs），否则近似为 1:1
    #     返回：
    #         格式化的 FLOPs 字符串（如 "8.24 GFLOPs"）
    #     """
    #     # 第一步：MACs 转 FLOPs
    #     if strict_conversion:
    #         flops = value * 2  # 严格换算：1 MAC = 2 FLOPs
    #     else:
    #         flops = value  # 近似换算：1 MAC ≈ 1 FLOPs（论文常用）
    #
    #     # 第二步：转换为合适的单位（GFLOPs / MFLOPs / kFLOPs）
    #     if flops >= 1e9:
    #         return f"{flops / 1e9:.2f} GFLOPs"
    #     elif flops >= 1e6:
    #         return f"{flops / 1e6:.2f} MFLOPs"
    #     elif flops >= 1e3:
    #         return f"{flops / 1e3:.2f} kFLOPs"
    #     else:
    #         return f"{flops:.2f} FLOPs"
    #
    # def convert_params_unit(params):
    #     """将参数数量转换为 M/B 单位"""
    #     if params >= 1e9:
    #         return f"{params / 1e9:.2f} B"
    #     elif params >= 1e6:
    #         return f"{params / 1e6:.2f} M"
    #     elif params >= 1e3:
    #         return f"{params / 1e3:.2f} K"
    #     else:
    #         return f"{params:.2f}"
    #
    # # 计算模型的 MACs 和参数（as_strings=False 便于数值处理）
    # macs, params = get_model_complexity_info(
    #     net,  # 你的模型
    #     (2, 3, 448, 224),  # 输入形状（不含批量维度）
    #     as_strings=False,  # 返回数值（而非带单位的字符串）
    #     print_per_layer_stat=False,
    #     verbose=False
    # )
    #
    # # 转换为 FLOPs（选择论文中常用的换算方式）
    # # 建议：论文中注明 "FLOPs 由 MACs 近似换算（1 MAC ≈ 1 FLOP）" 或 "严格换算（1 MAC = 2 FLOPs）"
    # # flops_str = convert_flops_unit(macs, strict_conversion=False)  # 近似换算（常用）
    # flops_str = convert_flops_unit(macs, strict_conversion=True)  # 严格换算（可选）
    #
    # # 转换参数单位
    # params_str = convert_params_unit(params)
    #
    # # 输出结果（符合学术论文格式）
    # print(f"FLOPs: {flops_str}")  # 例如：FLOPs: 4.12 GFLOPs
    # print(f"Params: {params_str}")  # 例如：Params: 25.56 M
    # print(f"Params2: {n_parameters / 1e6:.2f} M")
    return 'Parameters Count: {:.5f}M'.format(n_parameters / 1e6)
