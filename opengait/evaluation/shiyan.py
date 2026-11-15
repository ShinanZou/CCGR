from datetime import datetime
import numpy as np
import argparse

from torch import nn
import torch
def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='40000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result
# /8T/zsn3/multimodelgait1/fidk/e/findke
kx = 1
kn = 246
test1 = np.load("/12T/zsn5/mutimodelgait1/1/work/113410.npy")
# test1 = torch.from_numpy(test1)
test1 = torch.from_numpy(test1).unsqueeze(3)
print(test1.size())
test1 = torch.cat(torch.chunk(test1, kx, 2),3)
print(test1.size())
# test1 = (test1.median(2)[0]+test1.mean(2))
# test1 = (test1.max(2)[0])
# test1 = (test1.min(2)[0])
test1 = (test1.mean(2))
# test1 = test1.median(2)[0]
# print(test1.size())
# test1 = test1.numpy()
# test1 = (test1.max(2)[0])
# print(test1.size())



# test1 = torch.cat([test1.sort(2, descending=True)[0][:, :, 0:3],test1.sort(2, descending=True)[0][:, :, 124:128],
#                    test1.sort(2, descending=True)[0][:, :, 253:]],2)
# test1 = torch.cat([test1.sort(2, descending=True)[0][:, :, 0:3]],2)
# test1 = test1[:,:,1:2]
print(test1.size())
test1 = test1.numpy()
test2 = np.load("/12T/zsn5/mutimodelgait1/1/work/120002.npy")
test3 = np.load("/12T/zsn5/mutimodelgait1/1/work/120003.npy")
test4 = np.load("/12T/zsn5/mutimodelgait1/1/work/120004.npy")
test = (test1, test2, test3, test4)
print('Evaluating...')
acc = evaluation(test, conf1['data'])

# Print rank-1 accuracy of the best model
# e.g.
# ===Rank-1 (Include identical-view cases)===
# NM: 95.405,     BG: 88.284,     CL: 72.041
for i in range(1):
      print('===Rank-%d (Include identical-view cases)===' % (i + 1))
      print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
          np.mean(acc[0, :, :, i]),
          np.mean(acc[1, :, :, i]),
          np.mean(acc[2, :, :, i])))

# Print rank-1 accuracy of the best model，excluding identical-view cases
# e.g.
# ===Rank-1 (Exclude identical-view cases)===
#NM: 94.964,     BG: 87.239,     CL: 70.355
for i in range(1):
      print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
      print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
          de_diag(acc[0, :, :, i]),
          de_diag(acc[1, :, :, i]),
          de_diag(acc[2, :, :, i])))

# Print rank-1 accuracy of the best model (Each Angle)
# e.g.
# ===Rank-1 of each angle (Exclude identical-view cases)===
# NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
# BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
# CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]
np.set_printoptions(precision=2)
for i in range(1):
      print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
      print('NM:', de_diag(acc[0, :, :, i], True))
      print('BG:', de_diag(acc[1, :, :, i], True))
      print('CL:', de_diag(acc[2, :, :, i], True))