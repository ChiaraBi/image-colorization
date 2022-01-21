import lpips
import torch
import numpy as np
import json
from os import listdir
from os.path import isfile, join

from utils_alexnet import *

'''
LPIPS metric using AlexNet.
'''

# DATA NORMALIZATION IN RANGE [-1,1] AND RESHAPING TO 3x256x256

model = 'chromagan'  # original, chromagan, dahl, siggraph, su, zhang,
                     # baseline_cartoon, baseline_without_cartoon

if model == 'original':
    path = '../img/original/test/'
elif model == 'baseline_cartoon' or model == 'baseline_without_cartoon':
    path = '../img/colorized/baseline/'+model+'/epochs_50/'
else:
    path = '../img/colorized/'+model+'/test/'

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

data_metrics = np.empty((len(onlyfiles), 256, 256, 3))
i = 0
for files in onlyfiles:
    img_array = load_img(join(path, files))
    img_resized = resize_img(img_array, HW=(256, 256), resample=3)
    data_metrics[i, :, :, :] = img_resized
    print(i)
    i += 1

# Normalization in range [-1,1]
data_metrics_norm = np.empty(data_metrics.shape)
for i in range(0, data_metrics.shape[0]):
    data_metrics_norm[i, :, :, 0] = 2*(data_metrics[i, :, :, 0] - data_metrics[i, :, :, :].min()) / (data_metrics[i, :, :, :].max() - data_metrics[i, :, :, :].min()) - 1
    data_metrics_norm[i, :, :, 1] = 2*(data_metrics[i, :, :, 1] - data_metrics[i, :, :, :].min()) / (data_metrics[i, :, :, :].max() - data_metrics[i, :, :, :].min()) - 1
    data_metrics_norm[i, :, :, 2] = 2*(data_metrics[i, :, :, 2] - data_metrics[i, :, :, :].min()) / (data_metrics[i, :, :, :].max() - data_metrics[i, :, :, :].min()) - 1
# NxHxWx3 -> Nx3xHxW
data_metrics_norm = data_metrics_norm.transpose((0, 3, 1, 2))
np.save('../resources/LPIPS/data_metrics_'+model, data_metrics_norm)


# DATA LOADING
originals = np.load('../resources/LPIPS/data_metrics_original.npy')
colorized = np.load('../resources/LPIPS/data_metrics_'+model+'.npy')

# Transform Numpy into Pytorch tensor
originals = torch.from_numpy(originals).float()
colorized = torch.from_numpy(colorized).float()

loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization
                                        # we were unable to run it.


d_alex = loss_fn_alex(originals, colorized)   # type: torch.FloatTensor
# d_vgg = loss_fn_vgg(originals, colorized)

metrics = dict(max=d_alex.max().item(), idx_max=d_alex.argmax().item(),
               min=d_alex.min().item(), idx_min=d_alex.argmin().item(),
               mean=d_alex.mean().item(), std=d_alex.std().item())

print(metrics)

with open('../resources/LPIPS/metrics_lpips_'+model+'.txt', 'w') as f:
    f.write(json.dumps(metrics))
