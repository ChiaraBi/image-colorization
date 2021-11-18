import lpips
import torch
import numpy as np
from os import listdir
from os.path import isfile, join

from utils_alexnet import *

# Image normalization
model = 'chromagan'
# path = '../img/original/test/'
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
    data_metrics_norm[i, :, :, 0] = 2*(data_metrics[i, :, :, 0] - data_metrics[i, :, :, 0].min()) / (data_metrics[i, :, :, 0].max() - data_metrics[i, :, :, 0].min()) - 1
    data_metrics_norm[i, :, :, 1] = 2*(data_metrics[i, :, :, 1] - data_metrics[i, :, :, 1].min()) / (data_metrics[i, :, :, 1].max() - data_metrics[i, :, :, 1].min()) - 1
    data_metrics_norm[i, :, :, 2] = 2*(data_metrics[i, :, :, 2] - data_metrics[i, :, :, 2].min()) / (data_metrics[i, :, :, 2].max() - data_metrics[i, :, :, 2].min()) - 1

# NxHxWx3 -> Nx3xHxW
data_metrics_norm = data_metrics_norm.transpose((0, 3, 1, 2))

np.save('../resources/data_metrics_original', data_metrics_norm)
# data_metrics_norm = np.load('../resources/data_metrics_original.npy')

# Transform Numpy into Pytorch tensor
originals = torch.from_numpy(data_metrics_norm)
colorized = None

'''
loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

# image should be RGB, IMPORTANT: normalized to [-1,1]
d = loss_fn_alex(originals, colorized)
'''