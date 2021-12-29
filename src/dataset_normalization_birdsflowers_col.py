import numpy as np

from os import listdir
from os.path import isfile, join

from utils_alexnet import *

'''
Dataset normalization for finetuning.
'''

model = 'zhang'
test_path = '../img/colorized/'+model+'/BirdsFlowers_test/'

onlyfiles_test = [f for f in listdir(test_path) if isfile(join(test_path, f))]

data_test = np.empty((len(onlyfiles_test), 256, 256, 3))
i = 0
for files in onlyfiles_test:
    img_array = load_img(join(test_path, files))
    img_resized = resize_img(img_array, HW=(256, 256), resample=3)
    data_test[i, :, :, :] = img_resized
    print(i)
    i += 1

np.save('../resources/data_test_birdsflowers_'+model, data_test)

data_test = np.load('../resources/data_test_birdsflowers_'+model+'.npy')

# Normalization in range [0,1]
data_test_01 = np.empty(data_test.shape)
for i in range(0, data_test.shape[0]):
    data_test_01[i, :, :, 0] = (data_test[i, :, :, 0] - data_test[i, :, :, 0].min()) / (data_test[i, :, :, 0].max() - data_test[i, :, :, 0].min())
    data_test_01[i, :, :, 1] = (data_test[i, :, :, 1] - data_test[i, :, :, 1].min()) / (data_test[i, :, :, 1].max() - data_test[i, :, :, 1].min())
    data_test_01[i, :, :, 2] = (data_test[i, :, :, 2] - data_test[i, :, :, 2].min()) / (data_test[i, :, :, 2].max() - data_test[i, :, :, 2].min())

# Normalization wrt mean and std
train_mean = np.load('../resources/train_mean_birdsflowers_orig.npy')
train_std = np.load('../resources/train_std_birdsflowers_orig.npy')

data_test_scaled = np.empty(data_test_01.shape)
data_test_scaled[:, :, :, 0] = (data_test_01[:, :, :, 0] - train_mean[0]) / train_std[0]
data_test_scaled[:, :, :, 1] = (data_test_01[:, :, :, 1] - train_mean[1]) / train_std[1]
data_test_scaled[:, :, :, 2] = (data_test_01[:, :, :, 2] - train_mean[2]) / train_std[2]

np.save('../resources/data_test_scaled_birdsflowers_'+model, data_test_scaled)

