import numpy as np

from os import listdir
from os.path import isfile, join

from utils_alexnet import *

'''
Dataset normalization of ri-colorized images in the test set of 
finetuning and feature extraction experiments.
The mean and standard deviation of the normalized original images in the training set
is used for standardization.
'''

model = 'zhang'  # orig, chromagan, dahl, siggraph, su, zhang,
                 # baseline_cartoon, baseline_without_cartoon

# The following path refer to the test directory of ri-colorized images
# from ImageNet or Birds and Flowers.
# The train directory should contain about 75% of data, the test directory about 25%
test_path = '../img/colorized/'+model+'/dataset_test/'

dataset = 'imagenet'  # imagenet, birdsflowers

onlyfiles_test = [f for f in listdir(test_path) if isfile(join(test_path, f))]

data_test = np.empty((len(onlyfiles_test), 256, 256, 3))
i = 0
for files in onlyfiles_test:
    img_array = load_img(join(test_path, files))
    img_resized = resize_img(img_array, HW=(256, 256), resample=3)
    data_test[i, :, :, :] = img_resized
    print(i)
    i += 1

# Normalization in range [0,1]
data_test_01 = np.empty(data_test.shape)
for i in range(0, data_test.shape[0]):
    data_test_01[i, :, :, 0] = (data_test[i, :, :, 0] - data_test[i, :, :, 0].min()) / (data_test[i, :, :, 0].max() - data_test[i, :, :, 0].min())
    data_test_01[i, :, :, 1] = (data_test[i, :, :, 1] - data_test[i, :, :, 1].min()) / (data_test[i, :, :, 1].max() - data_test[i, :, :, 1].min())
    data_test_01[i, :, :, 2] = (data_test[i, :, :, 2] - data_test[i, :, :, 2].min()) / (data_test[i, :, :, 2].max() - data_test[i, :, :, 2].min())

# Normalization wrt mean and std of the training images
train_mean = np.load('../resources/standardization/train_mean_'+dataset+'_orig.npy')
train_std = np.load('../resources/standardization/train_std_'+dataset+'_orig.npy')

data_test_scaled = np.empty(data_test_01.shape)
data_test_scaled[:, :, :, 0] = (data_test_01[:, :, :, 0] - train_mean[0]) / train_std[0]
data_test_scaled[:, :, :, 1] = (data_test_01[:, :, :, 1] - train_mean[1]) / train_std[1]
data_test_scaled[:, :, :, 2] = (data_test_01[:, :, :, 2] - train_mean[2]) / train_std[2]

np.save('../resources/standardization/data_test_'+dataset+'_'+model, data_test_scaled)

