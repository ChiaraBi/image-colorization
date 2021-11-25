import numpy as np

from os import listdir
from os.path import isfile, join

from utils_alexnet import *

'''
Dataset normalization for finetuning.
'''

orig_train = '../img/original/finetuning_train/'
# orig_test = '../img/original/finetuning_test/'
orig_test = '../img/original/finetuning_test_/'

onlyfiles_train = [f for f in listdir(orig_train) if isfile(join(orig_train, f))]
onlyfiles_test = [f for f in listdir(orig_test) if isfile(join(orig_test, f))]

# data_train = np.empty((len(onlyfiles_train), 256, 256, 3))
data_train = np.empty((len(onlyfiles_train)-150, 256, 256, 3))
i = 0
# for files in onlyfiles_train:
for files in onlyfiles_train[0:(len(onlyfiles_train)-150)]:   # remove last class
    img_array = load_img(join(orig_train, files))
    _, img_array_BW = preprocess_img(img_array, HW=(256, 256), resample=3)  # resize to 256x256x3 and turn into BW

    img_bw = postprocess_tens(img_array_BW, torch.cat((0 * img_array_BW, 0 * img_array_BW), dim=1))

    data_train[i, :, :, :] = img_bw
    print(i)
    i += 1

# np.save('../resources/data_train', data_train)
np.save('../resources/data_train_LOO', data_train)

data_test = np.empty((len(onlyfiles_test), 256, 256, 3))
i = 0
for files in onlyfiles_test:
    img_array = load_img(join(orig_test, files))
    _, img_array_BW = preprocess_img(img_array, HW=(256, 256), resample=3)
    img_bw = postprocess_tens(img_array_BW, torch.cat((0 * img_array_BW, 0 * img_array_BW), dim=1))
    data_test[i, :, :, :] = img_bw
    print(i)
    i += 1

np.save('../resources/data_test', data_test)


# data_train = np.load('../resources/data_train.npy')
data_train = np.load('../resources/data_train_LOO.npy')
data_test = np.load('../resources/data_test.npy')

# Normalization in range [0,1]
data_train_01 = np.empty(data_train.shape)
for i in range(0, data_train.shape[0]):
    data_train_01[i, :, :, 0] = (data_train[i, :, :, 0] - data_train[i, :, :, 0].min()) / (data_train[i, :, :, 0].max() - data_train[i, :, :, 0].min())
    data_train_01[i, :, :, 1] = (data_train[i, :, :, 1] - data_train[i, :, :, 1].min()) / (data_train[i, :, :, 1].max() - data_train[i, :, :, 1].min())
    data_train_01[i, :, :, 2] = (data_train[i, :, :, 2] - data_train[i, :, :, 2].min()) / (data_train[i, :, :, 2].max() - data_train[i, :, :, 2].min())

data_test_01 = np.empty(data_test.shape)
for i in range(0, data_test.shape[0]):
    data_test_01[i, :, :, 0] = (data_test[i, :, :, 0] - data_test[i, :, :, 0].min()) / (data_test[i, :, :, 0].max() - data_test[i, :, :, 0].min())
    data_test_01[i, :, :, 1] = (data_test[i, :, :, 1] - data_test[i, :, :, 1].min()) / (data_test[i, :, :, 1].max() - data_test[i, :, :, 1].min())
    data_test_01[i, :, :, 2] = (data_test[i, :, :, 2] - data_test[i, :, :, 2].min()) / (data_test[i, :, :, 2].max() - data_test[i, :, :, 2].min())

# Normalization wrt mean and std
train_mean = [np.mean(data_train_01[:, :, :, 0], axis = 0), np.mean(data_train_01[:, :, :, 1], axis = 0), np.mean(data_train_01[:, :, :, 2], axis = 0)]
train_std = [np.std(data_train_01[:, :, :, 0], axis = 0), np.std(data_train_01[:, :, :, 1], axis = 0), np.std(data_train_01[:, :, :, 2], axis = 0)]

data_train_scaled = np.empty(data_train_01.shape)
data_train_scaled[:, :, :, 0] = (data_train_01[:, :, :, 0] - train_mean[0]) / train_std[0]
data_train_scaled[:, :, :, 1] = (data_train_01[:, :, :, 1] - train_mean[1]) / train_std[1]
data_train_scaled[:, :, :, 2] = (data_train_01[:, :, :, 2] - train_mean[2]) / train_std[2]

data_test_scaled = np.empty(data_test_01.shape)
data_test_scaled[:, :, :, 0] = (data_test_01[:, :, :, 0] - train_mean[0]) / train_std[0]
data_test_scaled[:, :, :, 1] = (data_test_01[:, :, :, 1] - train_mean[1]) / train_std[1]
data_test_scaled[:, :, :, 2] = (data_test_01[:, :, :, 2] - train_mean[2]) / train_std[2]

# np.save('../resources/train_mean', train_mean)
np.save('../resources/train_mean_LOO', train_mean)
np.save('../resources/train_std', train_std)
# np.save('../resources/data_train_scaled', data_train_scaled)
np.save('../resources/data_train_scaled_LOO', data_train_scaled)
np.save('../resources/data_test_scaled', data_test_scaled)

