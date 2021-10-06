import matplotlib.pyplot as plt
import torchvision
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
from colorization.colorizers.util import *
import matplotlib.pyplot as plt
import cv2

orig_train = '../img/original/finetuning_train/'
orig_test = '../img/original/finetuning_test/'

onlyfiles_train = [f for f in listdir(orig_train) if isfile(join(orig_train, f))]
onlyfiles_test = [f for f in listdir(orig_test) if isfile(join(orig_test, f))]

data_train = np.empty((len(onlyfiles_train), 256, 256))
i = 0
for files in onlyfiles_train:
    img_array = load_img(join(orig_train, files))
    _, img_array_BW = preprocess_img(img_array, HW=(256, 256), resample=3)  # resize to 256x256x3 and turn into BW

    # Save the BW images:
    # img_bw = postprocess_tens(img_array_BW, torch.cat((0 * img_array_BW, 0 * img_array_BW), dim=1))
    # plt.imsave('../img/original/'+files, img_bw)

    data_train[i,:,:] = img_array_BW[0,0,:,:]
    i += 1

np.save('../resources/data_train', data_train)

data_test = np.empty((len(onlyfiles_test), 256, 256))
i = 0
for files in onlyfiles_test:
    img_array = load_img(join(orig_test, files))
    _, img_array_BW = preprocess_img(img_array, HW=(256, 256), resample=3)
    data_test[i, :, :] = img_array_BW[0, 0, :, :]
    i += 1

np.save('../resources/data_test', data_test)


data_train = np.load('../resources/data_train.npy')
data_test = np.load('../resources/data_test.npy')

# Normalization in range [0,1]
data_train_01 = np.empty(data_train.shape)
for i in range(0, data_train.shape[0]):
    data_train_01[i, :, :] = (data_train[i, :, :] - data_train[i, :, :].min()) / (data_train[i, :, :].max() - data_train[i, :, :].min())

data_test_01 = np.empty(data_test.shape)
for i in range(0, data_test.shape[0]):
    data_test_01[i, :, :] = (data_test[i, :, :] - data_test[i, :, :].min()) / (data_test[i, :, :].max() - data_test[i, :, :].min())

# Normalization wrt mean and std
train_mean = np.mean(data_train_01, axis = 0)
train_std = np.std(data_train_01, axis = 0)

data_train_scaled = (data_train_01 - train_mean)/train_std
data_test_scaled = (data_test_01 - train_mean)/train_std

np.save('../resources/train_mean', train_mean)
np.save('../resources/train_std', train_std)
np.save('../resources/data_train_scaled', data_train_scaled)
np.save('../resources/data_test_scaled', data_test_scaled)

