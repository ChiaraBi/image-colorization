import matplotlib.pyplot as plt
import torchvision
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
from colorization.colorizers.util import *
import matplotlib.pyplot as plt
import cv2

model = torchvision.models.alexnet(pretrained=True)
orig_train = '../img/original/finetuning_train/'
orig_test = '../img/original/finetuning_test/'

onlyfiles_train = [f for f in listdir(orig_train) if isfile(join(orig_train, f))]
onlyfiles_test = [f for f in listdir(orig_test) if isfile(join(orig_test, f))]

data_train = np.empty((len(onlyfiles_train), 256, 256))
i = 0
for files in onlyfiles_train:
    img_array = load_img(join(orig_train, files))
    _, img_array_BW = preprocess_img(img_array, HW=(256,256), resample=3)  # resize to 256x256x3 and turn into BW
    img = cv2.cvtColor(img_array_BW,cv2.COLOR_GRAY2RGB)
    plt.imsave('../img/original/'+files, img)
    data_train[i,:,:] = img_array_BW[0,0,:,:]
    i += 1

data_test = np.empty((len(onlyfiles_test), 256, 256))
for files in onlyfiles_test:
    img_array = load_img(join(orig_test, files))
    _, img_array_BW = preprocess_img(img_array, HW=(256,256), resample=3)
    print(img_array_BW.shape)

#train_mean = orig_train.data.float().mean() / 255
#train_std = train_data.data.float().std() / 255

