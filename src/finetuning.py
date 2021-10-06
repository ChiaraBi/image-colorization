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

data_train_scaled = np.load('../resources/data_train_scaled.npy')
data_test_scaled = np.load('../resources/data_test_scaled.npy')

train_mean = np.load('../resources/train_mean.npy')
train_std = np.load('../resources/train_std.npy')