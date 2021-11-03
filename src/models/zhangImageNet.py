import matplotlib.pyplot as plt
import torch

import os
from os import listdir
from os.path import isfile, join

from colorization import colorizers
from colorization.colorizers.util import load_img, preprocess_img, postprocess_tens

colorizer_eccv16 = colorizers.eccv16().eval()
# colorizer_eccv16.cuda() # uncomment this if you're using GPU

# images' paths
imageNet_input_dir = '../../img/original/ImageNet/'
imageNet_output_dir = '../../img/colorized/zhang/ImageNet/'
# imageNet_output_dir = '../../img/colorized/siggraph/ImageNet/'  # use this path for Zhang's siggraph model

# IMAGENET
# directory list:
onlydirectories = [f for f in listdir(imageNet_input_dir) if not isfile(join(imageNet_input_dir, f))]

count = 0
for d in onlydirectories:
    onlyfiles = [f for f in listdir(imageNet_input_dir + d) if isfile(join(imageNet_input_dir + d, f))]
    for i in onlyfiles:
        count += 1
        if count%100 == 0:
            print(count)
        img = load_img(join(imageNet_input_dir + d, i))
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        # colorizer outputs 256x256 ab map
        # resize and concatenate to original L channel
        img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig, 0*tens_l_orig), dim=1))
        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        out_path = join(imageNet_output_dir + d, i)
        if not os.path.exists(imageNet_output_dir + d):
            os.makedirs(imageNet_output_dir + d)
        plt.imsave(out_path, out_img_eccv16)

