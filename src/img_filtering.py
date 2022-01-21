import cv2
import numpy as np

from os import listdir
from os.path import isfile, join

'''
Apply blurring filter, increase/decrease contrast, increase/decrease luminosity.
'''


def apply_blur_filter_and_save(img, kernel_sizes, output_path):
    filter_name = 'blur'
    for kernel_size in kernel_sizes:
        img_blurred = cv2.blur(np.asarray(img, dtype=np.float32), (kernel_size, kernel_size))
        save_path = output_path + '-' + filter_name + '-' + str(kernel_size) + '.jpeg'
        cv2.imwrite(save_path, img_blurred)


def change_contrast_and_save(img, contrast_values, output_path):
    filter_name = 'contrast'
    for alpha in contrast_values:
        img_contrast = cv2.convertScaleAbs(img, alpha=alpha)
        save_path = output_path + '-' + filter_name + '-' + str(alpha) + '.jpeg'
        cv2.imwrite(save_path, img_contrast)


def change_luminosity_and_save(img, luminosity_values, output_path):
    filter_name = 'luminosity'
    for beta in luminosity_values:
        img_luminosity = cv2.convertScaleAbs(img, beta=beta)
        save_path = output_path + '-' + filter_name + '-' + str(beta) + '.jpeg'
        cv2.imwrite(save_path, img_luminosity)


input_path = '../img/filtered/original'
output_path = '../img/filtered/'
img_paths = [f for f in listdir(input_path) if isfile(join(input_path, f))]

for img_name in img_paths:
    img_bgr = cv2.imread(join(input_path, img_name))  # BGR image

    # Blur filters
    kernel_sizes = [3, 7, 11]
    output_folder = join(output_path, 'blurred')
    o_path = join(output_folder, img_name[:-5])
    apply_blur_filter_and_save(img_bgr, kernel_sizes, o_path)

    # Low/High Contrast
    contrast_values = [0.7, 1.3]
    output_folder = join(output_path, 'contrast')
    o_path = join(output_folder, img_name[:-5])
    change_contrast_and_save(img_bgr, contrast_values, o_path)

    # Low/High Luminosity
    luminosity_values = [-30, 20]
    output_folder = join(output_path, 'luminosity')
    o_path = join(output_folder, img_name[:-5])
    change_luminosity_and_save(img_bgr, luminosity_values, o_path)

