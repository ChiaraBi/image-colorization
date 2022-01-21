# ChromaGAN
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras import applications
from keras.models import load_model
import numpy as np
import cv2

import os
from os import listdir
from os.path import isfile, join

# DIRECTORY INFORMATION
DATA_DIR = os.path.join('../../img/original/test')
OUT_DIR = os.path.join('../../img/colorized/chromagan/test')

MODEL_DIR = os.path.join('../../models')
BATCH_SIZE = 1

# TRAINING INFORMATION
PRETRAINED = "../../models/ChromaGAN.h5"

file_list = [f for f in listdir(DATA_DIR) if (isfile(join(DATA_DIR, f)) and f != '.DS_Store')]
size = len(file_list)  # 2323


def read_img(filename):
    IMAGE_SIZE = 224
    img = cv2.imread(filename, 3)
    height, width, channels = img.shape
    labimg = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
    labimg_ori = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    return np.reshape(labimg[:, :, 0], (IMAGE_SIZE, IMAGE_SIZE, 1)), labimg[:, :, 1:], img, np.reshape(
        labimg_ori[:, :, 0], (height, width, 1))


def generate_batch(data_index):
    batch = []
    labels = []
    filelist = []
    labimg_oritList = []
    originalList = []
    for i in range(BATCH_SIZE):
        filename = file_list[data_index]
        filelist.append(filename)
        file_path = join(DATA_DIR, filename)
        greyimg, colorimg, original, labimg_ori = read_img(file_path)
        batch.append(greyimg)
        labels.append(colorimg)
        originalList.append(original)
        labimg_oritList.append(labimg_ori)
        data_index = (data_index + 1) % size
    batch = np.asarray(batch) / 255  # values between 0 and 1
    labels = np.asarray(labels) / 255  # values between 0 and 1
    originalList = np.asarray(originalList)
    labimg_oritList = np.asarray(labimg_oritList) / 255
    return batch, labels, filelist, originalList, labimg_oritList, data_index


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)

    return result


def sample_images():
    data_index = 0

    avg_ssim = 0
    avg_psnr = 0
    VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True)
    save_path = PRETRAINED
    colorizationModel = load_model(save_path)
    assert size >= 0, "Your list of images to colorize is empty. Please load images."
    assert BATCH_SIZE <= size, "The batch size (" + str(
        BATCH_SIZE) + ") should be smaller or equal to the number of testing images (" + str(
        size) + ") --> modify it"
    total_batch = int(size / BATCH_SIZE)
    print("")
    print("number of ditectories of images to colorize: " + str(size))
    print("total number of batches to colorize: " + str(total_batch))
    print("")
    if not os.path.exists(OUT_DIR):
        print('created save result path')
        os.makedirs(OUT_DIR)
    for b in range(total_batch):
        batchX, batchY, filelist, original, labimg_oritList, data_index = generate_batch(data_index)
        predY, _ = colorizationModel.predict(np.tile(batchX, [1, 1, 1, 3]))
        predictVGG = VGG_modelF.predict(np.tile(batchX, [1, 1, 1, 3]))
        loss = colorizationModel.evaluate(np.tile(batchX, [1, 1, 1, 3]), [batchY, predictVGG], verbose=0)

        for i in range(BATCH_SIZE):
            originalResult = original[i]
            height, width, channels = originalResult.shape
            predY_2 = deprocess(predY[i])
            predY_2 = cv2.resize(predY_2, (width, height))
            labimg_oritList_2 = labimg_oritList[i]
            predResult_2 = reconstruct(deprocess(labimg_oritList_2), predY_2)
            ssim = tf.keras.backend.eval(tf.image.ssim(tf.convert_to_tensor(originalResult, dtype=tf.float32),
                                                       tf.convert_to_tensor(predResult_2, dtype=tf.float32),
                                                       max_val=255))
            psnr = tf.keras.backend.eval(tf.image.psnr(tf.convert_to_tensor(originalResult, dtype=tf.float32),
                                                       tf.convert_to_tensor(predResult_2, dtype=tf.float32),
                                                       max_val=255))
            avg_ssim += ssim
            avg_psnr += psnr
            save_path = os.path.join(OUT_DIR, filelist[i])
            print(save_path)
            cv2.imwrite(save_path, predResult_2)
            print("")
            print("Image " + str(i + 1) + "/" + str(BATCH_SIZE) + " in batch " + str(b + 1) + "/" + str(
                total_batch) + ". From left to right: grayscale image to colorize, colorized image ( PSNR =",
                  "{:.8f}".format(psnr), ")")
            print("and ground truth image. Notice that PSNR has no sense in original black and white images.")
            print("")
            print("")

        print("average ssim loss =", "{:.8f}".format(avg_ssim / (total_batch * BATCH_SIZE)))
        print("average psnr loss =", "{:.8f}".format(avg_psnr / (total_batch * BATCH_SIZE)))


sample_images()
