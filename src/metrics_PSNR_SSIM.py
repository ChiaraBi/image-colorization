import numpy as np
import json
import pickle
import os
from os import listdir
from os.path import isfile, join
import tensorflow as tf

model = 'dahl'   # siggraph

'''
if model == 'original':
    path = '../img/original/test/'
else:
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


np.save('../resources/data_metrics_'+model+'_0_255', data_metrics)
'''


originals = np.load('../resources/data_metrics_original_0_255.npy')
colorized = np.load('../resources/data_metrics_'+model+'_0_255.npy')

batch = 0   # range in [0, 22]
ssim = []
psnr = []

if batch == 22:
    for img in range(100 * batch, originals.shape[0]):
        ssim.append(tf.keras.backend.eval(tf.image.ssim(tf.convert_to_tensor(originals[img, :, :, :], dtype=tf.float32),
                                                        tf.convert_to_tensor(colorized[img, :, :, :], dtype=tf.float32),
                                                        max_val=255)))

        psnr.append(tf.keras.backend.eval(tf.image.psnr(tf.convert_to_tensor(originals[img, :, :, :], dtype=tf.float32),
                                                        tf.convert_to_tensor(colorized[img, :, :, :], dtype=tf.float32),
                                                        max_val=255)))
        print(img)
else:
    for img in range(100*batch, 100*(batch+1)):
        ssim.append(tf.keras.backend.eval(tf.image.ssim(tf.convert_to_tensor(originals[img,:,:,:], dtype=tf.float32),
                                                           tf.convert_to_tensor(colorized[img,:,:,:], dtype=tf.float32),
                                                           max_val=255)))

        psnr.append(tf.keras.backend.eval(tf.image.psnr(tf.convert_to_tensor(originals[img,:,:,:], dtype=tf.float32),
                                                           tf.convert_to_tensor(colorized[img,:,:,:], dtype=tf.float32),
                                                           max_val=255)))
        print(img)


if batch != 0:
    with open('../resources/metrics_batches/metrics_ssim_' + model + '_batch_' + str(batch-1) + '.txt', 'rb') as fp:
        btc = pickle.load(fp)
    ssim = btc + ssim

    with open('../resources/metrics_batches/metrics_psnr_' + model + '_batch_' + str(batch-1) + '.txt', 'rb') as fp:
        btc = pickle.load(fp)
    psnr = btc + psnr

    os.remove('../resources/metrics_batches/metrics_ssim_' + model + '_batch_' + str(batch - 1) + '.txt')
    os.remove('../resources/metrics_batches/metrics_psnr_' + model + '_batch_' + str(batch - 1) + '.txt')

with open('../resources/metrics_batches/metrics_ssim_' + model + '_batch_' + str(batch) + '.txt', 'wb') as fp:
    pickle.dump(ssim, fp)
with open('../resources/metrics_batches/metrics_psnr_' + model + '_batch_' + str(batch) + '.txt', 'wb') as fp:
    pickle.dump(psnr, fp)


'''
# Compute statistics:
with open('../resources/metrics_batches/metrics_ssim_' + model + '_batch_' + str(22) + '.txt', 'rb') as fp:
    ssim = pickle.load(fp)
mean_ssim = sum(ssim)/len(ssim)
std_ssim = np.sqrt(sum([((x - mean_ssim) ** 2) for x in ssim]) / len(ssim))
metrics_ssim = dict(max=max(ssim), idx_max=ssim.index(max(ssim)),
                    min=min(ssim), idx_min=ssim.index(min(ssim)),
                    mean=mean_ssim, std=std_ssim)
print(metrics_ssim)

with open('../resources/metrics_batches/metrics_psnr_' + model + '_batch_' + str(22) + '.txt', 'rb') as fp:
    psnr = pickle.load(fp)
mean_psnr = sum(psnr)/len(psnr)
std_psnr = np.sqrt(sum([((x - mean_psnr) ** 2) for x in psnr]) / len(psnr))
metrics_psnr = dict(max=max(psnr), idx_max=psnr.index(max(psnr)),
                    min=min(psnr), idx_min=psnr.index(min(psnr)),
                    mean=mean_psnr, std=std_psnr)
print(metrics_psnr)

with open('../resources/metrics_ssim_'+model+'.txt', 'w') as f:
    f.write(json.dumps(str(metrics_ssim)))

with open('../resources/metrics_psnr_'+model+'.txt', 'w') as f:
    f.write(json.dumps(str(metrics_psnr)))
'''