from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Reshape
from keras.models import Model
from keras.models import model_from_json

from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from skimage import io, color
from os import listdir
import numpy as np

dim = 256

model = 'su'
original_dir = '../img/original/filtering/new_names/'
model_dir = '../img/colorized/'+model+'/filtered/cartoonized/'
results_dir = '../img/colorized/'+model+'/cartoon/'

results = np.empty((dim, dim, 3))
i = 0
for i in range(1,18):
    i += 1
    print(i)

    original_rgb = img_to_array(load_img(original_dir + 'img ('+str(i)+').JPEG', target_size=(dim, dim)))
    original_rgb = 1.0 / 255 * original_rgb
    original_lab = color.rgb2lab(original_rgb)

    cartoon_rgb = img_to_array(load_img(model_dir + 'img ('+str(i)+').PNG', target_size=(dim, dim)))
    cartoon_rgb = 1.0 / 255 * cartoon_rgb
    cartoon_lab = color.rgb2lab(cartoon_rgb)

    results[:, :, 0] = original_lab[:, :, 0]
    results[:, :, 1] = cartoon_lab[:, :, 1]
    results[:, :, 2] = cartoon_lab[:, :, 2]
    results = color.lab2rgb(results)
    io.imsave(results_dir + 'img ('+str(i)+').PNG', results)

