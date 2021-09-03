import tensorflow.compat.v1 as tf
import skimage.transform
from skimage.io import imsave, imread

import os
from os import listdir
from os.path import isfile, join

tf.disable_v2_behavior()


def load_image(path):
    img = imread(path)
    # crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    img = skimage.transform.resize(crop_img, (224, 224))
    # desaturate image
    return (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3.0


with open("../models/colorize.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
grayscale = tf.placeholder(tf.float32, shape=(1, 224, 224, 1))
tf.import_graph_def(graph_def, input_map={"grayscale": grayscale}, name='')

imageNet_input_dir = '../img/original/ImageNet/'
imageNet_output_dir = '../img/colorized/dahl/ImageNet/'


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
        try:
            img = load_image(join(imageNet_input_dir + d, i)).reshape(1, 224, 224, 1)
        except:
            print("Failed to colorize: " + join(imageNet_input_dir + d, i))
            continue
        with tf.Session() as sess:
            inferred_rgb = sess.graph.get_tensor_by_name("inferred_rgb:0")
            inferred_batch = sess.run(inferred_rgb, feed_dict={grayscale: img})
            out_path = join(imageNet_output_dir + d, i)
            if not os.path.exists(imageNet_output_dir + d):
                os.makedirs(imageNet_output_dir + d)
            imsave(out_path, inferred_batch[0])
