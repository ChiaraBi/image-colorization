from os import listdir
from os.path import isfile, join

from utils_alexnet import *
import numpy as np
import cv2
import torch.nn as nn
import torchvision

'''
1. Dataset normalization with respect to the entire ImageNet.
2. Classification using the pretrained AlexNet without fine tuning.
'''

model = 'orig'  # orig, chromagan, dahl, siggraph, su, zhang
test_path = ''
if model == 'orig':
    test_path = '../img/original/finetuning_test_'
else:
    test_path = '../img/colorized/'+model+'/finetuning_test_'

onlyfiles_test = [f for f in listdir(test_path) if isfile(join(test_path, f))]

data_test = np.empty((len(onlyfiles_test), 256, 256, 3))
i = 0
for files in onlyfiles_test:
    img = cv2.imread(join(test_path, files))
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    #cv2.imwrite('../img/colorized/chromagan/'+files, img_resized)
    data_test[i, :, :, :] = img_resized
    i += 1

# Normalization in range [0,1]
data_test_01 = np.empty(data_test.shape)
for i in range(0, data_test.shape[0]):
    data_test_01[i, :, :, 0] = (data_test[i, :, :, 0] - data_test[i, :, :, 0].min()) / (data_test[i, :, :, 0].max() - data_test[i, :, :, 0].min())
    data_test_01[i, :, :, 1] = (data_test[i, :, :, 1] - data_test[i, :, :, 1].min()) / (data_test[i, :, :, 1].max() - data_test[i, :, :, 1].min())
    data_test_01[i, :, :, 2] = (data_test[i, :, :, 2] - data_test[i, :, :, 2].min()) / (data_test[i, :, :, 2].max() - data_test[i, :, :, 2].min())

# TODO: vedere come applicare transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
 )])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_test_scaled = np.empty(data_test_01.shape)
data_test_scaled[:, :, :, 0] = (data_test_01[:, :, :, 0] - mean[0]) / std[0]
data_test_scaled[:, :, :, 1] = (data_test_01[:, :, :, 1] - mean[1]) / std[1]
data_test_scaled[:, :, :, 2] = (data_test_01[:, :, :, 2] - mean[2]) / std[2]

labels = [482, 491, 497, 571, 574, 566, 701, 159, 258, 217, 0, 569]
test_labels = None
for l in labels:
    if test_labels is None:
        test_labels = np.full(50, l)
    else:
        test_labels = np.concatenate((test_labels, np.full(50, l)), axis=0)

data_test_scaled = data_test_scaled.transpose((0, 3, 1, 2))
test_dataset = MyDataset(list(data_test_scaled), test_labels)

BATCH_SIZE = 64
test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# USE TENSORFLOW IMPLEMENTATION FOR TESTING

alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.double()
device = torch.device('cpu')
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
alexnet = alexnet.to(device)

# TODO: vedere come fare l'evaluation di AlexNet
# model_testing(alexnet, test_iterator, criterion, device)
test_loss, test_acc = evaluate(alexnet, test_iterator, criterion, device)

with open('../resources/classification/Test_'+model+'.txt', 'w') as f:
    f.write("Test loss:" + str(test_loss) + '\n')
    f.write("Test acc:" + str(test_acc) + '\n')
