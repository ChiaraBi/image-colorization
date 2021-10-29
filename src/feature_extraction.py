import matplotlib.pyplot as plt
import torchvision
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
from colorization.colorizers.util import *
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import copy
import cv2
from utils_alexnet import *

data_train_scaled = np.load('../resources/data_train_scaled.npy')
data_test_scaled = np.load('../resources/data_test_scaled.npy')

lab = ["cassette player", "chain saw", "church", "gas pump", "golf ball", "French horn", "parachute",
          "Rhodesian ridgeback", "Samoyed", "English springer", "tench", "garbage truck"]
labels = [482, 491, 497, 571, 574, 566, 701, 159, 258, 217, 0, 569]

train_labels = None
for l in labels:
    if train_labels is None:
        train_labels = np.full(150, l)
    else:
        train_labels = np.concatenate((train_labels, np.full(150, l)), axis = 0)

test_labels = None
for l in labels:
    if test_labels is None:
        test_labels = np.full(50, l)
    else:
        test_labels = np.concatenate((test_labels, np.full(50, l)), axis = 0)

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        #self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1, 2, 0))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


data_train_scaled = data_train_scaled.transpose((0, 3, 1, 2))
data_test_scaled = data_test_scaled.transpose((0, 3, 1, 2))

train_dataset = MyDataset(list(data_train_scaled), train_labels)
test_dataset = MyDataset(list(data_test_scaled), test_labels)

# Validation set:
train_percentage = 0.8
num_train_examples = int(len(train_dataset) * train_percentage)
num_valid_examples = len(train_dataset) - num_train_examples
train_dataset, valid_dataset = data.random_split(train_dataset, [num_train_examples, num_valid_examples])

# Create iterators:
BATCH_SIZE = 64
train_iterator = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

#for (x,y) in train_iterator:
#    print(x.shape) # torch.Size([64, 3, 256, 256])
#    print(y.shape) # torch.Size([64])

# Pre-trained model:
alexnet = torchvision.models.alexnet(pretrained=True)
#print(alexnet.classifier[-1]) # Linear(in_features=4096, out_features=1000, bias=True)
alexnet.double()

# Freeze all layers except last Fully Connected layer:
for parameter in alexnet.features.parameters():
  parameter.requires_grad = False
for parameter in alexnet.classifier[:-1].parameters():
  parameter.requires_grad = False

# Feature extraction:
device = torch.device('cpu')
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(alexnet.parameters(), lr=1e-4)
alexnet = alexnet.to(device)

# Train the last FC layer:
N_EPOCHS = 5
train_losses, train_acc, valid_losses, valid_acc = model_training(N_EPOCHS, alexnet, train_iterator,
                                                                  valid_iterator, optimizer, criterion,
                                                                  device, 'alexnet_feat_extract.pt')

model_testing(alexnet, test_iterator, criterion, device, 'alexnet_feat_extract.pt')

test_loss_BW, test_acc_BW = evaluate(alexnet, test_iterator, criterion, device)

with open('../resources/Test_Results_BW.txt', 'w') as f:
    f.write("Test loss:" + str(test_loss_BW) + '\n')
    f.write("Test acc:" + str(test_acc_BW) + '\n')

with open('../resources/Train_Results_BW.txt', 'w') as f:
    f.write("Train loss:\n")
    f.writelines('\n'.join([str(i) for i in train_losses]))
    f.write("\nTrain acc:\n")
    f.writelines('\n'.join([str(i) for i in train_acc]))

with open('../resources/Valid_Results_BW.txt', 'w') as f:
    f.write("Valid loss:\n")
    f.writelines('\n'.join([str(i) for i in valid_losses]))
    f.write("\nValid acc:\n")
    f.writelines('\n'.join([str(i) for i in valid_acc]))