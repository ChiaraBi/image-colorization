import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision

from utils_alexnet import *

'''
Finetuning on original birds and flowers images.
- model='orig' -> the last layer of the model is trained and tested on a subset of the ImageNet dataset.
- model='dahl' (or 'chromagan'/'siggraph'/'su'/'zhang') -> the model is tested on the colorized images of the respective
models.
'''

def get_labels(labels, num):
    final_labels = None
    i = 0
    for label in labels:
        if final_labels is None:
            final_labels = np.full(num[i], label)
            i += 1
        else:
            final_labels = np.concatenate((final_labels, np.full(num[i], label)), axis=0)
            i += 1
    return final_labels


labels = ["Bald Eagle", "Cuban Tody", "Fire Tailed Myzornis", "Flamingo", "Giant White Arum Lily",
       "Grape Hyacinth", "Hibiscus", "Nicobar Pigeon", "Ostrich", "Pink Robin", "Purple Coneflower",
       "Rose", "Touchan", "Water Lily"]

coded_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

num_train = [80, 80, 80, 80, 40, 60, 80, 80, 80, 80, 70, 80, 80, 80]
num_test = [20, 20, 20, 20, 16, 22, 20, 20, 20, 20, 15, 20, 20, 20]

train_labels = get_labels(coded_labels, num_train)
test_labels = get_labels(coded_labels, num_test)


# Load scaled data
model = 'zhang'  # BW, orig, chromagan, dahl, siggraph, su, zhang
                 # baseline_cartoon, baseline_without_cartoon

data_train_scaled = np.load('../resources/finetuning/normalized data/data_train_scaled_birdsflowers_orig.npy')
data_test_scaled = np.load('../resources/finetuning/normalized data/data_test_scaled_birdsflowers_'+model+'.npy')

# AlexNet requires images to be with shape: 3x256x256
data_train_scaled = data_train_scaled.transpose((0, 3, 1, 2))
data_test_scaled = data_test_scaled.transpose((0, 3, 1, 2))

train_dataset = MyDataset(list(data_train_scaled), train_labels)
test_dataset = MyDataset(list(data_test_scaled), test_labels)

# Validation set:
train_percentage = 0.8
num_train_examples = int(len(train_dataset) * train_percentage)
num_valid_examples = len(train_dataset) - num_train_examples
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [num_train_examples, num_valid_examples])

# Create iterators:
BATCH_SIZE = 32
train_iterator = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Pre-trained model:
alexnet = torchvision.models.alexnet(pretrained=True)
# print(alexnet.classifier[-1]) # Linear(in_features=4096, out_features=1000, bias=True)

# Finetuning:
device = torch.device('cpu')
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
alexnet = alexnet.to(device)

OUTPUT_DIM = len(labels)
alexnet.classifier[-1] = nn.Linear(alexnet.classifier[-1].in_features, OUTPUT_DIM)
alexnet.double()
print(alexnet.classifier[-1])

if model == 'orig':

    optimizer = optim.Adam(alexnet.parameters(), lr=1e-4)

    N_EPOCHS = 2
    train_losses, train_acc, valid_losses, valid_acc = model_training(N_EPOCHS, alexnet, train_iterator,
                                                                  valid_iterator, optimizer, criterion,
                                                                  device, 'alexnet_feat_extract_birdsflowers.pt')

model_testing(alexnet, test_iterator, criterion, device, '../models/alexnet_feat_extract_birdsflowers.pt')

test_loss, test_acc = evaluate(alexnet, test_iterator, criterion, device)

# Save results to files
with open('../resources/classification/finetuning/Test_'+model+'.txt', 'w') as f:
    f.write("Test loss:" + str(test_loss) + '\n')
    f.write("Test acc:" + str(test_acc) + '\n')

if model == 'orig':
    with open('../resources/classification/finetuning/Train_orig.txt', 'w') as f:
        f.write("Train loss:\n")
        f.writelines('\n'.join([str(i) for i in train_losses]))
        f.write("\nTrain acc:\n")
        f.writelines('\n'.join([str(i) for i in train_acc]))

    with open('../resources/classification/finetuning/Valid_orig.txt', 'w') as f:
        f.write("Valid loss:\n")
        f.writelines('\n'.join([str(i) for i in valid_losses]))
        f.write("\nValid acc:\n")
        f.writelines('\n'.join([str(i) for i in valid_acc]))
