import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision

from utils_alexnet import *

'''
Feature Extraction on original ImageNet images.
- model='orig' -> the last layer of the model is trained and tested on a subset of the ImageNet dataset.
- model='dahl' (or 'chromagan'/'siggraph'/'su'/'zhang') -> the model is tested on the colorized images of the respective
models.
'''

def get_labels(labels, num):
    final_labels = None
    for label in labels:
        if final_labels is None:
            final_labels = np.full(num, label)
        else:
            final_labels = np.concatenate((final_labels, np.full(num, label)), axis=0)
    return final_labels


lab = ["cassette player", "chain saw", "church", "gas pump", "golf ball", "French horn", "parachute",
          "Rhodesian ridgeback", "Samoyed", "English springer", "tench", "garbage truck"]
labels = [482, 491, 497, 571, 574, 566, 701, 159, 258, 217, 0, 569]

train_labels = get_labels(labels, 150)
test_labels = get_labels(labels, 50)

# Load scaled data
model = 'zhang'  # BW, orig, chromagan, dahl, siggraph, su, zhang
                 # baseline_cartoon, baseline_without_cartoon

data_train_scaled = np.load('../resources/feature_extraction/normalized data/data_train_scaled_orig.npy')
data_test_scaled = np.load('../resources/feature_extraction/normalized data/data_test_scaled_'+model+'.npy')

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
alexnet.double()

# Feature extraction:
device = torch.device('cpu')
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
alexnet = alexnet.to(device)

# Train the last FC layer:
if model == 'orig':
    # Freeze all layers except last Fully Connected layer:
    for parameter in alexnet.features.parameters():
        parameter.requires_grad = False
    for parameter in alexnet.classifier[:-1].parameters():
        parameter.requires_grad = False

    optimizer = optim.Adam(alexnet.parameters(), lr=1e-4)

    N_EPOCHS = 2
    train_losses, train_acc, valid_losses, valid_acc = model_training(N_EPOCHS, alexnet, train_iterator,
                                                                  valid_iterator, optimizer, criterion,
                                                                  device, 'alexnet_feat_extract.pt')

model_testing(alexnet, test_iterator, criterion, device, '../models/alexnet_feat_extract.pt')

test_loss, test_acc = evaluate(alexnet, test_iterator, criterion, device)

# Save results to files
with open('../resources/classification/feature_extraction/Test_'+model+'.txt', 'w') as f:
    f.write("Test loss:" + str(test_loss) + '\n')
    f.write("Test acc:" + str(test_acc) + '\n')

if model == 'orig':
    with open('../resources/classification/feature_extraction/Train_orig.txt', 'w') as f:
        f.write("Train loss:\n")
        f.writelines('\n'.join([str(i) for i in train_losses]))
        f.write("\nTrain acc:\n")
        f.writelines('\n'.join([str(i) for i in train_acc]))

    with open('../resources/classification/feature_extraction/Valid_orig.txt', 'w') as f:
        f.write("Valid loss:\n")
        f.writelines('\n'.join([str(i) for i in valid_losses]))
        f.write("\nValid acc:\n")
        f.writelines('\n'.join([str(i) for i in valid_acc]))
