from utils_alexnet import *
import numpy as np
import cv2

orig_test = '../img/colorized/chromagan/finetuning_test/'
onlyfiles_test = [f for f in listdir(orig_test) if isfile(join(orig_test, f))]

data_test = np.empty((len(onlyfiles_test), 256, 256, 3))
i = 0
for files in onlyfiles_test:
    img = cv2.imread(join(orig_test,files))
    img_resized = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    #cv2.imwrite('../img/colorized/chromagan/'+files, img_resized)
    data_test[i, :, :, :] = img_resized
    i += 1

# Normalization in range [0,1]
data_test_01 = np.empty(data_test.shape)
for i in range(0, data_test.shape[0]):
    data_test_01[i, :, :, 0] = (data_test[i, :, :, 0] - data_test[i, :, :, 0].min()) / (data_test[i, :, :, 0].max() - data_test[i, :, :, 0].min())
    data_test_01[i, :, :, 1] = (data_test[i, :, :, 1] - data_test[i, :, :, 1].min()) / (data_test[i, :, :, 1].max() - data_test[i, :, :, 1].min())
    data_test_01[i, :, :, 2] = (data_test[i, :, :, 2] - data_test[i, :, :, 2].min()) / (data_test[i, :, :, 2].max() - data_test[i, :, :, 2].min())

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
        test_labels = np.concatenate((test_labels, np.full(50, l)), axis = 0)

data_test_scaled = data_test_scaled.transpose((0, 3, 1, 2))
test_dataset = MyDataset(list(data_test_scaled), test_labels)

BATCH_SIZE = 64
test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# USE TENSORFLOW IMPLEMENTATION FOR TESTING

alexnet = torchvision.models.alexnet(pretrained=True)


with open('../resources/Test_Results_Chromagan.txt', 'w') as f:
    f.write("Test loss:" + str(test_loss) + '\n')
    f.write("Test acc:" + str(test_acc) + '\n')
