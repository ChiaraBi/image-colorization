import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

import json
from os import listdir
from os.path import isfile, join

'''
Classification using the pretrained AlexNet without fine tuning.
'''

alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()

model = 'orig'  # orig, BW, chromagan, dahl, siggraph, su, zhang
if model == 'orig' or model == 'BW':
    test_path = '../img/original/finetuning_test/'
else:
    test_path = '../img/colorized/'+model+'/finetuning_test/'

# Read the ImageNet categories:
with open("../resources/img_classes/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

failed_files = {}
if model != 'orig':
    with open('../resources/failed_files.txt') as f:
        failed_files = json.loads(f.read())

onlydirectories = [f for f in listdir(test_path) if not isfile(join(test_path, f))]

if model == 'BW':
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# CALCULATION OF ACCURACY
accuracy = 0
count = 0
idx = 0
for d in onlydirectories:
    onlyfiles = [f for f in listdir(test_path + d) if isfile(join(test_path + d, f))]
    for i in onlyfiles:
        idx += 1
        if i not in failed_files.keys():
            count += 1
            if count % 100 == 0:
                print(count)
            filename = join(test_path + d, i)
            input_image = Image.open(filename)
            try:
                input_tensor = preprocess(input_image)
                input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

                with torch.no_grad():
                    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
                    output = alexnet(input_batch)

                # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                prob, catid = torch.topk(probabilities, 1)
                if categories[catid[0]] == d:
                    accuracy += 1
            except:
                failed_files[i] = filename
                count -= 1
                continue

print("count", count)    # 585
print("idx", idx)        # 600
                         # -> 15 failed files

accuracy = accuracy / count
print('Accuracy on '+model+' images: ', accuracy)

if model == 'orig':
    with open('../resources/failed_files.txt', 'w') as f:
        f.write(json.dumps(failed_files))

with open('../resources/classification/Test_'+model+'.txt', 'w') as f:
    f.write("Test acc:" + str(accuracy) + '\n')