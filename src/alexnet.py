import torch
from PIL import Image
import torchvision
from torchvision import transforms

import json
import os
from os import listdir
from os.path import isfile, join

model = torchvision.models.alexnet(pretrained=True)
model.eval()

imageNet_original_dir = '../img/original/ImageNet/'
imageNet_colorized_dir = '../img/colorized/zhang/ImageNet/'
imageNet_colorized_dir = '../img/colorized/dahl/ImageNet/'

# Read the categories
with open("../imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

onlydirectories = [f for f in listdir(imageNet_original_dir) if not isfile(join(imageNet_original_dir, f))]

# ORIGINAL IMAGES ACCURACY

accuracy = 0
count = 0
failed_files = {}
for d in onlydirectories:
    onlyfiles = [f for f in listdir(imageNet_original_dir + d) if isfile(join(imageNet_original_dir + d, f))]
    for i in onlyfiles:
        count += 1
        if count % 100 == 0:
            print(count)
        filename = join(imageNet_original_dir + d, i)
        input_image = Image.open(filename)
        try:
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

            with torch.no_grad():
                # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
                output = model(input_batch)

            # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            prob, catid = torch.topk(probabilities, 1)
            if categories[catid[0]] == d:
              accuracy += 1
        except:
            failed_files[i] = filename
            count -= 1
            continue

original_accuracy = accuracy/count
print('Accuracy on original images: ', original_accuracy)

with open('failed_files.txt', 'w') as convert_file:
    convert_file.write(json.dumps(failed_files))


#############################
# COLORIZED IMAGES ACCURACY #
#############################

# uncomment if you want to load the failed files from file
# failed_files = {}
# with open('failed_files.txt', 'r') as f:
#     failed_files = json.load(f)

accuracy = 0
count = 0
failed_files_c = {}
for d in onlydirectories:
    onlyfiles = [f for f in listdir(imageNet_colorized_dir + d) if isfile(join(imageNet_colorized_dir + d, f))]
    for i in onlyfiles:
        if i not in failed_files.keys():
            count += 1
            if count % 100 == 0:
                print(count)
            filename = join(imageNet_colorized_dir + d, i)
            input_image = Image.open(filename)
            try:
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                input_tensor = preprocess(input_image)
                input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

                with torch.no_grad():
                    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
                    output = model(input_batch)

                # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                prob, catid = torch.topk(probabilities, 1)
                if categories[catid[0]] == d:
                  accuracy += 1
            except:
                failed_files_c[i] = filename
                count -= 1
                continue

colorized_accuracy = accuracy/count
print('Accuracy on colorized images: ', colorized_accuracy)

with open('failed_files_c.txt', 'w') as convert_file:
    convert_file.write(json.dumps(failed_files_c))
