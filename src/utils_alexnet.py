from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed
import time

'''
Useful functions for dealing with image loading and pre-processing.
Useful functions for training, testing and evaluating AlexNet.
'''

def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if (out_np.ndim == 2):
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


def resize_img(img, HW=(256, 256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


def preprocess_img(img_rgb_orig, HW=(256, 256), resample=3):
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
    tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]

    return (tens_orig_l, tens_rs_l)


def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if (HW_orig[0] != HW[0] or HW_orig[1] != HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    # Train mode
    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y
        # Set gradients to zero
        optimizer.zero_grad()

        # Make Predictions
        y_pred = model(x)

        # Compute loss
        loss = criterion(y_pred, y)

        # Compute accuracy
        acc = calculate_accuracy(y_pred, y)

        # Backprop
        loss.backward()

        # Apply optimizer
        optimizer.step()

        # Extract data from loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    # Evaluation mode
    model.eval()

    # Do not compute gradients
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y

            # Make Predictions
            y_pred = model(x)

            # Compute loss
            loss = criterion(y_pred, y)

            # Compute accuracy
            acc = calculate_accuracy(y_pred, y)

            # Extract data from loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def predict(model, iterator, device):
    # Evaluation mode
    model.eval()

    labels = []
    pred = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_pred = model(x)

            # Get label with highest score
            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)

            labels.append(y.cpu())
            pred.append(top_pred.cpu())

    labels = torch.cat(labels, dim=0)
    pred = torch.cat(pred, dim=0)

    return labels, pred


def model_training(n_epochs, model, train_iterator, valid_iterator, optimizer, criterion, device,
                   model_name='best_model.pt'):
    # Initialize validation loss
    best_valid_loss = float('inf')

    # Save output losses, accs
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    # Loop over epochs
    for epoch in range(n_epochs):
        start_time = time.time()
        # Train
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        # Validation
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Save model
            torch.save(model.state_dict(), '../models'+model_name)
        end_time = time.time()

        print(f"\nEpoch: {epoch + 1}/{n_epochs} -- Epoch Time: {end_time - start_time:.2f} s")
        print("---------------------------------")
        print(f"Train -- Loss: {train_loss:.3f}, Acc: {train_acc * 100:.2f}%")
        print(f"Val -- Loss: {valid_loss:.3f}, Acc: {valid_acc * 100:.2f}%")

        # Save
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

    return train_losses, train_accs, valid_losses, valid_accs


def model_testing(model, test_iterator, criterion, device, model_name='best_model.pt'):
    # Test model
    model.load_state_dict(torch.load(model_name))
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    print(f"Test -- Loss: {test_loss:.3f}, Acc: {test_acc * 100:.2f} %")


def calculate_accuracy(y_pred, y):
    """
    Compute accuracy from ground-truth and predicted labels.

    Input
    ------
    y_pred: torch.Tensor [BATCH_SIZE, N_LABELS]
    y: torch.Tensor [BATCH_SIZE]

    Output
    ------
    acc: float
      Accuracy
    """
    y_prob = F.softmax(y_pred, dim=-1)
    y_pred = y_pred.argmax(dim=1, keepdim=True)
    correct = y_pred.eq(y.view_as(y_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        # self.targets = targets
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
