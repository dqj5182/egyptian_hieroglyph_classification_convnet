import os, os.path
import shutil, sys
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib
from torchvision.utils import save_image

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

from pathlib import Path
downloads_path = str(Path.home() / "Downloads")

data_dir = downloads_path + '/EgyptianHieroglyphDataset4/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')

classes = []

for filename in os.listdir(train_dir):
    if filename == '.DS_Store':
        pass
    else:
        classes.append(filename)

classes.sort()

print("Our classes:", classes)

data_transform_main = transforms.Compose([
                            transforms.ToTensor(),
                            #transforms.RandomApply([transforms.RandomHorizontalFlip()]),
                            #transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0)]),
                            #transforms.RandomRotation(degrees=(-30, 30)),
                            #transforms.RandomHorizontalFlip(p = 0.5),
                            #transforms.RandomAffine(degrees = 0, translate=(.1, .1)),
                            #transforms.RandomErasing(p=0.1, scale=(0.05, 0.05), ratio=(0.3, 3.3), value=0, inplace=False),
                            #transforms.RandomApply([transforms.ColorJitter()]),
                            transforms.Normalize((0.5,), (0.5,))
                        ])

# Generate more training data from data augmentation
data_transform_more = transforms.Compose([
                            transforms.ToTensor(),
                            #transforms.RandomApply([transforms.RandomHorizontalFlip()]),
                            transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0)]),
                            #transforms.RandomRotation(degrees=(-30, 30)),
                            #transforms.RandomHorizontalFlip(p = 0.5),
                            transforms.RandomAffine(degrees = 0, translate=(.1, .1)),
                            transforms.RandomErasing(p=0.1, scale=(0.05, 0.05), ratio=(0.3, 3.3), value=0, inplace=False),
                            #transforms.RandomApply([transforms.ColorJitter()]),
                            transforms.Normalize((0.5,), (0.5,))
                        ])

data_transform_train = transforms.Compose([transforms.ToTensor(),
                                           transforms.RandomHorizontalFlip(p = 0.5),
                                           #transforms.CenterCrop((10, 10)),
                                           transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
                                           #transforms.GaussianBlur(kernel_size=501),
                                           transforms.RandomRotation(degrees=(-30, 30)),
                                           transforms.RandomAffine(degrees = 0, translate=(.1, .1)),
                                           #transforms.RandomErasing(p=0.5, scale=(0.05, 0.05), ratio=(0.3, 3.3), value=0, inplace=False),
                                           transforms.Normalize((0.5,), (0.5,))])

data_transform_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.RandomRotation(degrees=(-5, 5)),
                                          transforms.RandomAffine(degrees = 0, translate=(.1, .1)),
                                          transforms.RandomApply([transforms.ColorJitter(brightness=(1, 1.2), contrast=(1, 1.5), saturation=(1, 1.5), hue=(0, 0.5))]),
                                          # transforms.RandomErasing(p=0.5, scale=(0.05, 0.05), ratio=(0.3, 3.3), value=0, inplace=False),
                                          transforms.Normalize((0.5,), (0.5,))])

data_transform_default = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.ImageFolder(train_dir, transform=data_transform_test)
test_data = datasets.ImageFolder(test_dir, transform=data_transform_test)

more_data = datasets.ImageFolder(train_dir, transform=data_transform_test)

# print out some data stats
print('Num training images: ', len(train_data))
print('Num test images: ', len(test_data))

# Generating more dataset using Data Augmentation
'''
img_num = 0
for _ in range(10):
  for img, label in more_data:
    print(img_num)
    download_directory = train_dir+ '/' + classes[label] + '/' + '100000' + str(img_num) + classes[label] + '.png'
    save_image(img, download_directory)
    img_num += 1
quit()
'''

# define dataloader parameters
batch_size = 20
num_workers=0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers, shuffle=True)

# Visualize some sample data

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])

# Load the pretrained model from pytorch
resnet50 = models.resnet50(pretrained=True)

# print out the model structure
print(resnet50)

#print(resnet152.fc.in_features)
#print(resnet152.fc.out_features)

# Freeze training for all "features" layers
for param in resnet50.parameters():
    param.requires_grad = False

import torch.nn as nn

n_inputs = resnet50.fc.in_features

# add last linear layer (n_inputs -> 5 flower classes)
# new layers automatically have requires_grad = True
last_layer = nn.Sequential(
                nn.Linear(n_inputs, 1024),
                nn.BatchNorm1d(num_features=1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(num_features=512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(num_features=256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(num_features=128),
                nn.ReLU(),
                nn.Linear(128, len(classes)))

resnet50.fc = last_layer

# if GPU is available, move the model to GPU
if train_on_gpu:
    resnet50.cuda()

# check to see that your last layer produces the expected number of outputs
# print(resnet152.fc.out_features)
#print(resnet152)

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

resnet50.fc.apply(init_weights)

import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (Adam) and learning rate = 0.001
# Adam is better than SGD
optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.001)

decayRate = 0.999
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# number of epochs to train the model
n_epochs = 200

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    # model by default is set to train
    for batch_i, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = resnet50(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()

        my_lr_scheduler.step()

        if batch_i % 20 == 19:  # print training loss every specified number of mini-batches
            print('Epoch %d, Batch %d loss: %.16f' %
                  (epoch, batch_i + 1, train_loss / 20))
            train_loss = 0.0

# track test loss
# over 5 flower classes
test_loss = 0.0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

resnet50.eval() # eval mode

# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = resnet50(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update  test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(len(target.data)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(len(classes)):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

# get sample outputs
output = resnet50(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images.cpu()[idx], (1, 2, 0)))
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))


