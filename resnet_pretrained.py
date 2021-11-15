import os, os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from pathlib import Path
from hieroglyph_data_preparation import load_data

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

##################################################Data Organization#####################################################
# The data_dir should be replaced with your own local path for hieroglyph image folder downloaded from my github
downloads_path = str(Path.home() / "Downloads")
data_dir = downloads_path + '/EgyptianHieroglyphDataset_Original/'

batch_size = 20
num_workers=0

train_loader, test_loader, classes = load_data(data_dir)

##################################################ResNet Model##########################################################
loss_function = "cross-entropy"
feature_extract = False
model_selection = "resnet-50"
optim_selection = "Adam"

def ConvNet_Model(model_selection, loss_function, optim_selection, classes, feature_extract = False, lr = 0.001, decayRate=0.999):
    if model_selection is "resnet-50":
        # Load the pretrained model from pytorch
        resnet50 = models.resnet50(pretrained=True)

        n_inputs = resnet50.fc.in_features
        last_layer = nn.Sequential(
                        nn.Linear(n_inputs, len(classes)))
        resnet50.fc = last_layer

        # if GPU is available, move the model to GPU
        if train_on_gpu:
            resnet50.cuda()

        if loss_function is "cross-entropy":
            # specify loss function (categorical cross-entropy)
            criterion = nn.CrossEntropyLoss()

        if optim_selection is "Adam":
            # specify optimizer (Adam) and learning rate = 0.001
            # Adam is better than SGD
            optimizer = optim.Adam(resnet50.parameters(), lr=lr)

        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        return resnet50, criterion, my_lr_scheduler, optimizer
########################################################################################################################

resnet50, criterion, my_lr_scheduler, optimizer = ConvNet_Model(model_selection, loss_function, optim_selection, classes)

##################################################Training##############################################################
# number of epochs to train the model
n_epochs = 300 # Use epoch of 300 if you want to get accuracy above 95%

def train_model(train_loader, optimizer, resnet50, criterion, my_lr_scheduler, n_epochs=300):
    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0

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

    return resnet50
########################################################################################################################

resnet50 = train_model(train_loader, optimizer, resnet50, criterion, my_lr_scheduler)

###################################################Testing##############################################################            
def test_model(classes, resnet50, test_loader, criterion):
    # track test loss
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
    
    return test_loss, class_correct, class_total
########################################################################################################################

test_loss, class_correct, class_total = test_model(classes, resnet50, test_loader, criterion)

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