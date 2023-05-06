# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# #### Importing Required Library

# %%
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch import optim
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
#import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from sklearn import metrics
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms.functional import resize
import torchvision.transforms.functional as F

# %% [markdown]
# #### Mounting my drive to access the dataset

#from google.colab import drive
#drive.mount('/content/drive')
#from torchvision.transforms import functional as F

F.to_pil_image(dataset_train.__getitem__(2)[0])

# #### Providing datapath
data_path = 'C:\\Users\\Raushan\\Downloads\\Code\\pytorch\\emotion\\data\\train\\'

batch_size = 1

# %% [markdown]
# #### Defining data transforms and data loader

# %%
transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(28, interpolation=3),
        transforms.ToTensor()
    ])


dataset_train = datasets.ImageFolder(os.path.join(data_path + "sample"), transform=transform)



train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True
)

train_loader.dataset.classes

print(f"Data loaded with {len(dataset_train)} train imgs.")


# #### Defining Model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Defining another 2D convolution layer
            nn.Conv2d(10, 15, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(15, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
      )

        self.linear_layers = nn.Sequential(
            
          nn.Linear(10*3*3, 7)
      )

  # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model = Net()
print(model)

# #### Setting up the optimizer and loss function

# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
# defining the loss function
criterion = nn.CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# #### Define epoch and train the model

#for images, labels in train_loader:
#plt.imshow(images)

epoch = 20

model.train()


for i in range(epoch):
    running_loss = 0
    for images, labels in train_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        #This is where the model learns by backpropagating
        loss.backward()

        #And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(train_loader)))

# %% [markdown]
# #### Setting model in eval mode and checking model accuracy

# %%
model.eval()


# %%
# getting predictions on training set and measuring the performance




l = []
p = []
for images,labels in train_loader:
    for i in range(len(labels)):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        img = images[i].view(1, 3, 224, 224)
        with torch.no_grad():
            logps = model(img)


        ps = torch.exp(logps)
        probab = list(ps.cpu()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu()[i]
        true_label = true_label.item()
        l.append(true_label)
        p.append(pred_label)

print("Number Of Images Tested =", len(l))
print("\nModel Accuracy = {:.3f}%".format(100*metrics.accuracy_score(l, p)) )

# %% [markdown]
# #### Saving model for future use

# %%
torch.save(model.state_dict(), '/content/drive/MyDrive/Dataset/drink_model.pth')

# %% [markdown]
# #### Testing the saved model (no need to run)

# %%
test_model = Net()
test_model.load_state_dict(torch.load('/content/drive/MyDrive/Dataset/drink_model.pth'))
if torch.cuda.is_available():
    test_model = test_model.cuda()
test_model.eval()


# %%
# getting predictions on test set and measuring the performance
l = []
p = []
for images,labels in train_loader:
    for i in range(len(labels)):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        img = images[i].view(1, 3, 224, 224)
        with torch.no_grad():
            logps = test_model(img)


        ps = torch.exp(logps)
        probab = list(ps.cpu()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu()[i]
        true_label = true_label.item()
        l.append(true_label)
        p.append(pred_label)

print("Number Of Images Tested =", len(l))
print("\nModel Accuracy = {:.3f}%".format(100*metrics.accuracy_score(l, p)) )

# %% [markdown]
# #### Function to plot the training images

# %%
from matplotlib.pyplot import imshow
def display_image(img):
  img = img.cpu()
  img = img.squeeze(0)
  img = img.permute(1,2,0)
  img = (img-img.min())/(img.max() - img.min())
  img = np.array(img*255, dtype=np.uint8)
  img = Image.fromarray(img)
  imshow(img)


# %%
# getting predictions on test set and measuring the performance
l = []
p = []
for images,labels in train_loader:
  for i in range(len(labels)):
    if torch.cuda.is_available():
      images = images.cuda()
      labels = labels.cuda()
    img = images[i].view(1, 3, 224, 224)
    with torch.no_grad():
      logps = test_model(img)


    ps = torch.exp(logps)
    probab = list(ps.cpu()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.cpu()[i]
    true_label = true_label.item()
    if(true_label == 2):
      display_image(img)
      break
  break









