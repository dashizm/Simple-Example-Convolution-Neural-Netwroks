#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn. functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


transform=transforms.ToTensor()


# In[3]:


train_data=datasets.MNIST(root='cnn_data',train=True, download=True, transform=transform)


# In[4]:


test_data=datasets.MNIST(root='/cnn_data',train=False, download=True, transform=transform)


# In[ ]:


train_data


# In[ ]:


test_data


# In[7]:


#create a small batch
train_loader=DataLoader(train_data,batch_size=10, shuffle=True)
test_loader=DataLoader(test_data,batch_size=10, shuffle=False)


# In[8]:


# Define our CNN Model
# Describe convoltion layer
conv1=nn.Conv2d(1,6,3,1)
conv2=nn.Conv2d(6,16,3,1)


# In[9]:


# Gran 1 MNIST
for i,(X_Train,y_train) in enumerate  (train_data):
    break


# In[ ]:


X_Train.shape


# In[11]:


x=X_Train.view(1,1,28,28)


# In[ ]:


# First Convolution
x=F.relu(conv1(x)) # Rectified Linear unit
x.shape # 1 is the number of image, 6 is the filter we asked, 26*26 is the image with padding


# In[ ]:


# pass through the pooling layer
x=F.max_pool2d(x,2,2) # Kernel of 2 and stride of 2
x.shape


# In[ ]:


# Second Convolution
x=F.relu(conv2(x))
x.shape # Again We didnt padding So, we loose 2 pixels


# In[ ]:


x=F.max_pool2d(x,2,2)
x.shape


# In[16]:


# Model Class
class ConvolutionalNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,6,3,1)
    self.conv2 = nn.Conv2d(6,16,3,1)
    # Fully Connected Layer
    self.fc1 = nn.Linear(5*5*16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, X):
    X = F.relu(self.conv1(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2
    # Second Pass
    X = F.relu(self.conv2(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2

    # Re-View to flatten it out
    X = X.view(-1, 16*5*5) # negative one so that we can vary the batch size

    # Fully Connected Layers
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)
    return F.log_softmax(X, dim=1)


# In[ ]:


# Instance of the model
torch.manual_seed(41)
model=ConvolutionalNetwork()
model


# In[18]:


#Loss Function Optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)


# In[ ]:


import time
start_time = time.time()

# Create Variables To Tracks Things
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# For Loop of Epochs
for i in range(epochs):
  trn_corr = 0
  tst_corr = 0


  # Train
  for b,(X_train, y_train) in enumerate(train_loader):
    b+=1 # start our batches at 1
    y_pred = model(X_train) # get predicted values from the training set. Not flattened 2D
    loss = criterion(y_pred, y_train) # how off are we? Compare the predictions to correct answers in y_train

    predicted = torch.max(y_pred.data, 1)[1] # add up the number of correct predictions. Indexed off the first point
    batch_corr = (predicted == y_train).sum() # how many we got correct from this batch. True = 1, False=0, sum those up
    trn_corr += batch_corr # keep track as we go along in training.

    # Update our parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    # Print out some results
    if b%600 == 0:
      print(f'Epoch: {i}  Batch: {b}  Loss: {loss.item()}')

  train_losses.append(loss)
  train_correct.append(trn_corr)


  # Test
  with torch.no_grad(): #No gradient so we don't update our weights and biases with test data
    for b,(X_test, y_test) in enumerate(test_loader):
      y_val = model(X_test)
      predicted = torch.max(y_val.data, 1)[1] # Adding up correct predictions
      tst_corr += (predicted == y_test).sum() # T=1 F=0 and sum away


  loss = criterion(y_val, y_test)
  test_losses.append(loss)
  test_correct.append(tst_corr)



current_time = time.time()
total = current_time - start_time
print(f'Training Took: {total/60} minutes!')

