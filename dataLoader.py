#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv('myCSV.csv')
X = df.Image_Path.values
y = df.Image_Label.values


# In[4]:


(xtrain, xtest, ytrain, ytest) = (train_test_split(X, y, 
                                test_size=0.25, random_state=42))


# In[5]:


class signData(Dataset):
    def __init__(self, imglbl, imgPath, transform = None):
        self.X = imglbl
        self.y = imgPath
        if transform == 1:
            self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.5], std=[0.5])])
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = Image.open(self.X[i])
        image = self.transform(image=np.array(image))
        label = self.y[i]
        
        return image, torch.tensor(label, dtype = torch.long)

train_data = signData(xtrain, ytrain, transform = 1)
test_data = signData(xtest, ytest, transform = 1)

train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
testloader = DataLoader(test_data, batch_size=32, shuffle=False)

