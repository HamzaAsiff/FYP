#Required Libraries and Dependencies
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
from torch.utils.data import Dataset, DataLoader, TensorDataset

#Change Directory
print(os.getcwd())

#Read CSV File
df = pd.read_csv('myCSV.csv')
#Get CSV File Length
dfLen = len(df.index)+1

#Arrays Declarations
tensorArr = []
tensorList = []
labelList = []
label = ""

#Get Img Labels
imglblList = []
for column in df[['Image_Label']]:
    imglbl = df[column]
#    print('Colunm Name : ', column)
#    print('Column Contents : \n', imglbl.values)
for i in imglbl.values:
    imglblList.append(i)

#Get Img Frames    
imgFramesList = []
for column in df[['Image_Frames']]:
    imgFrames = df[column]
#    print('Colunm Name : ', column)
#    print('Column Contents : \n', imgFrames.values)
for i in imgFrames.values:
    imgFramesList.append(i)
    
#Get Img Path
imgPathList = []
for column in df[['Image_Path']]:
    imgPath = df[column]
#    print('Colunm Name : ', column)
#    print('Column Contents : \n', imgPath.values)
for i in imgPath.values:
    imgPathList.append(i)

X = df.Image_Path.values
y = df.Image_Label.values




#Custom Dataloader
class signData(Dataset):
    def __init__(self, X, Y, transform = None):
        #X is TensorList as argument
        self.X = X
        #Y is LabelList as argument
        self.Y = Y
        
        if transform == 1:
            self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.5], std=[0.5])])
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):
        #Accessing every instance of NpTensorArr
        imageStack = self.X[index]
        #Accesing every instance of NpLabelList
        label = self.Y[index]
        return imageStack, imageStack.size, label 

#Use my_collate for Image Batching and normalizing image sizes
def my_collate(batch):
    data = [item[0] for item in batch]
    data_size = torch.LongTensor(data)
    target = [item[1] for item in batch]
    target_size = torch.LongTensor(target)
    return data, data_size, target, target_size    

#Transform Images
trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.5], std=[0.5])])

#Open, Resize and Transform Images
def loader(imglbl, imgPath, imgFrames):
        imgDir = os.chdir(imgPath)
        for x in range(imgFrames+1):
            image = Image.open('frame'+str(x)+'.jpg')
            new_image = image.resize((400, 400))
            new_image2 = trans(new_image)
            tensorArr.append(new_image2)
        
        label = imglbl    
        return tensorArr, new_image2, label

#Image output and labels from loader will be stacked into a tensor
for i in range(dfLen-1):
    #Function call to loader
    loader(imglblList[i], imgPathList[i], imgFramesList[i])
 
    stackTensor = torch.stack(tensorArr)
    
    #Numpy Array of the TensorList
    tensorList.append(stackTensor)
    
    #Numpy Array of the LabelList
    labelList.append(label)

#Intialise signData class
Data = signData(tensorList, labelList, transform=1)
train_arg = dict(batch_size=5, shuffle=False, collate_fn = my_collate) 
train_loader = DataLoader(Data, **train_arg)
#testloader = torch.utils.data.DataLoader(combList, batch_size=32, shuffle=False)