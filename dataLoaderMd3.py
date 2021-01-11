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

print(os.getcwd())

#Read CSV File
df = pd.read_csv('myCSV.csv')
#Get CSV File Length
dfLen = len(df.index)+1

#TensorArr
tensorArr = []

#empty Stack tensor
#stackTensor = torch.empty(2,3)


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

(xtrain, xtest, ytrain, ytest) = (train_test_split(X, y, 
                                test_size=0.25, random_state=42))




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
        self.X = self.X[index]
        #Accesing every instance of NpLabelList
        self.Y = self.Y[index]
        #return 
        





    def loader(self, imglbl, imgPath, imgFrames):
        self.X = imglbl
        self.y = imgPath
        self.z = imgFrames
        #print(self.y)
        imgDir = os.chdir(self.y)
        #print(os.getcwd())
        #stackArr counter
        #counter = 0
        #print(imgFramesList[self.counter])
#        '''
        #tensorArr = []
        for x in range(imgFrames+1):
            image = Image.open('frame'+str(x)+'.jpg')
            new_image = image.resize((400, 400))
            #image = np.array(image)
            new_image2 = self.transform(new_image)
            #print(type(image))
            #print("Frame"+str(x))
            tensorArr.append(new_image2)
            
            #Numpy Array of the tensors
            npTensorArr = np.array(tensorArr)
            label = self.X
            
        #returning npTensorArr 
        return npTensorArr, new_image2, label
        #return tensorArr, new_image2, label
#        '''


#Intialise signData class
#train_data = signData(xtrain, ytrain, transform = 1)
#test_data = signData(xtest, ytest, transform = 1)
tensorList = []
labelList = []
for i in range(dfLen-1):
    #print(imglblList[i],imgPathList[i])
    data = signData(transform=1)
    
    #Function call to loader
    data.loader(imglblList[i], imgPathList[i], imgFramesList[i])
    #print("Sample '"+str(imglblList[i])+"' done")
    stackTensor = torch.stack(tensorArr)
    
    #Numpy Array of the TensorList
    tensorList.append(stackTensor)
    npTensorList = np.array(tensorList)
    
    #Numpy Array of the LabelList
    labelList.append(data.X)
    npLabelList = np.array(labelList)

#print(stackTensor)
#print(stackTensor.shape)
#print(tensorList)
#print(len(tensorList[0]))
#print(tensorList[0])
#print(len(labelList))
#print(labelList[0])



#Intialise DataLoader class
combList = []
for i in range(len(labelList)):
    combList.append([tensorList[i],labelList[i]])

#print(combList[0])
#Give 2 zeros to access Tensors
#print(combList[0][0])
#print(combList[0][1])    

train_loader = torch.utils.data.DataLoader(combList[0][0], batch_size=5, shuffle=False)
#testloader = torch.utils.data.DataLoader(combList, batch_size=32, shuffle=False)
