import os
import csv

def image():
    imglbl = []
    imgPath = []
    tempPath = []
    imgFrm = []
    
    path = os.getcwd()
    newPath = "trainData"
    newPath = os.path.join(path, newPath)
    os.chdir(newPath)
    dirLen = len(os.listdir())
    imglbl.append(os.listdir())    
    os.chdir(path)    

    with open('myCSV.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(["Image_Label", "Image_Frames", "Image_Path"])
        for i in range(dirLen):
            x = imglbl[0][i]
            customPath = os.path.join(newPath, str(x))
            os.chdir(customPath)
            y = (os.listdir())
            for j in range(len(y)):
                imgPath.append(customPath + '\\' + y[j])
                tempPath.append(imgPath[-1])
                os.chdir(tempPath[-1])
                imgFrm.append(len(os.listdir())-1)                
                writer.writerow([str(imglbl[0][i]), imgFrm[-1], imgPath[-1]])
image()