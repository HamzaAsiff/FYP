import json
import numpy as np
import cv2
import os
import torchvision.io

def readVideo():

    glossList = []
    vidList = []
    bblst = []
    path = os.getcwd()
    content = json.load(open('WLASL_v0.3.json'))
    for ent in content:
        gloss = ent['gloss']
        instances = ent['instances']
        for inst in instances:
            split = inst['split']
            if split == "train":
                video = inst['video_id'] + ".mp4"
                x = inst['bbox']
                
                x1 = x[1]
                x2 = x[3]
                y1 = x[0]
                y2 = x[2]
                if os.path.isfile(str(video)):
                    
                    glossList.append(str(gloss))
                    vidList.append(str(video))
                    bblst.append([x1,x2,y1,y2])
    
    print("---------------------------")
    oldPath = os.getcwd()
    path = os.path.join(str(path),"Frames")
    #print(path)
#'''
    i = 0            
    for j in vidList: #use this to get the video number
        newPath = os.path.join(path, str(glossList[i]))
        print(newPath)
        if not os.path.exists(newPath):
            os.mkdir(newPath)    
        #print(j)
        #break
#'''
#'''
        vidcap = cv2.VideoCapture(j)
        os.chdir(newPath)
        print(os.getcwd())
        newPath2 = os.path.join(newPath, (j))
        #print(newPath2)
        if not os.path.exists(newPath2):
            os.mkdir(newPath2)
            os.chdir(newPath2)
        print(os.getcwd())
        #getting bbox of the vid
        t = bblst[i]
        x1 = t[0]
        x2 = t[1]
        y1 = t[2]
        y2 = t[3]
        success,image = vidcap.read()
        #os.chdir(oldPath)
        #print(os.getcwd())
        count = 0
        #break
#'''
#'''
        while success:
            image = image[x1:x2,y1:y2]
            cv2.imwrite("frame%d.jpg" % count, image)
            success,image = vidcap.read()
            count += 1
        os.chdir(oldPath)
        print(os.getcwd())
        i+=1
#'''
        print("-----------------------------------------------")
        #break                 
readVideo()
