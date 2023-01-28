# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 20:11:54 2022

@author: Asus
"""

import cv2
img=cv2.imread('2.jpg')
#cap=cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)

classNames=[]
classFile='/media/revanth/Data/Toshiba/rasbbery/coco.names.txt'
with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
    print(classNames)
configpath='/media/revanth/Data/Toshiba/rasbbery/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt.txt'
weightspath='/media/revanth/Data/Toshiba/rasbbery/frozen_inference_graph.pb'

net=cv2.dnn_DetectionModel(weightspath,configpath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


classIds,confs,bbox=net.detect(img,confThreshold=0.5)
print(classIds,bbox)
for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox) :
    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
cv2.imshow("output",img)
cv2.waitKey(1000)