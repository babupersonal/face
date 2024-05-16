import cv2
import numpy as np
import random
img = cv2.imread('babu.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
faceCascade = cv2.CascadeClassifier('face_detect.xml')
faceRect = faceCascade.detectMultiScale(gray,4,3)
print(len(faceRect))

for (x,y,w,h) in faceRect:
    count=0
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),10)
    count += 1  # 更新计数器
    label = f"Customer_ID:{random.randint(0, 255)} (X,Y)={x}, {y} Pass)"


    # 在矩形上方添加文本
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.7, (255, 255, 255), 10)


cv2.imshow('babu1',img)
cv2.waitKey(0) == ord('q')