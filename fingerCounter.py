import cv2
import mediapipe as mp
import time
import os
from handTrackingModule import handDetector
import numpy as np

wCam, hCam = 640, 480 #width and height
pTime = 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = handDetector(detectionCon=0.7)

folderPath = "numbers"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imgPath in myList:
    image = cv2.imread(folderPath+"/"+imgPath)
    image = cv2.resize(image, (200, 200)) #resizing image
    print(folderPath+"/"+imgPath)
    overlayList.append(image)

print(len(overlayList))

#getting tips
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        fingers = []
        #thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #other 4 fingers
        for id in range(1, 5):
            #index finger
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        # overlaying the images from the folder
        """h, w, c = overlayList[totalFingers].shape  # to put images always on the corner
        img[0:h, 0:w] = overlayList[totalFingers]"""
        #overlaying numbers
        cv2.rectangle(img, (0, 0), (200, 200), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 150), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 20)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # displaying FPS on screen
    cv2.putText(img, str(int(fps)), (10, 250), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    # for exit if you press 'q' it will break the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
