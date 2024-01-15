import cv2
from OPENCV_MEDIAPIPE import handTrackingModule
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)
detector = handTrackingModule.handDetector(maxHands=1)
offset = 20
imgSize = 300

folder_train = "C:\\Users\\aruna\\PycharmProjects\\AdvancedComputerVision\\COMPUTER_VISION_ENGINEER\\sign_language_detection\\hand_sign_dataset\\train\\Z"
folder_val = "C:\\Users\\aruna\\PycharmProjects\\AdvancedComputerVision\\COMPUTER_VISION_ENGINEER\\sign_language_detection\\hand_sign_dataset\\val\\Z"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    #cropping image
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        #creating a white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        #overlaying the img white and img crop
        try:
            #imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop
            #overlaying image to the middle
            aspectRatio = h/w
            #fixing height
            if aspectRatio > 1:
                k = imgSize/h #constant
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
            #fixing width
            else:
                k = imgSize / w  # constant
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
        except:
            continue

        cv2.imshow("imageResize", imgResize)
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    #saving the image
    if key == ord("s"):
        counter+=1
        cv2.imwrite(f"{folder_train}/Image_{counter}.jpg", imgWhite)
        cv2.imwrite(f"{folder_val}/Image_{counter}.jpg", imgWhite)
        print(counter)
    if key == ord("q"):
        break
