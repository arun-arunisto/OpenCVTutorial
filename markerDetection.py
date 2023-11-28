import cv2
import numpy as np
from cv2 import aruco

#for detecting markers first we need to call the module for seeting the size of the marker
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

#object for detecting markers
param_markers = aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    #need to convert the img to grayscale for detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, reject = aruco.detectMarkers(
        gray_img, marker_dict, parameters=param_markers
    )
    #print(marker_ids)
    #we are going to draw the ids and corners into the detected marker
    if marker_corners:
        for ids, corners in zip(marker_ids, marker_corners):
            #drawing the lines
            cv2.polylines(img, [corners.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
            #showing the ids/corners
            #print(corners.shape) #the output will be (1, 4, 2)
            #print(corners) # the output will be like this [[[467.  56.] [508. 279.] [272. 308.] [243.  74.]]]
            #reshaping to (4, 2)
            corners = corners.reshape(4, 2) #this will convert to (4, 2)
            #print(corners.shape)
            #print(corners) #the output will be like [[467.  56.] [508. 279.] [272. 308.] [243.  74.]]
            #print(corners[0].ravel())
            corners = corners.astype(int)
            print(corners)
            top_right = corners[0].ravel()
            #to draw the image id in top-right corner
            cv2.putText(img, f"id: {ids[0]}", top_right, cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
