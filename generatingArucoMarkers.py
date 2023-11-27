import cv2
from cv2 import aruco #module for aruco markers generating

#to specify the markers that we want
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) #4x4 aruco markers

#specifying marker size
MARKER_SIZE = 400 #pixels

#we are going to generate markers
"""
marker_image = aruco.generateImageMarker(marker_dict, 0, MARKER_SIZE)
cv2.imshow("img", marker_image)
cv2.waitKey(0)
"""
#we are going to use for loop to generate multiple markers at one time
for id in range(5): # we are going to generate 5 markers
    marker_image = aruco.generateImageMarker(marker_dict, id, MARKER_SIZE)
    #saving generated markers to the markers folder
    cv2.imwrite(f"markers/marker-id-{id}.png", marker_image)

print("Successfully Generated")

