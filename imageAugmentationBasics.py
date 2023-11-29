import cv2
import numpy as np
from cv2 import aruco

#function for image augmentation
def image_augmentation(frame, src_image, dst_points):
    #source image height and width
    src_h, src_w = src_image.shape[:2]
    #current frame's height and width
    frame_h, frame_w = frame.shape[:2]
    #creating a mask for the same size of the frame
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    #setting the corner points in order (top-right[0, 0], top-left[src_w, 0], bottom-left[src_w, src_h], bottom-right[0, src_h])
    src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])

    #after this finding homography to place the image on aruco marker
    H, _ = cv2.findHomography(srcPoints=src_points, dstPoints=dst_points)
    #wrapping the image in aruco marker
    warp_image = cv2.warpPerspective(src_image, H, (frame_w, frame_h))
    #now the image on another frame of our current frame to check that use below code
    #cv2.imshow("warp image:",warp_image)
    #so we are going to display the image on our frame
    cv2.fillConvexPoly(mask, dst_points, 255)
    results = cv2.bitwise_and(warp_image, warp_image, frame, mask=mask)



marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
param_markers = aruco.DetectorParameters()

aug_img = cv2.imread("images/1.png")
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, reject = aruco.detectMarkers(
        gray_img, marker_dict, parameters=param_markers
    )

    if marker_corners:
        for ids, corners in zip(marker_ids, marker_corners):
            cv2.polylines(img, [corners.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            print(f"images/{ids[0]}.png")
            image_augmentation(img, cv2.imread(f"images/{ids[0]}.png"), corners)

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
