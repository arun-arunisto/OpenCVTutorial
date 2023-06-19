#camera Accessing using OpenCV
"""
import cv2 as cv
import sys

s = 0
print(sys.argv)
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv.VideoCapture(s)

win_name = 'Camera Preview'
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

while cv.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv.imshow(win_name, frame)

source.release()
cv.destroyWindow(win_name)
"""
import cv2 as cv
source = cv.VideoCapture(0)
cv.namedWindow("Camera Test", cv.WINDOW_NORMAL)

while True:
    has_frame, frame = source.read()
    cv.imshow("Camera Test", frame)
    if cv.waitKey(1) == 27: #esc
        break

source.release()
cv.DestroyWindow("Camera Test")
