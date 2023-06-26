import cv2 as cv
import numpy
import sys

PREVIEW = 0 #preview mode
BLUR = 1 #Blurring Filter
FEATURES = 2 #corner feature detector
CANNY = 3 #canny edge detector

feature_params = dict(maxCorners=500, qualityLevel=0.2,
                      minDistance=15, blockSize=9)
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True
win_name = "Camera Filter"
cv.namedWindow(win_name, cv.WINDOW_NORMAL)
result = None

source = cv.VideoCapture(s)

while alive:
    has_frame, frame = source.read()

    if not has_frame:
        break
    frame = cv.flip(frame, 1)

    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv.Canny(frame, 80, 150)
    elif image_filter == BLUR:
        result = cv.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            print(corners)
            for x, y in numpy.float32(corners).reshape(-1, 2):
                print(x, y)
                cv.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)

    cv.imshow(win_name, result)
    key = cv.waitKey(1)
    if key == ord('Q') or key == ord("q") or key == 27:
        alive = False
    elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    elif key == ord("B") or key == ord("b"):
        image_filter = BLUR
    elif key == ord("F") or key == ord("f"):
        image_filter = FEATURES
    elif key == ord("P") or key == ord("p"):
        image_filter = PREVIEW

source.release()
cv.destroyWindow(win_name)


