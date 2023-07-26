#face detection

import cv2 as cv
import sys

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv.VideoCapture(s)

win_name = "Camera Preview"
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

net = cv.dnn.readNetFromCaffe("facedetection/deploy.prototxt",
                              "facedetection/res10_300x300_ssd_iter_140000_fp16.caffemodel")

#model parameters
in_width = 300
in_height = 300

mean = [104, 117, 123]
conf_threshold = 0.7

while cv.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    blob = cv.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)

    #running a model
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = f"Confidence: {confidence}"
            label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_COMPLEX, 0.5, 1)

            cv.rectangle(frame, (x_left_bottom, y_left_bottom-label_size[1]),
                         (x_left_bottom+label_size[0], y_left_bottom+base_line),
                         (255, 255, 255), cv.FILLED,)
            cv.putText(frame, label, (x_left_bottom, y_left_bottom),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    t, _ = net.getPerfProfile()
    label = f"Inference time: {t*1000.0/cv.getTickFrequency()} ms"
    cv.putText(frame, label, (0, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv.imshow(win_name, frame)
source.release()
cv.destroyWindow(win_name)
