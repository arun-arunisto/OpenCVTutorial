import cv2
from ultralytics import YOLO
import math
from sort import *
import numpy as np


#corener rectangle
def cornerRect(img, bbox, l=30, t=5, rt=1,
               colorR=(255, 0, 255), colorC=(0, 255, 0)):

    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    if rt != 0:
        cv2.rectangle(img, bbox, colorR, rt)
    # Top Left  x,y
    cv2.line(img, (x, y), (x + l, y), colorC, t)
    cv2.line(img, (x, y), (x, y + l), colorC, t)
    # Top Right  x1,y
    cv2.line(img, (x1, y), (x1 - l, y), colorC, t)
    cv2.line(img, (x1, y), (x1, y + l), colorC, t)
    # Bottom Left  x,y1
    cv2.line(img, (x, y1), (x + l, y1), colorC, t)
    cv2.line(img, (x, y1), (x, y1 - l), colorC, t)
    # Bottom Right  x1,y1
    cv2.line(img, (x1, y1), (x1 - l, y1), colorC, t)
    cv2.line(img, (x1, y1), (x1, y1 - l), colorC, t)

    return img

def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
                colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
                offset=10, border=None, colorB=(0, 255, 0)):

    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)

    return img, [x1, y2, x2, y1]


cap = cv2.VideoCapture("../data/videos/shop-cctv.mp4")
model = YOLO("../yolo-weights/yolov8n.pt")
names = model.names
mask = cv2.imread("../data/images/person-counter-mask.png")
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

out_li = [450, 240, 573, 240]
limits = [450, 250, 573, 250]
in_li = [450, 260, 573, 260]

total_id_in_frame = []
in_out_data = {}

total_in = []
total_out = []

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))  # used for resize
    maskRegion = cv2.bitwise_and(frame, mask)
    results = model(maskRegion, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)
            conf = math.ceil((box.conf[0]*100))/100
            cls = box.cls[0]
            classname = names[int(cls)]
            if classname == "person":
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    tracker_results = tracker.update(detections)
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)
    cv2.line(frame, (out_li[0], out_li[1]), (out_li[2], out_li[3]), (0, 0, 255), 2)
    cv2.line(frame, (in_li[0], in_li[1]), (in_li[2], in_li[3]), (255, 0, 0), 2)
    for results in tracker_results:

        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cx, cy = x1 + w // 2, y1 + h // 2


        cornerRect(frame, (x1, y1, w, h))
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        if Id not in total_id_in_frame:
            total_id_in_frame.append(Id)
            in_out_data[Id] = []
        if out_li[0] < cx < out_li[2] and out_li[1]-20 < cy < out_li[1]+20:
            if "out" not in in_out_data[Id]:
                in_out_data[Id].append("out")
        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[1]+20:
            if "-" not in in_out_data[Id]:
                in_out_data[Id].append("-")
        if in_li[0] < cx < in_li[2] and in_li[1]-20 < cy < in_li[1]+20:
            if "in" not in in_out_data[Id]:
                in_out_data[Id].append("in")

        if len(in_out_data[Id]) == 3 and in_out_data[Id] == ["out","-","in"]:
            if Id not in total_in:
                total_in.append(Id)
        elif len(in_out_data[Id]) == 3 and in_out_data[Id] == ["in","-","out"]:
            if Id not in total_out:
                total_out.append(Id)

    #print("Total In:",len(total_in))
    #print("Total Out:",len(total_out))
    putTextRect(frame, f"Customer   In: {len(total_in)}", (0, 38), colorR=(255, 0, 0))
    putTextRect(frame, f"Customer Out: {len(total_out)}", (0, 90), colorR=(0, 255, 0))
    cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()