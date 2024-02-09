### YOLO with images ###
"""
from ultralytics import YOLO
import cv2

#model = YOLO('Yolo Weights/yolov8n.pt') #model -> yolo version 8 n for nano
#you can also change the model nano is the smaller one and there is medium and large
#model = YOLO('Yolo Weights/yolov8l.pt') #l -> large its slower than other two
model = YOLO('Yolo Weights/yolov8m.pt') #m -> for medium

results = model('images/cars-on-road.jpg', show=True)
cv2.waitKey(0)
"""

### YOLO with webcam ###
from ultralytics import YOLO
import cv2
from cv_modules import utilsModule
import math

#cap = cv2.VideoCapture(0) #for webcam
cap = cv2.VideoCapture("videos/motorbikes-1.mp4")

cap.set(3, 1280)
cap.set(4, 720)

#yolo model
model = YOLO('Yolo Weights/yolov8l.pt')
names = model.names #class names for labels

while True:
    success, img = cap.read()
    results = model(img, stream=True) #generators it will be more efficient
    #for bounding box
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # converting to integer to work with the bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # drawing bounding box using cv2
            #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            #drawing bbox using own module
            w, h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)
            #print(x1, y1, w, h)
            utilsModule.cornerRect(img, bbox=bbox)

            #finding the confidence
            conf = math.ceil((box.conf[0]*100))/100 #to rounding off to two decimal places
            #print(conf)

            #print("box", box)

            #class name
            cls = box.cls[0]
            # drawing text
            utilsModule.putTextRect(img, f"{names[int(cls)]} - {conf}", (max(0, x1), max(35, y1 - 15)), scale=1, thickness=1)




    cv2.imshow("frame", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


