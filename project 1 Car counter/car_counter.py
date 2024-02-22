### YOLO with webcam ###
import numpy as np
from ultralytics import YOLO
import cv2
from cv_modules import utilsModule
import math
from sort import * #this is used to track

#cap = cv2.VideoCapture(0) #for webcam
cap = cv2.VideoCapture("../videos/cars.mp4")

cap.set(3, 1280)
cap.set(4, 720)

#yolo model
model = YOLO('../Yolo Weights/yolov8l.pt')
names = model.names #class names for labels

mask = cv2.imread("mask.png") #opening mask image for selecting particular area

#tracking cars
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

#we are going to draw a line for if the vehicle cross the line it will count for that first we are going to fix the pointers for line
limits = [410, 297, 673, 297]
total_count = []
while True:
    success, img = cap.read()
    maskRegion = cv2.bitwise_and(img, mask) #overlaying mask and frame
    results = model(maskRegion, stream=True) #generators it will be more efficient

    #for tracker we are going to save the results elements as a numpy array
    detections = np.empty((0, 5))

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


            #finding the confidence
            conf = math.ceil((box.conf[0]*100))/100 #to rounding off to two decimal places
            #print(conf)

            #print("box", box)

            #class name
            cls = box.cls[0]
            classname = names[int(cls)]
            if classname == "car" and conf >= 0.7:
                # drawing text
                #utilsModule.putTextRect(img, f"{classname}", (max(0, x1), max(35, y1 - 15)), scale=1, thickness=1, offset=5)

                #utilsModule.cornerRect(img, bbox=bbox, l=9)
                #for tracker
                currentArray = np.array([x1, y1, x2, y2, conf])
                #adding the above array to the detections (numpy array)
                detections = np.vstack((detections, currentArray))

    tracker_results = tracker.update(detections)

    #we are going to draw the line after the tracker with the points stored in limits
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    for results in tracker_results:
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #width and height for displaying tracker results
        w, h = x2-x1, y2-y1
        #displaying rectangle of tracker
        utilsModule.cornerRect(img, (x1, y1, w, h), l=9, colorR=(0, 255, 0))
        #displaying the unique id's
        #utilsModule.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=5)

        #we are going to check that the center part of the car is crossed the line then it make a count
        #first finding center
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        #find the center of the object(car) now checking that cross the line
        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if Id not in total_count:
                total_count.append(Id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 5)

    #displaying car count
    utilsModule.putTextRect(img, f"Total Cars: {len(total_count)}", (0, 38))

    cv2.imshow("frame", img)
    #cv2.imshow("maskRegion", maskRegion)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


