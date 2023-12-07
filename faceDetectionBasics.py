import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

#using facedetection model
mpFaceDetection = mp.solutions.face_detection
#drawing utils
mpDraw = mp.solutions.drawing_utils
#initializing facedetection model
faceDetection = mpFaceDetection.FaceDetection()

pTime = 0
while True:
    success, img = cap.read()
    #converting img to imgRGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    #getting the id and results
    if results.detections:
        for id, detection in enumerate(results.detections):
            #print(id, detection)
            #drawing using the function provided by mediapipe
            #mpDraw.draw_detection(img, detection)
            #getting boundingbox data to draw our own detection
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            #height, width channel of our img
            h, w, c = img.shape
            bbox = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
            #drawing our own bounty box
            cv2.rectangle(img, bbox, (0, 255, 0), 1)
            #visualizing prediction score above the bounty box
            cv2.putText(img, f"{int(detection.score[0]*100)}%",
                        (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)
            #fancy draw of boundy box
            x, y, w, h = bbox
            l, t = 40, 10
            x1, y1 = x+w, y+h
            #top left x, y
            cv2.line(img, (x, y), (x+l, y), (0, 0, 255), t)
            cv2.line(img, (x, y), (x, y+l), (0, 0, 255), t)
            #top-right x1, y
            cv2.line(img, (x1, y), (x1-l, y), (0, 0, 255), t)
            cv2.line(img, (x1, y), (x1, y+l), (0, 0, 255), t)
            #bottom-left
            cv2.line(img, (x, y1), (x + l, y1), (0, 0, 255), t)
            cv2.line(img, (x, y1), (x, y1-l), (0, 0, 255), t)
            #bottom-right
            cv2.line(img, (x1, y1), (x1 - l, y1), (0, 0, 255), t)
            cv2.line(img, (x1, y1), (x1, y1 - l), (0, 0, 255), t)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #frame per second on screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break