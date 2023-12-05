import cv2
import mediapipe as mp
import time

#pose detection utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("videos/1.mp4")
pTime = 0
while True:
    success, img = cap.read()
    #converting img to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #model configuring
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        # drawing pose connection
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        #getting landmarks id
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            #print(id,lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            print(cx, cy)
            if id == 16: #right wrist
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            if id == 14: #right elbow
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            if id == 12: #right shoulder
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            if id == 11: #left shoulder
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            if id == 13: #left elbow
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            if id == 15: #left wrist
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)


    #fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break