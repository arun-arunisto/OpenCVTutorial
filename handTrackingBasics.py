import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

#to detect hands we are creating an object from our class
mpHands = mp.solutions.hands #the model we used to detect hands
hands = mpHands.Hands() #object that we created
mpDraw = mp.solutions.drawing_utils #to draw the landmarks

#FPS - Frame Per Second
pTime = 0 #previous time
cTime = 0 #current time


while True:
    success, img = cap.read()
    #converting the img into RGB becuase the object only uses rgb images
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks) #to get the landmarks of a hand

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #getting the information - like id and landmark
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                #getting height, width and channel of our image
                h, w, c = img.shape
                #finding the position
                #center-axis
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(cx, cy)
                #the above print will print the cx, cy value of every landmark
                #to know the value of the specified one
                print(id, cx, cy)
                #the above id=landmark, cx, cy = for getting the axis
                if id == 0: #wrist
                    #we are going to draw a circle on the wrist
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if id == 4: #thumb tip
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                if id == 8: #index finger tip
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                if id == 12: #middle finger tip
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                if id == 16: #ring finger tip
                    cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)
                if id == 20: #pinky tip
                    cv2.circle(img, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

            #mpDraw.draw_landmarks(img, handLms) #we are drawing in img property with handLms for landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #the above code only display marks this line will display the connections between the marks

    #FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    #displaying FPS on screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)