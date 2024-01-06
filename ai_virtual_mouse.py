import cv2
import time
import numpy as np
from handTrackingModule import handDetector
import time
import autopy
from pynput.mouse import Controller


#scroll up
def scroll_up():
    mouse = Controller()
    mouse.scroll(0, 1)

#scroll down():
def scroll_down():
    mouse = Controller()
    mouse.scroll(0, -1)

wCam, hCam = 640, 480
frameR = 100 #frame reduction
smoothening = 7
plocX, plocY = 0, 0 #previous location
clocX, clocY = 0, 0 #current location


pTime = 0  # previous time
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = handDetector()
wScr, hScr = autopy.screen.size() #width and height of the screen
#print(wScr, hScr)
while True:
    success, img = cap.read()
    detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        #getting the tips of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1, y1, x2, y2)
        # setting range of movements
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (0, 0, 255), 2)
        #checking which fingers up
        fingers = detector.fingersUp()
        #print(fingers)
        #indexfinger - for moving mouse
        if fingers[1] == 1 and fingers[0] == 0 and fingers[2] ==0 and fingers[3]==0 and fingers[4]== 0:
            #converting coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            #smoothen the values
            clocX = plocX + (x3-plocX)/smoothening
            clocY = plocY + (y3-plocY)/smoothening
            #moving mouse
            autopy.mouse.move(wScr-clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # four fingers for scrolling the mouse
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1 and fingers[0] == 0:
            print("scrolling up")
            scroll_up()

        #scolling down three fingers up
        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            print("Scroll down")
            scroll_down()

        #index and middle finger - for clicking the mouse
        if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
            #finding distance
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                #clicking the mouse
                autopy.mouse.click()


    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # displaying FPS on screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)

    # for exit if you press 'q' it will break the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break