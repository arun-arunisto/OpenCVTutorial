import cv2
from handTrackingModule import handDetector
from utilsModule import cornerRect
import time
import numpy as np

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", ";"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", "Clear"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "Space"]]

final_text  = ""
#creating buttons
class Button:
    def __init__(self, pos, text, size=[80, 80]):
        self.pos = pos
        self.text = text
        self.size = size


def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8) #for transparent
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        if button.text == "Clear" or button.text == "Space":
            #cv2.rectangle(imgNew, button.pos, (x + 180, y + h), (255, 0, 0), cv2.FILLED)
            cornerRect(imgNew, (x, y, w+100, h), 20, rt=0)
            cv2.rectangle(imgNew, button.pos, (x + 180, y + h), (255, 0, 0), cv2.FILLED)
            cv2.putText(imgNew, button.text, (x + 17, y + 60), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 255, 255), 5)
        elif button.text == "W" or button.text == "A" or button.text == "S" or button.text == "D":
            cornerRect(imgNew, (x, y, w, h), 20, rt=0, colorC=(255, 0, 0))
            cv2.rectangle(imgNew, button.pos, (x + w, y + h), (0, 0, 255), cv2.FILLED)
            cv2.putText(imgNew, button.text, (x + 17, y + 60), cv2.FONT_HERSHEY_PLAIN,
                        4, (255, 255, 255), 5)
        else:
            cornerRect(imgNew, (x, y, w, h), 20, rt=0)
            cv2.rectangle(imgNew, button.pos, (x + w, y + h), (255, 0, 0), cv2.FILLED)
            cv2.putText(imgNew, button.text, (x + 17, y + 60), cv2.FONT_HERSHEY_PLAIN,
                        4, (255, 255, 255), 5)

    #for transparent
    out = img.copy()
    alpha = 0
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]
    return out

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)
detector = handDetector(detectionCon=0.8)
pTime = 0

"""my_button_q = Button([80, 80], "Q")
my_button_w = Button([180, 80], "W")
my_button_e = Button([280, 80], "E")"""

buttonList = []
for i in range(len(keys)):
    for x, key in enumerate(keys[i]):
        buttonList.append(Button([80 + (x * 100), 80 + (i * 100)], key))

while True:
    success, img = cap.read()
    handInfo, img = detector.findHands(img)
    img = drawAll(img, buttonList)
    #print(handInfo)
    """sample button created
    cv2.rectangle(img, (80, 80), (160, 160), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, "Q", (90, 145), cv2.FONT_HERSHEY_PLAIN,
                5, (255, 255, 255), 5)
    """

    """img = my_button_q.draw(img)
    img = my_button_w.draw(img)
    img = my_button_e.draw(img)"""
    #for clicking the keys
    if len(handInfo) > 0:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            #the below code will use to select keys using index finger
            if x < handInfo[0]['lmList'][8][0] < x+w and y < handInfo[0]['lmList'][8][1] < y+h:
                if button.text == "Clear" or button.text == "Space":
                    cv2.rectangle(img, button.pos, (x + 180, y + h), (255, 255, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 17, y + 60), cv2.FONT_HERSHEY_PLAIN,
                                3, (255, 0, 0), 5)
                else:
                    cv2.rectangle(img, button.pos, (x + w, y + h), (255, 255, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 17, y + 60), cv2.FONT_HERSHEY_PLAIN,
                                4, (255, 0, 0), 5)

                #for clicking we are going to use the index and thumb finger
                l, _, _ = detector.findDistance(8, 4, img)
                print(l)
                if l<30:
                    if button.text == "Clear":
                        cv2.rectangle(img, button.pos, (x + 180, y + h), (255, 0, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 17, y + 60), cv2.FONT_HERSHEY_PLAIN,
                                    3, (255, 255, 255), 5)
                        final_text = final_text[0:len(final_text)-1]
                    elif button.text == "Space":
                        cv2.rectangle(img, button.pos, (x + 180, y + h), (255, 0, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 17, y + 60), cv2.FONT_HERSHEY_PLAIN,
                                    3, (255, 255, 255), 5)
                        final_text+=" "
                    else:
                        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 17, y + 60), cv2.FONT_HERSHEY_PLAIN,
                                    4, (255, 255, 255), 5)
                        final_text+=button.text

    #text to write
    cv2.rectangle(img, (80, 400), (1160, 480), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, final_text, (90, 460), cv2.FONT_HERSHEY_PLAIN, 4,
                (255, 0, 0), 4)





    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # displaying FPS on screen
    cv2.putText(img, str(int(fps)), (0, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    # for exit if you press 'q' it will break the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break