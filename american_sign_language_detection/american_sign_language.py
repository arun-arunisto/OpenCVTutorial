import cv2
from ultralytics import YOLO
import numpy as np
from OPENCV_MEDIAPIPE import handTrackingModule
import time


final_text = ""
cap = cv2.VideoCapture(0)
#image of asl
image = cv2.imread('/COMPUTER_VISION_ENGINEER/sign_language_detection/asl.png')
image = cv2.resize(image, (300, 300))
detector = handTrackingModule.handDetector(maxHands=1)
offset = 20
model = YOLO("/COMPUTER_VISION_ENGINEER/sign_language_detection/hand_sign_dataset/runs/classify/train/weights/last.pt")
cap.set(3, 1280)
cap.set(4, 720)

pTime = 0
while True:
    success, frame = cap.read()
    hands, img = detector.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        frameCrop = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        try:
            result = model(frameCrop)
            names_dict = result[0].names
            probs = result[0].probs.data.tolist()
            # visualizing the words
            cv2.rectangle(frame, (0, 0), (150, 150), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, names_dict[np.argmax(probs)], (35, 100), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 0),
                        15)  # checking for the correct word then pressing enter to select the word
            if cv2.waitKey(1) == ord("e"): #clicking e to select the word
                if (len(final_text) == 0):
                    final_text+=names_dict[np.argmax(probs)]
                else:
                    if final_text[-1] != names_dict[np.argmax(probs)]:
                        final_text+=names_dict[np.argmax(probs)]
        except:
            continue
    #instructions
    cv2.rectangle(frame, (1280, 0), (680, 150), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, "press 's' for space.", (690, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4)
    cv2.putText(frame, "press 'e' for select.", (690, 85), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4)
    cv2.putText(frame, "press 'c' for clear.", (690, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4)

    #overlaying image in the frame
    h, w, c = image.shape
    frame[160:h+160, 980:w+980] = image

    #text box
    cv2.rectangle(frame, (80, 600), (1200, 530), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, final_text, (90, 590), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # displaying FPS on screen
    cv2.putText(frame, str(int(fps)), (0, 180), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        final_text+=" "
    if key == ord("c"):
        final_text = final_text[:-1]
    if key == ord("q"):
        break
