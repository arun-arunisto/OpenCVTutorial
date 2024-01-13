import cv2
import pickle
from utils import empty_or_not

video_path = "C:\\Users\\aruna\\PycharmProjects\\AdvancedComputerVision\\COMPUTER_VISION_ENGINEER\\dataset_generation\\parking_1920_1080.mp4"
cap = cv2.VideoCapture(video_path)

#width and height for bbox
width, height = 80, 25

#opening the bbox pickle file generated
pickle_path = "C:\\Users\\aruna\\PycharmProjects\\AdvancedComputerVision\\COMPUTER_VISION_ENGINEER\\dataset_generation\\car_parking_pos"
with open(pickle_path, "rb") as f:
    posList = pickle.load(f)

"""#loading scikit_model
model_path = "C:\\Users\\aruna\\PycharmProjects\\AdvancedComputerVision\\COMPUTER_VISION_ENGINEER\\Image_classification\\image_classification_scikit_learn\\model.p"
best_estimator = pickle.load(open(model_path, "rb"))"""

while True:
    #video running infinite
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, frame = cap.read()

    arr = []
    #bboxes
    for pos in posList:
        x, y = pos
        frame_crop = frame[y:y+height, x:x+width]
        frame_status = empty_or_not(frame_crop)
        if frame_status:
            cv2.rectangle(frame, pos, (pos[0]+width, pos[1]+height), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), (0, 0, 255), 2)
        arr.append(frame_status)
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Available spots: {str(arr.count(True))} / {str(len(arr))}", (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #setting the video to window size
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()