from ultralytics import YOLO
import numpy as np

#loading model
model = YOLO("/COMPUTER_VISION_ENGINEER/sign_language_detection/hand_sign_dataset/runs/classify/train/weights/last.pt")

result = model("C:\\Users\\aruna\\PycharmProjects\\AdvancedComputerVision\\COMPUTER_VISION_ENGINEER\\sign_language_detection\\hand_sign_dataset\\train\\T\\Image_88.jpg")

names_dict = result[0].names

probs = result[0].probs.data.tolist()
print(probs)
print(names_dict)
print(names_dict[np.argmax(probs)])