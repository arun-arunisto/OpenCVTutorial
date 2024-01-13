from ultralytics import YOLO
import numpy as np

# Loading our custom model that we are trained
model = YOLO('C:\\Users\\aruna\\PycharmProjects\\AdvancedComputerVision\\COMPUTER_VISION_ENGINEER\\Image_classification\\image_classification_with_yolo\\runs\classify\\train2\\weights\\last.pt')

# Predict with the model
results = model('C:\\Users\\aruna\\PycharmProjects\\AdvancedComputerVision\\COMPUTER_VISION_ENGINEER\\Image_classification\\weather_dataset\\train\\rain\\rain1.jpg') #predicting an image

#print(results)
#print(results[0].names)
names_dict = results[0].names

#probabilities
#print(results[0].probs.data.tolist())
probs = results[0].probs.data.tolist()

print(probs)
print(names_dict)
print(names_dict[np.argmax(probs)]) #with epoch  1 the result gets as sunrise but with epoch 20 the result will become accurate