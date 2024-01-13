from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')


# Train the model
results = model.train(data='C:\\Users\\aruna\\PycharmProjects\\AdvancedComputerVision\\COMPUTER_VISION_ENGINEER\\Image_classification\\weather_dataset', epochs=20, imgsz=64) #its just a dummy training (if you want better result change epochs to higher values and it will take time)

