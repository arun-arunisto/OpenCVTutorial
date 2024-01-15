from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

results = model.train(data='C:\\Users\\aruna\\PycharmProjects\\AdvancedComputerVision\\COMPUTER_VISION_ENGINEER\\sign_language_detection\\hand_sign_dataset', epochs=30, imgsz=64)