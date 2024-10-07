from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Export the model to TensorRT format
model.export(format="engine", imgsz=800)  # creates 'yolov8n.engine'




