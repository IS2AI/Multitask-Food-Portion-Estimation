import os
os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO

model = YOLO('yolov8m.pt')  

# Train the model
model.train(data='data.yaml', epochs=200, patience=30, imgsz=640, batch=16, device=[0,1])

model.val(data='data.yaml', device="cuda:0", split='test')
