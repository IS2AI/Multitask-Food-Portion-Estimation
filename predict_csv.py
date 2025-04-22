
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def get_image_dimensions(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return width, height

if __name__ == "__main__":
    images_dir = "/path/to/your/images"  # Replace with your image directory
    model_weight_path = "/path/to/your/model/best.pt"  # Replace with your model weight path
    output_dir = "/path/to/your/output"  # Replace with your output directory
    device_str = "cuda:1"

    model = YOLO(model_weight_path)
    my_classes = model.names
    predictions_list = []

    for img in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img)

        results = model.predict(source=img_path, imgsz=640, device=device_str)
        result = results[0].cpu().numpy()
        box = result.boxes.data

        if len(box) == 0:
            continue

        for i in range(len(box)):
            x1, y1, x2, y2, confidence, label = box[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            pred_weight = results[0].mtl[i][2].item()  
            image_name = os.path.basename(img_path)

            predictions_list.append({
                'image_name': image_name,
                'class_id': int(label),
                'xmin': x1,
                'ymin': y1,
                'xmax': x2,
                'ymax': y2,
                'weight': pred_weight,
                'conf': confidence
            })

    predictions_df = pd.DataFrame(predictions_list)
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print("Predictions saved to predictions.csv")

