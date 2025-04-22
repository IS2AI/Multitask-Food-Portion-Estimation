import os
import glob
from PIL import Image
from ultralytics import YOLO



model = YOLO('best.pt') 

image_folder = "/path/to/your/images"  # Replace with your image folder path
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
out_folder = "/path/to/your/output"  # Replace with your output folder path
os.makedirs(out_folder, exist_ok=True)


for img in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img)
    results = model.predict(img_path, imgsz=640, device="cpu", show=False, save=False, save_txt=False)

    txt_save_path = os.path.join(out_folder, os.path.splitext(img)[0] + ".txt")
    with open(txt_save_path, "w") as f:
        for i, pred in enumerate(results[0].boxes.data):
            class_id = int(pred[5].item())
            x_center, y_center, width, height, confidence = pred[:5].tolist()

            mtl = results[0].mtl
            pred_weight = mtl[i][2].item()  

            f.write(f"{int(class_id)} {x_center} {y_center} {width} {height} {confidence} {pred_weight}\n")

print("Inference and saving completed.")
