import os
os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO
import os
import glob

# Load a model
model = YOLO('best.pt') 

image_folder = ""
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

out_folder = ""


for img in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img)
    results = model.predict(img_path, imgsz=640, device=3, show=False, save=False, save_txt=False)
    
    txt_save_path = os.path.join(out_folder, os.path.splitext(img)[0] + ".txt")
    with open(txt_save_path, "w") as f:
        for i, pred in enumerate(results[0].boxes.data):
            class_id = int(pred[5].item()) 
            x_center, y_center, width, height, confidence = pred[:5].tolist()
            
            mtl = results[0].mtl
            pred_weight = mtl[i][2].item()  # Extract custom data

            # Save in YOLO format with the additional custom weight value
            f.write(f"{int(class_id)} {x_center} {y_center} {width} {height} {confidence} {pred_weight}\n")

print("Inference and saving completed.")
