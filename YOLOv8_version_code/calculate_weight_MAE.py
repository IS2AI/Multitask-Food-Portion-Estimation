import os
import argparse
import cv2
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns



def get_image_dimensions(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Get image dimensions
    height, width, _ = image.shape

    return width, height

def denormalize_landmarks(landmarks, image_width, image_height):
    """
    Denormalize landmarks.

    Args:
    - landmarks (list): List of normalized landmarks [x1, y1, x2, y2, ..., xn, yn].
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Returns:
    - list: Denormalized landmarks [x1, y1, x2, y2, ..., xn, yn].
    """
    denormalized_landmarks = []

    for i in range(0, len(landmarks)):
        x_normalized, y_normalized = landmarks[i][0], landmarks[i][1]

        # Denormalize the coordinates
        x_denormalized = x_normalized * image_width
        y_denormalized = y_normalized * image_height

        denormalized_landmarks.append([x_denormalized, y_denormalized])

    return denormalized_landmarks

def match(ground_truth_bboxes,predicted_bboxes):
        # Set a distance threshold for matching
    distance_threshold = 50

    # Initialize an array to store matched pairs (index in predicted, index in ground truth)
    sort_gt = []
    sort_pred=[]
    index_pred=[]
    index_gt=[]
    
    # Iterate through predicted landmarks
    for j,gt_bbox in enumerate(ground_truth_bboxes):
        # Find the corresponding bounding box for the predicted landmark
        gt=[]
        sc=50
        # Iterate through ground truth bounding boxes
        for i, pred in enumerate(predicted_bboxes):
            pred_bbox = predicted_bboxes[i]

            # Calculate the distance between the centroids
            distance = np.linalg.norm(np.array(pred_bbox[0]) - np.array(gt_bbox[0]))
            #print(distance)

            # Check if the distance is below the threshold
            if distance < distance_threshold:
                # Add the pair to the matched pairs list
                if sc>distance: 
                    gt=gt_bbox[0:]
                    predd=pred_bbox
                    index_i=i
                    index_j=j
                    sc=distance
        if len(gt)>0:
            sort_gt.append(gt)
            sort_pred.append(predd)
            index_pred.append(index_i)
            index_gt.append(index_j)
          
    return sort_gt,sort_pred,index_gt,index_pred


def test(image, class_names, show, output_folder=None):
    pred_weight_ev = []
    gt_weight = []
    gt_bbs = []
    labels_path = image.replace("/images/", "/labels/")
    labels_path = labels_path[:labels_path.rfind(".jpg")] + ".txt"
    image_name = os.path.basename(image)

    with open(labels_path, "r") as file:
        gt_labels = file.readlines()
        image_width, image_height = get_image_dimensions(image)
        for line in gt_labels:
            labels = line.replace("\n", "").split(" ")
            gt_weight.append(float(labels[5]))       
            label = np.array(labels[1:5])
            gt_bb = label.reshape(2, 2).astype(float)
            gt_bb = denormalize_landmarks(gt_bb, image_width, image_height)
            gt_bbs.append(gt_bb)

    results = model.predict(source=image, imgsz=640, device=2)
    result = results[0].cpu().numpy()
    box = result.boxes.data
    bbs = []
    image = cv2.imread(image)
    d = len(box)
    if d > 0:
        for i in range(len(box)):
            if len(box) != 0:
                x1, y1, x2, y2, confidence, label = box[i][0], box[i][1], box[i][2], box[i][3], box[i][4], box[i][5]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                width = (box[i][2] - box[i][0])
                height = (box[i][3] - box[i][1]) 
                x_center = (box[i][0] + width / 2)
                y_center = (box[i][1] + height / 2)

                # Draw the bounding box on the image
                color = (0, 0, 255)  # Bounding box color
                thickness = 3
                fontscale = 1.2
                text_thickness = 2
                text_color = (0, 0, 0) # White text
                bg_color = (255, 255, 255)   # Black background

                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                # Predictions
                mtl = results[0].mtl
                pred_weight = mtl[i][2].item()
                pred_weight_ev.append(pred_weight)

                bbs.append([[x_center, y_center], [width, height]])

                # Define and draw text in the desired order
                additional_text_y = y1 - 90  # Start above the bounding box
                texts = [
                    f'Confidence: {confidence:.2f}',
                    f'Label: {class_names[int(label)]}',
                    f'Weight: {pred_weight:.2f}',
                ]

                for text in texts:
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontscale, text_thickness)
                    cv2.rectangle(image, 
                                (x1, additional_text_y - text_height - 5), 
                                (x1 + text_width, additional_text_y + 5), 
                                bg_color, -1)
                    cv2.putText(image, text, 
                                (x1, additional_text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, fontscale, text_color, text_thickness)
                    additional_text_y += 35  # Move to the next line                

        if show:
            # Save the annotated image
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f'annotated_{os.path.splitext(image_name)[0]}.jpg')
            cv2.imwrite(output_path, image)
            print("Prediction is saved to: ", output_path)

    return pred_weight_ev, gt_weight, bbs, gt_bbs






images_dir="path/to/your/images/"  # Directory containing images
model = YOLO("best.pt")  # Load the trained YOLOv12 model
class_names = ['achichuk', 'almond', 'apple_juice', 'apple_strudel', 'balqaymaq', 'bauyrsak', 'beef', 'beef_cutlet', 'beefstroganov', 'beet_salad', 'belise_tea', 'beshbarmak', 'black_tea', 'bliny', 'borek', 'borsh', 'bread', 'broccoli', 'buckwheat', 'bukteme', 'bun', 'burger', 'butter', 'caesar_salad', 'capuccino', 'cereal', 'chak_chak', 'cheburek', 'cheese_sticks', 'cheesecake', 'chelpek', 'chicken', 'chicken_in_plum', 'chocolate', 'ciabatta', 'coke', 'compote', 'cottage_cheese', 'croissant', 'cupcake', 'dapanji', 'doner', 'dried_apricot', 'dumpling_with_soup', 'egg', 'fanta', 'fettucini', 'fish_soup', 'french_fries', 'fresh_salad', 'fried_aubergine', 'fried_dumplings', 'fried_lagman', 'funchosa', 'golubcy', 'greek_salad', 'green_tea', 'grilled_vegetables', 'honey', 'hvorost', 'icecream', 'irimshik', 'kazy', 'kefir', 'kespe', 'ketchup', 'khachapuri', 'kirieshki_cheese', 'kirieshki_shashlyk', 'koktal', 'korean_carrot', 'kurt', 'kuyrdak', 'kymyz', 'lemonade', 'lentiil_soup', 'lime', 'lulya_kebab', 'malibu_salad', 'manpar', 'mashed_potato', 'mayonnaise', 'meat_assorty', 'meatball', 'multifruit', 'napoleon', 'naryn', 'nauryz_koje', 'nuggets', 'okroshka', 'olivie_salad', 'onion', 'orama_nan', 'pahlava', 'pie', 'pirojki', 'pistachio', 'pizza', 'plov', 'pod_shuboi', 'pomergranate_juice', 'pop_corn', 'potato', 'quesadilla', 'raisins', 'ramen', 'raspberry_jam', 'rice', 'rice_porridge', 'rollton', 'salad_mushrooms', 'salmon', 'samsa', 'sandwich', 'sausage', 'sausage_in_dough', 'shashlyk_beef', 'shashlyk_chicken', 'shorpa', 'shubat', 'sirne', 'sorpa', 'spinach', 'sprite', 'syrniki', 'taba_nan', 'tea_with_milk', 'tiramisu', 'toast', 'tom_yam', 'tongue_salad', 'tuc', 'udon', 'vinegret', 'walnut', 'water', 'zhal_zhaya', 'zhent']
output_folder="output/"  # Directory to save annotated image predictions





pred_weight=[]
pred_kcal=[]
pred_macronutrients=[]
weight_gt=[]
kcal_gt=[]
macronutrients_gt=[]

for img in os.listdir(images_dir):
    pred_weight_evs, gt_weight, pred_bbs, gt_bb = test(images_dir + img, class_names, show=False, output_folder=output_folder)
    sort_gt, sort_pred, index_gt, index_pred = match(gt_bb, pred_bbs)
    if len(pred_weight_evs) == 0: continue
    if len(sort_pred) == 0: continue
    for i, j in zip(index_pred, index_gt):
        if gt_weight[j] == -1:  # Skip invalid weights
            continue
        pred_weight_ev = pred_weight_evs[i]
        pred_weight.append(pred_weight_ev)
        weight_gt.append(gt_weight[j])

print("weight_gt: ", weight_gt)
print("pred_weight: ", pred_weight)
min_gt_weight = min(weight_gt)
max_gt_weight = max(weight_gt)
weight_gain = 1/ (max_gt_weight - min_gt_weight) if (max_gt_weight - min_gt_weight) != 0 else 0  # Avoid division by zero
print("Weight Gain:", weight_gain)
mae_weight = mean_absolute_error(weight_gt, pred_weight)
print("mae_weight: ", mae_weight)
print("Mean Absolute Error (MAE) for Weight Prediction:", mae_weight * weight_gain)




