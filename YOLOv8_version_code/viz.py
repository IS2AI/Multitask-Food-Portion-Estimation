import os
import cv2
import numpy as np

def plot_ground_truth(image_folder, label_folder, class_names, output_folder):
    """
    Plot images with ground truth labels (bounding boxes and associated text).

    Args:
        image_folder (str): Path to the folder containing images.
        label_folder (str): Path to the folder containing label files.
        class_names (list): List of class names for the dataset.
        output_folder (str): Path to save the annotated images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all images in the folder
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):  # Process image files
            image_path = os.path.join(image_folder, image_file)
            label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + ".txt")
            
            # Read the image
            image = cv2.imread(image_path)
            image_height, image_width = image.shape[:2]

            # Read the corresponding label file
            if not os.path.exists(label_path):
                print(f"Label file not found for {image_file}, skipping.")
                continue

            with open(label_path, "r") as file:
                labels = file.readlines()
            
            for line in labels:
                values = line.strip().split()
                class_id = int(values[0])
                bbox = list(map(float, values[1:5]))
                weight = float(values[5])
                kcal = float(values[6])
                macronutrients = list(map(float, values[7:]))  # Protein, Fat, Carbs

                # Convert YOLO bbox format (x_center, y_center, width, height) to pixel coordinates
                x_center, y_center, w, h = bbox
                x1 = int((x_center - w / 2) * image_width)
                y1 = int((y_center - h / 2) * image_height)
                x2 = int((x_center + w / 2) * image_width)
                y2 = int((y_center + h / 2) * image_height)

                # Draw bounding box
                color = (0, 0, 255)  # Green for bounding box
                thickness = 2
                fontscale = 1.2
                text_thickness = 2
                text_color = (0, 0, 0)  # White text
                bg_color = (255, 255, 255)  # Black background

                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                # Define text to display
                texts = [
                    f"Class: {class_names[class_id]}",
                    f"Weight: {weight:.2f} g",
                ]

                # Display text above the bounding box
                text_y = y1 - 100
                for text in texts:
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontscale, text_thickness)
                    # Draw background rectangle for text
                    cv2.rectangle(image, 
                                  (x1, text_y - text_height - 5), 
                                  (x1 + text_width, text_y + 5), 
                                  bg_color, -1)
                    # Draw text on top
                    cv2.putText(image, text, 
                                (x1, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, fontscale, text_color, text_thickness)
                    text_y += 35  # Move up for next line of text

            # Save the annotated image
            output_path = os.path.join(output_folder, f"annotated_{image_file}")
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to: {output_path}")

# Example usage
image_folder = "test/images"  # Path to the images folder
label_folder = "test/labels"  # Path to the labels folder
output_folder = "out_gt"  # Path to save annotated images

# List of class names
class_names = ['achichuk', 'apple_juice', 'apple_strudel', 'balqaymaq', 'bauyrsak', 'beef', 'beef_cutlet', 'beefstroganov', 'beet_salad', 'belise_tea', 'beshbarmak', 'black_tea', 'bliny', 'borek', 'borsh', 'bread', 'broccoli', 'buckwheat', 'bukteme', 'bun', 'burger', 'butter', 'caesar_salad', 'capuccino', 'cereal', 'chak_chak', 'cheburek', 'cheese_sticks', 'cheesecake', 'chelpek', 'chicken', 'chicken_in_plum', 'chocolate', 'ciabatta', 'coke', 'compote', 'cottage_cheese', 'croissant', 'cupcake', 'dapanji', 'doner', 'dumpling_with_soup', 'eastern_sweets', 'egg', 'fanta', 'fettucini', 'fish_soup', 'french_fries', 'fresh_salad', 'fried_aubergine', 'fried_dumplings', 'fried_lagman', 'funchosa', 'golubcy', 'greek_salad', 'green_tea', 'grilled_vegetables', 'honey', 'hvorost', 'icecream', 'irimshik', 'kazy', 'kefir', 'kespe', 'ketchup', 'khachapuri', 'kirieshki_cheese', 'kirieshki_shashlyk', 'koktal', 'korean_carrot', 'kurt', 'kuyrdak', 'kymyz', 'lemonade', 'lentiil_soup', 'lime', 'lulya_kebab', 'malibu_salad', 'manpar', 'mashed_potato', 'mayonnaise', 'meat_assorty', 'meatball', 'multifruit', 'napoleon', 'naryn', 'nauryz_koje', 'nuggets', 'okroshka', 'olivie_salad', 'onion', 'orama_nan', 'pahlava', 'pie', 'pirojki', 'pizza', 'plov', 'pod_shuboi', 'pomergranate_juice', 'pop_corn', 'potato', 'quesadilla', 'ramen', 'raspberry_jam', 'rice', 'rice_porridge', 'rollton', 'salad_mushrooms', 'salmon', 'samsa', 'sandwich', 'sausage', 'sausage_in_dough', 'shashlyk_beef', 'shashlyk_chicken', 'shorpa', 'shubat', 'sirne', 'spinach', 'sprite', 'syrniki', 'taba_nan', 'tea_with_milk', 'tiramisu', 'toast', 'tom_yam', 'tongue_salad', 'tuc', 'udon', 'vinegret', 'water', 'zhal_zhaya', 'zhent']

plot_ground_truth(image_folder, label_folder, class_names, output_folder)
