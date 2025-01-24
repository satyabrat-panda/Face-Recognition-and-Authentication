import cv2
import numpy as np
import os
from PIL import Image

# Create LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to the dataset directory
path = "datasets"

def get_image_paths_and_labels(path):
    image_paths = []
    ids = []
    label_id_map = {}
    current_id = 0
    
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # Assign a unique ID to each person
        if folder_name not in label_id_map:
            label_id_map[folder_name] = current_id
            current_id += 1
        
        person_id = label_id_map[folder_name]
        
        for image_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_name)
            image_paths.append(img_path)
            ids.append(person_id)
    
    return image_paths, ids, list(label_id_map.keys())

def preprocess_images(image_paths):
    processed_images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        np_img = np.array(img, 'uint8')
        processed_images.append(np_img)
    return processed_images

# Load dataset images and IDs
image_paths, ids, names = get_image_paths_and_labels(path)
processed_images = preprocess_images(image_paths)

# Train the recognizer
recognizer.train(processed_images, np.array(ids))

# Save the trained model
recognizer.write("Trainer.yml")

# Save the names list
with open("names.txt", "w") as f:
    for name in names:
        f.write(f"{name}\n")

print("Training Completed")
