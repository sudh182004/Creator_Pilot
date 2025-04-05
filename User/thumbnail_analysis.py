import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
from collections import Counter
import torch
from torchvision import transforms

# Load face detection and YOLOv5 model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Extract brightness, contrast, text presence, face count
def extract_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_count = len(faces)
    text_presence = 1 if contrast > 30 else 0
    return brightness, contrast, text_presence, face_count

# Get dominant color
def dominant_color(image_path):
    img = Image.open(image_path).resize((50, 50))
    pixels = list(img.getdata())
    colors = Counter(pixels).most_common(1)
    return str(colors[0][0])

# Detect number of objects using YOLO
def detect_objects(image_path):
    img = Image.open(image_path)
    results = model(img)
    return len(results.xyxy[0])

# Main processing function
def process_all_thumbnails(category="Sports"):
    image_folder = f"user/thumbnails/{category}/"
    csv_file = f"user/thumbnails/{category}_data.csv"

    df = pd.read_csv(csv_file)
    print("📊 CSV Columns:", df.columns)
    all_data = []

    for idx, row in df.iterrows():
        image_path = os.path.join(image_folder, row['Image Name'])
        brightness, contrast, text_presence, face_count = extract_features(image_path)
        dom_color = dominant_color(image_path)
        object_count = detect_objects(image_path)

        all_data.append([
            row['Image Name'], row['Views'], row['Views Per Day'], row['Category'],
            brightness, contrast, text_presence, face_count, dom_color, object_count
        ])

    output_file = f"user/thumbnails/{category}_features.csv"
    final_df = pd.DataFrame(all_data, columns=[
        'Image Name', 'Views', 'Views Per Day', 'Category',
        'Brightness', 'Contrast', 'Text Presence', 'Face Count', 'Dominant Color', 'Object Count'
    ])
    final_df.to_csv(output_file, index=False)
    print(f"✅ Features extracted and saved to {output_file}")
