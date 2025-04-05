import torch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load the YOLOv5 model
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Move YOLOv5 model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model_yolo = model_yolo.to(device)
print(f"✅ YOLOv5 model loaded on {device}")

# Load data
df = pd.read_csv("user/thumbnails/Sports_features.csv")

# Encode 'Dominant Color'
le = LabelEncoder()
df['Dominant Color Encoded'] = le.fit_transform(df['Dominant Color'])

# Define features and target
features = [
    'Brightness', 'Contrast', 'Text Presence', 'Face Count',
    'Object Count', 'Dominant Color Encoded'
]
target = 'Views Per Day'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📈 RMSE: {rmse:.2f}")
print(f"📊 R² Score: {r2:.2f}")

# Save model and label encoder
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/thumbnail_ctr_model(Sports).pkl")
joblib.dump(le, "model/color_label_encoder(Sports).pkl")

print("✅ Model saved to model/thumbnail_ctr_model.pkl")
