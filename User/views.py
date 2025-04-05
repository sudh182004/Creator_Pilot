from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import login,logout,authenticate
import os 
import json
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import cv2
import torch
import numpy as np
import joblib
import pandas as pd
import requests
from PIL import Image
from collections import Counter
from django.core.files.storage import default_storage
from django.conf import settings
from .forms import ThumbnailUploadForm
import json

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load YOLOv5
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = yolo_model.to(device)

# Preload models per category
CATEGORY_MODELS = {
    "Gaming": {
        "model": joblib.load("model/thumbnail_ctr_model.pkl"),
        "encoder": joblib.load("model/color_label_encoder.pkl"),
    },
    "Music": {
        "model": joblib.load("model/thumbnail_ctr_model(Music).pkl"),
        "encoder": joblib.load("model/color_label_encoder(Music).pkl"),
    },
    "Entertainment": {
        "model": joblib.load("model/thumbnail_ctr_model(Entertainment).pkl"),
        "encoder": joblib.load("model/color_label_encoder(Entertainment).pkl"),
    },
    "Sports": {
        "model": joblib.load("model/thumbnail_ctr_model(Sports).pkl"),
        "encoder": joblib.load("model/color_label_encoder(Sports).pkl"),
    }
}

TEXTCORTEX_API_KEY = "key_cortex"

def extract_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_count = len(faces)
    text_presence = 1 if contrast > 30 else 0
    return brightness, contrast, text_presence, face_count

def dominant_color(image_path):
    img = Image.open(image_path).resize((50, 50))
    pixels = list(img.getdata())
    colors = Counter(pixels).most_common(1)
    return str(colors[0][0])

def detect_objects(image_path):
    img = Image.open(image_path)
    results = yolo_model(img)
    return len(results.xyxy[0])

import requests

def get_ai_suggestions(category, features, predicted_ctr, brightness_percent, contrast_percent):
    prompt = f"""
    I have a YouTube thumbnail in the "{category}" category with these features:
    - Brightness: {features['Brightness']}
    - Contrast: {features['Contrast']}
    - Face Count: {features['Face Count']}
    - Text Presence: {features['Text Presence']}
    - Object Count: {features['Object Count']}
    - Dominant Color: {features['Dominant Color']}
    The predicted click-through rate (CTR) is {predicted_ctr:.2f}%.

    Based on this analysis:
    - Is the brightness {brightness_percent}% too high or too low? How can it be adjusted for better visibility?
    - Is the contrast at {contrast_percent}% too low for grabbing attention?
    - Should the text be bolder or larger for better engagement?
    - If faces are detected, should they be more prominent?
    - Given the dominant color {features['Dominant Color']}, suggest an eye-catching color contrast.
    
    Suggest **specific improvements** to increase CTR.
    """

    headers = {
        "Authorization": f"Bearer {TEXTCORTEX_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gemini-2-0-flash",
        "n": 1,
        "max_tokens": 1000,
        "text": prompt
    }

    response = requests.post("https://api.textcortex.com/v1/texts/completions", json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["data"]["outputs"][0]["text"]
    return "AI suggestion is currently unavailable."




def analyze_thumbnail(request):
    if request.method == 'POST':
        form = ThumbnailUploadForm(request.POST, request.FILES)
        if form.is_valid():
            category = form.cleaned_data['category']
            image_file = request.FILES['image']
            save_path = "user/temp_uploaded_thumbnail.jpg"
            with open(save_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            # Extract features
            brightness, contrast, text_presence, face_count = extract_features(save_path)
            dom_color = dominant_color(save_path)
            object_count = detect_objects(save_path)

            # Convert brightness & contrast to percentage
            brightness_percent = round((brightness / 255) * 100, 2)
            contrast_percent = round((contrast / 128) * 100, 2)  # Assuming 128 as high contrast

            # Load the right model and encoder
            ctr_model = CATEGORY_MODELS[category]["model"]
            label_encoder = CATEGORY_MODELS[category]["encoder"]

            try:
                dom_color_encoded = label_encoder.transform([dom_color])[0]
            except ValueError:
                dom_color_encoded = -1  

            feature_vector = [[
                brightness, contrast, text_presence, face_count,
                object_count, dom_color_encoded
            ]]
            predicted_ctr = ctr_model.predict(feature_vector)[0]

            # Normalize predicted CTR
            df = pd.read_csv(f"user/thumbnails/{category}_features.csv")
            min_views = df['Views Per Day'].min()
            max_views = df['Views Per Day'].max()
            normalized_ctr = (predicted_ctr - min_views) / (max_views - min_views)
            normalized_ctr_percent = round(normalized_ctr * 100, 2)

            # Standardized feature summary
            feature_summary = {
                'Brightness': f"{brightness:.2f} (Pixel Intensity, 0-255) - {brightness_percent}%",
                'Contrast': f"{contrast:.2f} (Standard Deviation, 0-128) - {contrast_percent}%",
                'Text Presence': "Yes" if text_presence else "No",
                'Face Count': f"{face_count} (Detected Faces)",
                'Dominant Color': f"{dom_color} (RGB)",
                'Object Count': f"{object_count} (Detected Objects)"
            }

            # Improved AI suggestion prompt
            ai_suggestion = get_ai_suggestions(category, feature_summary, predicted_ctr, brightness_percent, contrast_percent)

            os.remove(save_path)

            return render(request, 'user/analyze_thumbnail_result.html', {
                'predicted_ctr': predicted_ctr,
                'normalized_ctr_percent': normalized_ctr_percent,
                'category': category,
                'features': feature_summary,
                'ai_suggestion': ai_suggestion
            })
    else:
        form = ThumbnailUploadForm()

    return render(request, 'user/analyze_thumbnail.html', {'form': form})








def fetch_images(request):
    API_KEY = "key_freepik"
    user_query = request.GET.get("q", "").strip()
    if not user_query:
        return JsonResponse({"error": "Missing query"}, status=400)

    url = f"https://api.freepik.com/v1/resources?term=YouTube thumbnail {user_query}&filters[content_type][template]=1&filters[orientation][landscape]=1&filters[license][freemium]=1&limit=6&order=relevance"
    headers = {"x-freepik-api-key": API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response_data = response.json()

        # Extract image URLs
        images = [item["image"]["source"]["url"] for item in response_data.get("data", [])]

        print(images)
        return JsonResponse(images, safe=False)

    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": str(e)}, status=500)   



def thumnail_api(request):
    return render(request,'user/thumbnail_api.html')

def instagram_trending_songs(request):
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def register_views(request):
    if request.user.is_authenticated:

        return redirect('home')      
    else:
        if request.method=="POST":
            username = request.POST.get('username')
            email = request.POST.get('email')
            password = request.POST.get('password')
            confirm_password = request.POST.get('confirm-password')

            if password!=confirm_password:
                messages.error(request, "Passwords do not match!")
                return redirect('register')
            
            if  User.objects.filter(username=username).exists():
                messages.error(request, "Username already exists!")
                return redirect('register')

            if User.objects.filter(email=email).exists():
                messages.error(request,"Email already registered!")
                return redirect('register')
        
            user= User.objects.create_user(
                    username=username,
                    email=email,
                    password=password
            )
            user.save()
            login(request,user)
            messages.success(request, "Account created successfully! You can now log in.")
            return redirect('home')
        return render(request,'auth/register.html')


def login_views(request):
    if request.user.is_authenticated:
        return redirect('home')  
    else:
        if request.method=="POST":
            username = request.POST.get('username')
            password = request.POST.get('password')
            user  = authenticate(request,username=username,password=password)
            if user is not None:
                login(request,user)
                return redirect('home')
            else:
                return render(request,'auth/login.html',{'error_message':"Invalid username or password"})
        return render(request, 'auth/login.html')

def logout_views(request):
   if request.method=="POST":
        logout(request)
        return redirect('login')
   return render(request,'auth/logout.html')

def Home(request):

    if request.user.is_authenticated:

        return render(request,'user/home.html',
    {
    "freepik": os.getenv("FREEPIK_KEY"),
    "instagram": os.getenv("INSTA_KEY"),
    "youtube": os.getenv("YOUTUBE_KEY"),
    "cortex": os.getenv("CORTEXT_KEY"),})
    else:
        return redirect('login')
    
def song_views(request):
    return render(request,'user/songs.html',
    {
    "youtube": os.getenv("YOUTUBE_KEY"),
    "instagram": os.getenv("INSTA_KEY")
    })

def content_g_views(request):
    return render(request,'user/content-g.html')

def content_i_views(request):
    return render(request,'user/content-i.html',{"youtube": os.getenv("YOUTUBE_KEY")})

def seo_hashtag_views(request):
    return render(request,'user/seo-hashtag.html',{"youtube": os.getenv("YOUTUBE_KEY"),
                                                   "instagram": os.getenv("INSTA_KEY")
                                                   })

def posting_time_views(request):
    return render(request,'user/posting-time.html',{"youtube": os.getenv("YOUTUBE_KEY"),
                                                     "instagram": os.getenv("INSTA_KEY")})

def Dashboard(request):
    return render(request,'dashboard.html')