import os
import requests
import csv
from datetime import datetime
from googleapiclient.discovery import build

# Replace with your API Key
API_KEY = ""
youtube = build("youtube", "v3", developerKey=API_KEY)

# Base path to save thumbnails (inside Django project folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THUMBNAIL_DIR = os.path.join(BASE_DIR, 'thumbnails')
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

def get_videos_by_category(category_name="Sports", max_results=50):
    request = youtube.search().list(
        part="snippet",
        maxResults=max_results,
        q=category_name,
        type="video",
        order="viewCount"
    )
    response = request.execute()
    videos = []

    for item in response['items']:
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        thumbnail_url = item['snippet']['thumbnails']['high']['url']
        published_at = item['snippet']['publishedAt']
        videos.append((video_id, title, thumbnail_url, published_at))

    return videos

def get_video_stats(video_id):
    stats_request = youtube.videos().list(
        part="statistics,snippet",
        id=video_id
    )
    stats_response = stats_request.execute()
    item = stats_response['items'][0]
    view_count = int(item['statistics'].get('viewCount', 0))
    published_date = item['snippet']['publishedAt']

    # Calculate views per day
    published_dt = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ")
    days_since_upload = max((datetime.utcnow() - published_dt).days, 1)
    views_per_day = view_count / days_since_upload

    return view_count, views_per_day

def download_thumbnail(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

def build_dataset(category_name="Sports"):
    csv_path = os.path.join(BASE_DIR, 'thumbnail_dataset.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Views", "Views Per Day", "Category"])

        videos = get_videos_by_category(category_name)
        for video_id, title, thumb_url, published_at in videos:
            try:
                views, vpd = get_video_stats(video_id)
                img_name = f"{video_id}.jpg"
                save_path = os.path.join(THUMBNAIL_DIR, img_name)

                download_thumbnail(thumb_url, save_path)
                writer.writerow([img_name, views, round(vpd, 2), category_name])
                print(f"Saved: {img_name} | {round(vpd,2)} views/day")

            except Exception as e:
                print(f"Error processing {video_id}: {e}")

    print("✅ Dataset collection complete!")

# To run it as a standalone script
if __name__ == "__main__":
    build_dataset("Sports")  # You can change the category here
