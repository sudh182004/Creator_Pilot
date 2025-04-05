from django.contrib import admin
from django.urls import path,include
from .views import register_views,login_views,logout_views,Home,song_views,content_g_views,content_i_views,seo_hashtag_views,posting_time_views,instagram_trending_songs,Dashboard,analyze_thumbnail,fetch_images,thumnail_api

from django.urls import path
from .views import instagram_trending_songs

urlpatterns = [
    path('register/',register_views,name='register'),
    path('login/',login_views,name='login'),
    path('logout/',logout_views,name='logout'),
    path('home/',Home,name='home'),
    path('home/songs/',song_views,name='song'),
    path('home/content-g/',content_g_views,name='content-g'),
    path('home/content-i/',content_i_views,name='content-i'),
    path('home/seo-hashtag/',seo_hashtag_views,name='seo-hashtag'),
    path('home/posting-time/',posting_time_views,name='posting-time'),
    path('api/instagram-trending/', instagram_trending_songs, name='instagram_trending'),
    path('home/dashboard', Dashboard, name='dashboard'),
    path('home/thumbnail_api', thumnail_api, name='thumbnail_api'),
    path('fetch-images/', fetch_images, name='fetch_images'),
     path('home/analyze-thumbnail/', analyze_thumbnail, name='analyze_thumbnail'),
]
