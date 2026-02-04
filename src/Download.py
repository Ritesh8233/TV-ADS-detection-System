from pytubefix import YouTube
import os

# YouTube URL
url = "https://www.youtube.com/watch?v=G-zOjqb6Ggo"

# Output folder
output_dir = r"E:\Inditronics\nurr tv"
os.makedirs(output_dir, exist_ok=True)

# Create YouTube object
yt = YouTube(url)

# Pick the **highest resolution video-only stream**
video_stream = (
    yt.streams
    .filter(only_video=True, file_extension="mp4")  # video-only
    .order_by("resolution")                         # sort by resolution
    .desc()                                         # highest first
    .first()
)

if not video_stream:
    print("❌ No video stream found")
else:
    # Download video
    video_path = video_stream.download(output_path=output_dir)
    print(f"✅ Video downloaded successfully: {video_path}")
    print(f"Resolution: {video_stream.resolution}")

