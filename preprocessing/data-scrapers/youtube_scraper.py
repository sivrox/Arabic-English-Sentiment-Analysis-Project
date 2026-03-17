from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd
import time
import os

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

# UAE-focused channels and videos that get Gulf Arabic + English comments
# These are channels whose audiences write in Gulf dialect
TARGET_CHANNELS = [
    "UCFg1W6_PFk5dEEa8lgxlqhg",  # Dubai TV
    "UCt-erDbm5JDjDVRLoRMZqRg",  # Al Arabiya
]

# Or target specific videos — food/service review videos get the best comments
TARGET_VIDEO_QUERIES = [
    # These get complaints/reviews mixing both languages
    "افضل مطعم دبي 2024",
    "تجربة طلبات دبي delivery",
    "كارييم vs أوبر دبي",
    "مول الامارات shopping experience",
    "برج خليفة tourist",
    "دبي مول review",
    "فلوق دبي vlog",       # vlogs attract code-switched comments
    "يوميات دبي daily life",
]

def get_video_ids(query, max_results=10):
    response = youtube.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=max_results,
        relevanceLanguage="ar",
        regionCode="AE"  # UAE region
    ).execute()
    return [item["id"]["videoId"] for item in response["items"]]

def get_comments(video_id, max_comments=200):
    comments = []
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        ).execute()
        
        while response and len(comments) < max_comments:
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": comment["textDisplay"],
                    "likes": comment["likeCount"],
                    "date": comment["publishedAt"],
                    "video_id": video_id,
                    "source": "youtube"
                })
            
            # get next page if exists
            if "nextPageToken" in response and len(comments) < max_comments:
                response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=response["nextPageToken"],
                    textFormat="plainText"
                ).execute()
            else:
                break
                
    except Exception as e:
        print(f"Error on video {video_id}: {e}")
    
    return comments

all_comments = []

for query in TARGET_VIDEO_QUERIES:
    print(f"Searching: {query}")
    video_ids = get_video_ids(query)
    
    for vid_id in video_ids:
        print(f"  Getting comments for video: {vid_id}")
        comments = get_comments(vid_id)
        all_comments.extend(comments)
        time.sleep(1)

df = pd.DataFrame(all_comments)
df.drop_duplicates(subset="text", inplace=True)
df.to_csv("data/raw/youtube_gulf.csv", index=False, encoding="utf-8-sig")
print(f"Collected {len(df)} unique comments")