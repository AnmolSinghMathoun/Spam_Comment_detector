# 1. Imports, Settings & Preprocessing Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, re, string
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', None)

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# For YouTube API
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# For model building
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set up the stemmer and stopwords
stemmer = SnowballStemmer('english')
stopword = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopword]
    return ' '.join(words)

# 2. YouTube API Setup

# IMPORTANT: Replace "YOUR_API_KEY" with a valid YouTube Data API key.
API_KEY = "AIzaSyCZ8Wd1RX9ZkJOdg8_SqOkH-V6T9a7R8K4"
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_id(url):
    """
    Extract the video ID from a YouTube URL.
    Supports standard and shortened URLs.
    """
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path == '/watch':
            query = parse_qs(parsed.query)
            return query.get("v", [None])[0]
        if parsed.path.startswith(("/embed/", "/v/")):
            return parsed.path.split('/')[2]
    return None

def get_comments(video_id, max_results=100):
    """
    Extract comments from a YouTube video using the YouTube Data API.
    Gathers top-level comments from the video.
    """
    comments = []
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, textFormat="plainText", maxResults=max_results
    )
    try:
        while request:
            response = request.execute()
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            request = youtube.commentThreads().list_next(request, response)
    except HttpError as e:
        print("Error while fetching comments:", e)
        return []
    return comments

# 3. Model Training (Using Pre-Collected Data)

# For demonstration purposes, we'll assume that training data has been loaded and processed.
# Here, we use a small sample training routine with preprocessed comments.

data = {
    'CONTENT': [
        clean_text("Check out my channel and subscribe now!"),
        clean_text("I really enjoyed this video, thank you!"),
        clean_text("Don't miss my latest video, click the link!"),
        clean_text("Great work on this tutorial.")
    ],
    'CLASS': ['Spam', 'Not Spam', 'Spam', 'Not Spam']
}
df_train = pd.DataFrame(data)
X = df_train['CONTENT']
y = df_train['CLASS']

# Vectorize text using CountVectorizer (unigrams and bigrams)
vect = CountVectorizer(ngram_range=(1,2))
X_train_dtm = vect.fit_transform(X)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_dtm, y)

# 4. Real-Time YouTube Comment Extraction and Prediction 

def predict_youtube_comments(video_url):
    # Extract video ID from URL
    video_id = get_video_id(video_url)
    if not video_id:
        print("Error: Could not extract video ID from the URL.")
        return
    
    # Check if the video is live or upcoming
    try:
        video_details = youtube.videos().list(part="snippet", id=video_id).execute()
        if video_details["items"]:
            live_status = video_details["items"][0]["snippet"].get("liveBroadcastContent", "none")
            if live_status != "none":
                print("This video is live or upcoming. Please provide an existing, non-live video.")
                return
    except HttpError as e:
        print("Error fetching video details:", e)
        return
    
    print("Extracting comments from video ID:", video_id)
    comments = get_comments(video_id)
    if not comments:
        print("No comments found or an error occurred while fetching comments.")
        return
    
    # Create a DataFrame from the comments and preprocess the text
    df_comments = pd.DataFrame(comments, columns=["CONTENT"])
    df_comments["CONTENT"] = df_comments["CONTENT"].apply(clean_text)
    
    # Transform comments using the trained vectorizer
    comments_dtm = vect.transform(df_comments["CONTENT"])
    
    # Predict using the trained model
    predictions = model.predict(comments_dtm)
    df_comments["Prediction"] = predictions
    
    # Calculate percentages
    spam_count = (df_comments["Prediction"] == "Spam").sum()
    ham_count = (df_comments["Prediction"] == "Not Spam").sum()
    total = len(df_comments)
    spam_percent = (spam_count / total) * 100
    ham_percent = (ham_count / total) * 100
    
    print(f"\nTotal Comments Extracted: {total}")
    print(f"Spam Comments: {spam_count} ({spam_percent:.2f}%)")
    print(f"Ham Comments: {ham_count} ({ham_percent:.2f}%)")
    
    # Plot a pie chart
    plt.figure(figsize=(6,6))
    plt.pie([spam_count, ham_count],
            labels=["Spam", "Not Spam"],
            autopct='%1.1f%%',
            startangle=140,
            colors=['red', 'lightgreen'])
    plt.title("Spam vs Not Spam Distribution")
    plt.axis('equal')
    plt.show()
    
    return df_comments

# 5.Run on an Existing (Non-Live) YouTube Video Link


video_url = input("Enter a YouTube video URL: ")
results_df = predict_youtube_comments(video_url)

if results_df is not None:
    print(results_df.head())

    from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    video_link = data.get('link')

    # Dummy result for now
    result = {"spam": 10, "ham": 90}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
