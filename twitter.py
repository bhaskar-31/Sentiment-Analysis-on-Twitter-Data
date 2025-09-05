# Required packages
!pip install tweepy pandas matplotlib seaborn wordcloud nltk emoji

import tweepy
import pandas as pd
import re
import emoji
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# =============================
# 1. Twitter API Authentication
# =============================
bearer_token = "YOUR_BEARER_TOKEN_HERE"  # Replace with your Bearer token

client = tweepy.Client(bearer_token=bearer_token)

# =============================
# 2. Fetch Tweets
# =============================
def fetch_tweets(query, max_results=50):
    tweets = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=["lang", "text"]
    )
    tweet_data = []
    if tweets.data:
        for tweet in tweets.data:
            if tweet.lang == "en":   # keep only English tweets
                tweet_data.append(tweet.text)
    return tweet_data

tweets = fetch_tweets("Pahalgam", max_results=50)
df = pd.DataFrame(tweets, columns=["text"])

print("Sample Original Tweets:")
print(df.head())

# =============================
# 3. Preprocessing & Emoji Handling
# =============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove urls
    text = re.sub(r"[^a-zA-Z\s]", "", text)      # remove punctuation & digits
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

df["emoji_text"] = df["text"].apply(extract_emojis)
df["clean_text"] = df["text"].apply(clean_text)
df["combined"] = df["clean_text"] + " " + df["emoji_text"]

print("\nSample Processed Tweets:")
print(df[["combined"]].head())

# =============================
# 4. Tokenization & Stopword Removal
# =============================
stop_words = set(stopwords.words("english"))

def tokenize(text):
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

df["tokens"] = df["combined"].apply(tokenize)

# Flatten token list
all_words = [word for tokens in df["tokens"] for word in tokens]

# =============================
# 5. Sentiment Analysis (VADER)
# =============================
sia = SentimentIntensityAnalyzer()

df["sentiment"] = df["combined"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["label"] = df["sentiment"].apply(lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral"))

# Sentiment distribution
sent_counts = df["label"].value_counts().reset_index()
sent_counts.columns = ["sentiment", "count"]

plt.figure(figsize=(6,4))
sns.barplot(x="sentiment", y="count", data=sent_counts, palette="Set2")
plt.title("Sentiment Distribution")
plt.show()

# =============================
# 6. Word Frequency
# =============================
word_freq = pd.Series(all_words).value_counts().head(20)

plt.figure(figsize=(8,4))
sns.barplot(x=word_freq.values, y=word_freq.index, palette="Blues_r")
plt.title("Most Frequent Words")
plt.show()

# =============================
# 7. Wordcloud
# =============================
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_words))
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud of Tweets")
plt.show()