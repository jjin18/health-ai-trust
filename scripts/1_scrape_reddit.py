"""
STEP 1: Reddit Scraper for Health AI Trust Research
=====================================================
Scrapes posts from health/AI subreddits where people describe
bad or untrustworthy health AI experiences.

SETUP (free):
1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app" → select "script"
3. Fill in name/description, set redirect to http://localhost:8080
4. Copy your client_id (under app name) and client_secret
5. pip install praw pandas

Then fill in your credentials below and run: python 1_scrape_reddit.py
"""

import praw
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "health_ai_trust_research")

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
)

SUBREDDITS = [
    "ChatGPT",
    "AskDocs",
    "HealthIT",
    "artificial",
    "medicine",
    "nursing",
    "healthcare",
]

# Keywords that signal health AI trust breakdown
KEYWORDS = [
    "health ai", "ai doctor", "chatgpt symptom", "chatgpt diagnosis",
    "ai health", "ai wrong", "ai misdiagnosis", "ai advice",
    "copilot health", "gemini health", "didn't trust", "wrong diagnosis",
    "ai told me", "medical ai", "ai gave me", "health chatbot",
    "symptom checker", "ai said i had", "ai was wrong",
]

results = []

for sub_name in SUBREDDITS:
    print(f"Scraping r/{sub_name}...")
    subreddit = reddit.subreddit(sub_name)

    for keyword in KEYWORDS:
        try:
            for post in subreddit.search(keyword, limit=25, time_filter="year"):
                # Only keep posts with some engagement
                if post.score < 2:
                    continue

                results.append({
                    "source":    f"r/{sub_name}",
                    "keyword":   keyword,
                    "title":     post.title,
                    "body":      post.selftext[:1000],
                    "score":     post.score,
                    "url":       f"https://reddit.com{post.permalink}",
                    "created":   pd.to_datetime(post.created_utc, unit="s"),
                    "num_comments": post.num_comments,
                })

            time.sleep(1)  # be polite to the API

        except Exception as e:
            print(f"  Error on {keyword}: {e}")
            continue

df = pd.DataFrame(results).drop_duplicates(subset="url")
df.to_csv("../data/reddit_health_ai.csv", index=False)
print(f"\nDone. Saved {len(df)} posts to data/reddit_health_ai.csv")
print(df[["source", "title", "score"]].head(20))
