"""
STEP 2: App Store Review Scraper for Health AI Apps
=====================================================
Scrapes Google Play reviews for health AI apps.
Focuses on 1-3 star reviews where trust breaks down.

SETUP:
pip install google-play-scraper pandas

Run: python 2_scrape_reviews.py
"""

from google_play_scraper import reviews, Sort
import pandas as pd

# Health AI apps on Google Play
APPS = [
    ("com.webmd.android",                    "WebMD"),
    ("com.adahealth.app",                    "Ada Health"),
    ("com.khealth.android",                  "K Health"),
    ("com.buoyhealth.android",               "Buoy Health"),
    ("com.babylon.patientapp",               "Babylon Health"),
    ("ai.youper.android",                    "Youper AI"),
    ("com.symptomate.mobile",                "Symptomate"),
    ("com.healthtap.android",                "HealthTap"),
    ("com.sharecare.android",                "Sharecare"),
    ("com.infermedica.app",                  "Infermedica"),
    ("com.livongo.health",                   "Livongo Health"),
    ("com.omadahealth.omada",                "Omada Health"),
    ("com.noom.coach",                       "Noom"),
    ("com.teladoc.members",                  "Teladoc"),
    ("com.mdlive.mobile",                    "MDLive"),
    ("com.doctorondemand.android",           "Doctor On Demand"),
]

all_reviews = []

for app_id, app_name in APPS:
    print(f"Scraping {app_name}...")
    try:
        # Get 1-star reviews (most critical)
        result_1, _ = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=100,
            filter_score_with=1,
        )

        # Get 2-star reviews
        result_2, _ = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=75,
            filter_score_with=2,
        )

        # Get 3-star reviews (mixed trust signals)
        result_3, _ = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=50,
            filter_score_with=3,
        )

        for review in result_1 + result_2 + result_3:
            all_reviews.append({
                "app":            app_name,
                "app_id":         app_id,
                "app_store_url":  f"https://play.google.com/store/apps/details?id={app_id}",
                "score":          review["score"],
                "text":           review["content"],
                "date":           review["at"],
                "thumbs":         review.get("thumbsUpCount", 0),
            })

        print(f"  Got {len(result_1) + len(result_2) + len(result_3)} reviews")

    except Exception as e:
        print(f"  Error: {e}")
        continue

df = pd.DataFrame(all_reviews)
df.to_csv("../data/app_store_reviews.csv", index=False)
print(f"\nDone. Saved {len(df)} reviews to data/app_store_reviews.csv")
print(df.groupby(["app", "score"]).size())
