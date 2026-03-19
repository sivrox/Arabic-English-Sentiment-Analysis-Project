from google_play_scraper import reviews, Sort
import pandas as pd
import re

UAE_APPS = {
    "com.careem.acma": "Careem",
    "com.talabat": "Talabat",
    "com.noon.buyers": "Noon",
    "com.du.mydu": "du telecom",
    "ae.etisalat.myetisalat": "Etisalat",
    "com.adcb.mobilebanking": "ADCB Bank",
    "com.emirates": "Emirates Airlines",
    "com.amazon.mShop.android.shopping": "Amazon AE",
}

def classify_text(text):
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', str(text)))
    has_english = bool(re.search(r'[a-zA-Z]{3,}', str(text)))
    if has_arabic and has_english:
        return "code_switched"
    elif has_arabic:
        return "pure_arabic"
    elif has_english:
        return "pure_english"
    else:
        return "other"

def star_to_sentiment(stars):
    if stars >= 4: return "positive"
    if stars <= 2: return "negative"
    return "neutral"

all_reviews = []

for app_id, app_name in UAE_APPS.items():
    print(f"Scraping {app_name}...")
    for lang_code in ["ar", "en"]:  # scrape BOTH language versions
        try:
            result, _ = reviews(
                app_id,
                lang=lang_code,
                country="ae",       # UAE store specifically
                sort=Sort.NEWEST,
                count=1000
            )
            for r in result:
                if not r["content"]:
                    continue
                all_reviews.append({
                    "text": r["content"],
                    "stars": r["score"],
                    "label": star_to_sentiment(r["score"]),
                    "text_type": classify_text(r["content"]),
                    "app": app_name,
                    "date": r["at"],
                    "source": "google_play"
                })
        except Exception as e:
            print(f"  Failed {app_name} ({lang_code}): {e}")

df = pd.DataFrame(all_reviews)
df.drop_duplicates(subset="text", inplace=True)

print("\n=== Text Type Distribution ===")
print(df["text_type"].value_counts())
print(f"\nTotal: {len(df)}")

df.to_csv("data/raw/appstore.csv", index=False, encoding="utf-8-sig")

# save code-switched specifically
cs_df = df[df["text_type"] == "code_switched"]
cs_df.to_csv("data/raw/appstore_codeswitched.csv", index=False, encoding="utf-8-sig")
print(f"\nCode-switched samples saved: {len(cs_df)}")