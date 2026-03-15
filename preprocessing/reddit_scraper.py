import requests
import pandas as pd
import time
import re

headers = {'User-Agent': 'Mozilla/5.0 (research project)'}

def scrape_subreddit(subreddit, limit=500):
    posts = []
    after = None
    while len(posts) < limit:
        url = f"https://www.reddit.com/r/{subreddit}/top.json?limit=100&t=year"
        if after:
            url += f"&after={after}"
        try:
            r = requests.get(url, headers=headers, timeout=10)
            data = r.json()['data']
            for post in data['children']:
                p = post['data']
                posts.append({
                    'text': f"{p['title']} {p['selftext']}".strip(),
                    'score': p['score'],
                    'subreddit': subreddit,
                    'source': 'reddit'
                })
            after = data.get('after')
            if not after:
                break
            time.sleep(2)
        except Exception as e:
            print(f"Error: {e}")
            break
    return posts

all_posts = []
for sub in ['dubai', 'UAE', 'saudiarabia', 'qatar', 'kuwait']:
    print(f"Scraping r/{sub}...")
    all_posts.extend(scrape_subreddit(sub))

df = pd.DataFrame(all_posts)
df.drop_duplicates(subset='text', inplace=True)

def classify(text):
    has_ar = bool(re.search(r'[\u0600-\u06FF]', str(text)))
    has_en = bool(re.search(r'[a-zA-Z]{3,}', str(text)))
    if has_ar and has_en: return 'code_switched'
    if has_ar: return 'pure_arabic'
    return 'other'

df['text_type'] = df['text'].apply(classify)
print(df['text_type'].value_counts())
df[df['text_type'] == 'code_switched'].to_csv(
    'data/raw/reddit_codeswitched.csv', index=False, encoding='utf-8-sig'
)