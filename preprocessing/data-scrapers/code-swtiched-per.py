import pandas as pd
import re

def is_code_switched(text):
    if not isinstance(text, str):
        return False
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
    has_english = bool(re.search(r'[a-zA-Z]{2,}', text))
    return has_arabic and has_english

# load any of your scraped files
df = pd.read_csv("data/raw/appstore.csv")
df["is_code_switched"] = df["text"].apply(is_code_switched)

print(f"Total samples: {len(df)}")
print(f"Code-switched: {df['is_code_switched'].sum()} ({df['is_code_switched'].mean():.1%})")
print(f"Pure Arabic only: {(~df['is_code_switched']).sum()}")