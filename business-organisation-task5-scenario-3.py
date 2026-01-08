import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# -------------------------
# Load data
# -------------------------
CSV_PATH = "steam_reviews.csv"
df = pd.read_csv(CSV_PATH)

# Keep only English reviews
df = df[df["lang"] == "en"].copy()

# Drop missing reviews
df = df.dropna(subset=["review"])

# -------------------------
# Sentiment analysis
# -------------------------
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["review"].apply(get_sentiment)

# -------------------------
# Plot 1: Sentiment distribution
# -------------------------
sentiment_counts = df["sentiment"].value_counts()

plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Distribution of Steam Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# -------------------------
# Simple semantic theme extraction (keyword-based)
# -------------------------
themes = {
    "Gameplay": ["gameplay", "mechanic", "combat", "level"],
    "Performance": ["lag", "fps", "performance", "slow", "crash"],
    "Bugs": ["bug", "glitch", "broken", "issue"],
    "Price / Value": ["price", "worth", "cost", "value"],
    "Graphics": ["graphics", "visual", "art", "design"],
}

def extract_theme(text):
    text_lower = text.lower()
    for theme, keywords in themes.items():
        for kw in keywords:
            if kw in text_lower:
                return theme
    return "Other"

df["theme"] = df["review"].apply(extract_theme)

# -------------------------
# Theme frequency by sentiment (positive vs negative)
# -------------------------

# Keep only positive and negative reviews
theme_sentiment_df = df[df["sentiment"].isin(["Positive", "Negative"])].copy()

# Count themes by sentiment
theme_sentiment_counts = (
    theme_sentiment_df
    .groupby(["theme", "sentiment"])
    .size()
    .unstack(fill_value=0)
)

# Ensure consistent column order
theme_sentiment_counts = theme_sentiment_counts[["Positive", "Negative"]]

# Plot stacked bar chart
plt.figure(figsize=(9, 5))
theme_sentiment_counts.plot(
    kind="bar",
    stacked=True,
    ax=plt.gca()
)

plt.title("Themes in Steam Reviews by Sentiment")
plt.xlabel("Theme")
plt.ylabel("Number of Reviews")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.show()


# -------------------------
# Sentiment over time (2022 to present)
# -------------------------

# Convert post_date to datetime
df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
df = df.dropna(subset=["post_date"])

# Filter from 2022 onwards
df = df[df["post_date"] >= "2022-01-01"].copy()

# Create monthly period
df["month"] = df["post_date"].dt.to_period("M").dt.to_timestamp()

# Count sentiment per month
sentiment_time = (
    df.groupby(["month", "sentiment"])
      .size()
      .unstack(fill_value=0)
      .sort_index()
)

# Plot sentiment over time
plt.figure(figsize=(10, 5))
sentiment_time.plot(ax=plt.gca())

plt.title("Sentiment of Steam Reviews Over Time (2022–Present)")
plt.xlabel("Month")
plt.ylabel("Number of Reviews")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.show()
