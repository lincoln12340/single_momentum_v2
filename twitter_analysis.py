import requests
from datetime import datetime, timedelta, timezone
from openai import OpenAI
import json
from collections import defaultdict
import streamlit as st
from transformers import pipeline
import streamlit as st
# Setup OpenAI client
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Twitter API details
url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
headers = {"X-API-Key": st.secrets["TWITTER_API_KEY"]}

# Timeframe map
tf_map = {
    "7 Days": timedelta(days=7),
    "14 Days": timedelta(days=14),
    "30 Days": timedelta(days=30),
    "1 Month": timedelta(days=30),   # approx
    "3 Months": timedelta(days=90),  # approx
    "6 Months": timedelta(days=180), # approx
    "1 Year": timedelta(days=365)    # approx
}

sentiment_model = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

def analyze_sentiment(tweet_text, company_name):
    # """Run LLM-based sentiment analysis on a tweet."""
    # system_prompt = """
    # You are a financial tweet sentiment analysis expert.
    # Given a tweet related to a company, your task is to analyze the overall sentiment (Positive, Negative, or Neutral) as it relates to the company's outlook, performance, or investor perception.

    # Instructions:
    # - Read the tweet carefully.
    # - If the tweet expresses improvement, optimism, bullishness, or strong performance for the company, return "Positive".
    # - If the tweet expresses problems, pessimism, bearishness, negative analyst opinions, or weak performance, return "Negative".
    # - If the tweet is neutral, balanced, unclear, or promotional without impact on perception, return "Neutral".
    # - Output ONLY valid JSON. Do not include markdown, code fences, or extra commentary.

    # Output:
    # Return ONLY a valid JSON object with these keys:
    # - sentiment: [Positive or Negative or Neutral]
    # - reason: [A short, concise reason for your sentiment decision]
    # """
    # try:
    #     response = client.chat.completions.create(
    #         model="gpt-4.1",
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": f"Company: {company_name}\nContent: {tweet_text}"}
    #         ]
    #     )
    #     raw_output = response.choices[0].message.content
    #     sentiment_data = json.loads(raw_output)  # assume valid JSON
    #     return sentiment_data.get("sentiment", "Neutral"), sentiment_data.get("reason", "N/A")
    # except Exception as e:
    #     print(f"Sentiment analysis failed: {e}")
    #     return "Error", str(e)

    sentiment_model = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

def analyze_sentiment(tweet_text, company_name=None):
    """
    Run fast HuggingFace-based sentiment analysis on a tweet.
    Returns a JSON-like dict with sentiment and reason.
    """
    try:
        result = sentiment_model(tweet_text, truncation=True)[0]
        label = result["label"]
        score = result["score"]

        # Map labels to your Positive/Negative/Neutral style
        sentiment_map = {
            "LABEL_0": "Negative",   # model returns LABEL_0/1/2 for some configs
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive",
            "Negative": "Negative",
            "Neutral": "Neutral",
            "Positive": "Positive"
        }
        sentiment = sentiment_map.get(label, label)

        # Short reason based on confidence
        reason = f"Model classified as {sentiment} with confidence {score:.2f}"

        return {
            "sentiment": sentiment,
            "reason": reason
        }
    except Exception as e:
        return {
            "sentiment": "Neutral",
            "reason": f"Error during analysis: {e}"
        }



def analyze_company_tweets(company_name: str, ticker: str, timeframe: str = "3 Months"):
    """Fetch tweets for a company, analyze sentiment, and return results summary."""

    # --- Compute since_date from timeframe ---
    delta = tf_map.get(timeframe, timedelta(days=30))
    since_date = (datetime.now(timezone.utc) - delta).strftime("%Y-%m-%d_%H:%M:%S_UTC")

    # --- Fetch Tweets ---
    querystring = {
        "queryType": "Latest",
        "query": f"{company_name} since:{since_date} lang:en"
    }

    all_tweets = []
    page = 1
    MAX_TWEETS = 1000

    while True:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()

        tweets = data.get("tweets", [])
        all_tweets.extend(tweets)

        print(f"Fetched Page {page}: {len(tweets)} tweets (Total so far: {len(all_tweets)})")

        # --- Break if limit reached ---
        if len(all_tweets) >= MAX_TWEETS:
            all_tweets = all_tweets[:MAX_TWEETS]  # trim to exact limit
            break

        if data.get("has_next_page") and "next_cursor" in data:
            querystring["cursor"] = data["next_cursor"]
            page += 1
        else:
            break
    
    # --- Sentiment Analysis ---
    analyzed = 0
    sentiment_max = len(all_tweets)
    for tw in all_tweets:
        if analyzed >= sentiment_max:
            break
        text = (tw.get("text") or "").strip()
        if not text:
            continue

        result = analyze_sentiment(text, company_name)
        tw["sentiment"] = result["sentiment"]
        tw["sentiment_reason"] = result["reason"]
        analyzed += 1

        print(f"[{analyzed}/{sentiment_max}] {text}")
        print(f" → Sentiment: {result['sentiment']} | Reason: {result['reason']}\n")

        if analyzed % 50 == 0 or analyzed == sentiment_max:  
            print(f"Analyzed {analyzed}/{sentiment_max} tweets...")

    # --- Monthly Aggregation ---
    monthly_data = defaultdict(lambda: {
        "tweets": [],
        "pos": 0,
        "neg": 0,
        "neu": 0,
        "total": 0,
        "score": 0.0
    })

    for tw in all_tweets:
        created_str = tw.get("createdAt")
        try:
            created_dt = datetime.strptime(created_str, "%a %b %d %H:%M:%S %z %Y")
        except Exception:
            continue

        week_start = created_dt - timedelta(days=created_dt.weekday())
        week_label = f"Week of {week_start.strftime('%Y-%m-%d')}"
        month_label = created_dt.strftime("%Y-%m")

        sentiment = tw.get("sentiment")
        if sentiment == "Positive":
            monthly_data[month_label]["pos"] += 1
        elif sentiment == "Negative":
            monthly_data[month_label]["neg"] += 1
        elif sentiment == "Neutral":
            monthly_data[month_label]["neu"] += 1



    for monthly, stats in monthly_data.items():
        stats["total"] = stats["pos"] + stats["neg"] + stats["neu"]
        stats["score"] = (stats["pos"] - stats["neg"]) / stats["total"] if stats["total"] > 0 else 0.0

    # --- Global Totals ---
    total_pos = sum(stats["pos"] for stats in monthly_data.values())
    total_neg = sum(stats["neg"] for stats in monthly_data.values())
    total_neu = sum(stats["neu"] for stats in monthly_data.values())
    total_tweets = total_pos + total_neg + total_neu
    overall_sentiment_score = (total_pos - total_neg) / total_tweets if total_tweets > 0 else 0.0

    # --- Top Tweets ---
    top10_likes = sorted(all_tweets, key=lambda x: x.get("likeCount", 0), reverse=True)[:10]
    top10_retweets = sorted(all_tweets, key=lambda x: x.get("retweetCount", 0), reverse=True)[:10]
    top10_views = sorted(all_tweets, key=lambda x: x.get("viewCount", 0), reverse=True)[:10]

    # --- Final Results ---
    results = {
        "company": company_name,
        "ticker": ticker,
        "timeframe": timeframe,
        "monthly_summary": {month: {
            "pos": stats["pos"],
            "neg": stats["neg"],
            "neu": stats["neu"],
            "total": stats["total"],
            "score": stats["score"]
        } for month, stats in monthly_data.items()},
        "top10_likes": top10_likes,
        "top10_views": top10_views,
        "top10_retweets": top10_retweets,
        "overall_sentiment_score": overall_sentiment_score,
        "total_positive": total_pos,
        "total_negative": total_neg,
        "total_neutral": total_neu,
        "total_tweets": total_tweets
    }

    return results
