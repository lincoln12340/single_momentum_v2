import streamlit as st
import pandas_ta as ta
from openai import OpenAI
import time
import requests
#import gspread
from alpha_vantage.timeseries import TimeSeries
#from oauth2client.service_account import ServiceAccountCredentials
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import tempfile
import json
from pydantic import BaseModel
import os 
import re

#from dotenv import load_dotenv
#from curl_cffi import requests as curl_requests
from datetime import datetime, timedelta,date
import pandas as pd
from serpapi import GoogleSearch
from dateutil.relativedelta import relativedelta
import pdfplumber
import markdown2
from bs4 import BeautifulSoup
import io
#from docx import Document
from datetime import date
import re
import calendar
from datetime import datetime, timedelta, timezone
from news_analysis import get_news_sentiment_gathered_data
from news_html import system_prompt_html
from twitter_html import twitter_system_prompt
from twitter_analysis import analyze_company_tweets


def extract_json(raw_text):
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    return match.group(1) if match else raw_text.strip()

def analyze_sentiment(tweet_text, company_name):
    system_prompt = """
    You are a financial tweet sentiment analysis expert.
    Given a tweet related to a company, your task is to analyze the overall sentiment (Positive, Negative, or Neutral) as it relates to the company's outlook, performance, or investor perception.

    Instructions:
    - Read the tweet carefully.
    - If the tweet expresses improvement, optimism, bullishness, or strong performance for the company, return "Positive".
    - If the tweet expresses problems, pessimism, bearishness, negative analyst opinions, or weak performance, return "Negative".
    - If the tweet is neutral, balanced, unclear, or promotional without impact on perception, return "Neutral".
    - Output ONLY valid JSON. Do not include markdown, code fences, or extra commentary.

    Output:
    Return ONLY a valid JSON object with these keys:
    - sentiment: [Positive or Negative or Neutral]
    - reason: [A short, concise reason for your sentiment decision]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Company: {company_name}\nContent: {tweet_text}"}
            ]
        )
        raw_output = response.choices[0].message.content
        json_output = extract_json(raw_output)
        sentiment_data = json.loads(json_output)
        return sentiment_data.get("sentiment", "Neutral"), sentiment_data.get("reason", "N/A")
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        return "Error", str(e)


def _parse_dt(dt_val):
    """Parse ISO/RFC strings or epoch (sec/ms) to aware UTC datetime."""
    if isinstance(dt_val, (int, float)):
        if dt_val > 1e12:  # ms -> sec
            dt_val = dt_val / 1000.0
        return datetime.fromtimestamp(dt_val, tz=timezone.utc)

    if isinstance(dt_val, str):
        # Add Twitter/X 'created_at' formats first
        for fmt in (
            "%a %b %d %H:%M:%S %z %Y",  # e.g. Sun Aug 10 17:58:05 +0000 2025
            "%a %b %d %H:%M:%S %Z %Y",  # sometimes %Z appears
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(dt_val, fmt)
                # Ensure UTC awareness
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        # Fallback: ISO with 'Z' or offset
        try:
            dt = datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    # Last-resort fallback: now (avoid if possible)
    return datetime.now(timezone.utc)

def _week_start(d: datetime) -> datetime:
    """Return Monday 00:00:00 UTC of the week containing d."""
    d = d.astimezone(timezone.utc)
    return (d - timedelta(days=d.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def _subtract_months(d, months):
    y, m = d.year, d.month
    m -= months
    while m <= 0:
        m += 12
        y -= 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return d.replace(year=y, month=m, day=day)

def _subtract_years(d, years):
    y = d.year - years
    m = d.month
    day = min(d.day, calendar.monthrange(y, m)[1])
    return d.replace(year=y, month=m, day=day)

def _resolve_date_range(period):
    today = date.today()
    end = today

    if isinstance(period, str):
        m = re.match(r'^\s*(\d+)\s*([dmy])\s*$', period.lower())
        if not m:
            raise ValueError("time_period must look like '7d', '3m', or '1y' (or pass a dict).")
        n = int(m.group(1))
        unit = m.group(2)
        if unit == 'd':
            start = date.fromordinal(today.toordinal() - n)
        elif unit == 'm':
            start = _subtract_months(today, n)
        else:  # 'y'
            start = _subtract_years(today, n)
    elif isinstance(period, dict):
        start = today
        if 'days' in period:
            start = date.fromordinal(start.toordinal() - int(period['days']))
        if 'months' in period:
            start = _subtract_months(start, int(period['months']))
        if 'years' in period:
            start = _subtract_years(start, int(period['years']))
    else:
        raise ValueError("time_period must be a string like '7d'/'3m'/'1y' or a dict.")

    return (start.isoformat(), end.isoformat())

def _build_search_terms_for_ticker(ticker: str):
    t = ticker.strip().upper()
    return [
        f"${t}",
        f"\"{t}\" stock",
        f"{t} price",
        f"($${t}) (up OR down OR buy OR sell OR earnings OR guidance)"
    ]

def fetch_ticker_tweets(
    token: str,
    ticker: str,
    time_period,
    *,
    actor_id: str = "mpS4GhoarZWx8LMzZ",
    max_items: int = 200,
    sort: str = "Latest",
    tweet_language: str = "en",
    twitter_handles=None,
    start_urls=None,
    sentiment_max: int = 120,   # cap for cost/latency
):
    if not token:
        raise ValueError("Missing Apify API token.")
    if not ticker:
        raise ValueError("Missing ticker.")

    start, end = _resolve_date_range(time_period)
    client_apify = ApifyClient(token)

    run_input = {
        "startUrls": start_urls or [],
        "searchTerms": [ticker],
        "twitterHandles": twitter_handles or [],
        "maxItems": max_items,
        "sort": sort,
        "tweetLanguage": tweet_language,
        "start": start,
        "end": end,
        "customMapFunction": "(object) => ({ ...object })",
    }

    # --- Run actor 3x and pick dataset with highest len(items) ---
    runs = []
    for _ in range(3):
        run = client_apify.actor(actor_id).call(run_input=run_input)
        ds_id = run["defaultDatasetId"]
        items = list(client_apify.dataset(ds_id).iterate_items())
        runs.append({"dataset_id": ds_id, "items": items})

    best_run = max(runs, key=lambda r: len(r["items"]))
    raw_items = best_run["items"]
    selected_dataset_id = best_run["dataset_id"]
    all_run_dataset_ids = [r["dataset_id"] for r in runs]

    # ---- Normalize ----
    normalized = []
    for t in raw_items:
        tw = {
            "id": t.get("id") or t.get("tweetId"),
            "url": t.get("url"),
            "text": t.get("full_text") or t.get("text"),
            "createdAt": t.get("created_at") or t.get("timestamp"),
            "authorHandle": (t.get("author") or {}).get("screen_name"),
            "authorName": (t.get("author") or {}).get("name"),
            "likeCount": t.get("favorite_count") or t.get("likeCount") or 0,
            "retweetCount": t.get("retweet_count") or 0,
            "replyCount": t.get("reply_count"),
            "quoteCount": t.get("quote_count"),
            "views": t.get("view_count") or t.get("views"),
            "lang": t.get("lang") or t.get("language"),
            "raw": t,
            # Ensure the tweet carries its own sentiment fields:
            "sentiment": "Unanalyzed",
            "sentiment_reason": None,
        }
        normalized.append(tw)

    # ---- First-pass sentiment (bounded by sentiment_max) ----
    analyzed = 0
    pos = neu = neg = err = 0
    firm = ticker

    for tw in normalized:
        if analyzed >= sentiment_max:
            break
        text = (tw.get("text") or "").strip()
        if not text:
            continue
        sentiment, reason = analyze_sentiment(text, firm)
        tw["sentiment"] = sentiment
        tw["sentiment_reason"] = reason
        analyzed += 1
        if sentiment == "Positive":
            pos += 1
        elif sentiment == "Negative":
            neg += 1
        elif sentiment == "Neutral":
            neu += 1
        else:
            err += 1

    # ---- Weekly hits ----
    start_dt = _parse_dt(start)
    end_dt = _parse_dt(end)
    start_w = _week_start(start_dt)
    end_w = _week_start(end_dt)

    weekly_counts = {}
    for tw in normalized:
        tw_dt = _parse_dt(tw["createdAt"])
        w = _week_start(tw_dt).date().isoformat()
        weekly_counts[w] = weekly_counts.get(w, 0) + 1

    weekly_hits = []
    w = start_w
    while w <= end_w:
        key = w.date().isoformat()
        weekly_hits.append({"week_start": key, "count": weekly_counts.get(key, 0)})
        w += timedelta(days=7)

    # ---- Top lists (guarantee sentiment on items shown to user) ----
    top_likes = sorted(normalized, key=lambda x: x["likeCount"], reverse=True)[:10]
    top_retweets = sorted(normalized, key=lambda x: x["retweetCount"], reverse=True)[:10]

    # def ensure_sentiment(tw_list):
    #     nonlocal analyzed, pos, neu, neg, err
    #     for tw in tw_list:
    #         if tw["sentiment"] == "Unanalyzed":
    #             text = (tw.get("text") or "").strip()
    #             if not text:
    #                 continue
    #             sentiment, reason = analyze_sentiment(text, firm)
    #             tw["sentiment"] = sentiment
    #             tw["sentiment_reason"] = reason
    #             analyzed += 1
    #             if sentiment == "Positive":
    #                 pos += 1
    #             elif sentiment == "Negative":
    #                 neg += 1
    #             elif sentiment == "Neutral":
    #                 neu += 1
    #             else:
    #                 err += 1

    # ensure_sentiment(top_likes)
    # ensure_sentiment(top_retweets)

    # ---- Sentiment summary & map ----
    sentiment_summary = {
        "Positive": pos,
        "Neutral": neu,
        "Negative": neg,
        "Errors": err,
        "Analyzed Count": analyzed
    }
    sentiment_map = {tw["id"]: tw["sentiment"] for tw in normalized if tw.get("id")}

    return {
        "total_hits": len(normalized),
        "weekly_hits": weekly_hits,
        "tweets": normalized,        # each tweet carries sentiment + reason
        "top_likes": top_likes,      # items guaranteed to have sentiment
        "top_retweets": top_retweets,# items guaranteed to have sentiment
        "sentiment_summary": sentiment_summary,
        "sentiment_map": sentiment_map
    }



#load_dotenv()

api_key = st.secrets["OPENAI_API_KEY"]
google_sheet_url = st.secrets["GOOGLE_SHEET_URL"]
private_key = st.secrets["PRIVATE_KEY"]
project_id = st.secrets["PROJECT_ID"]
private_key_id = st.secrets["PRIVATE_KEY_ID"]
client_email = st.secrets["CLIENT_EMAIL"]
client_id = st.secrets["CLIENT_ID"]
auth_uri = st.secrets["AUTH_URI"]
token_uri = st.secrets["TOKEN_URI"]
auth_provider_x509_cert_url = st.secrets["AUTH_PROVIDER_X509_CERT_URL"]
client_x509_cert_url = st.secrets["CLIENT_X509_CERT_URL"]
universe_domain = st.secrets["UNIVERSE_DOMAIN"]
type_sa = st.secrets["TYPE"]
alpha_vantage_key = st.secrets["ALPHA_VANTAGE_API_KEY"]



client = OpenAI(api_key= api_key)

@st.cache_data(ttl=3600)
def fetch_alpha_vantage_data(ticker, period):
    """Fetch data from Alpha Vantage and filter by period"""
    ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
    
    try:
        # Get full daily data (we'll filter it later)
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        data.index = pd.to_datetime(data.index)
        
        # Filter based on selected period
        today = pd.Timestamp.today()
        period_map = {
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }
        cutoff_days = period_map.get(period, 365)
        cutoff_date = today - pd.Timedelta(days=cutoff_days)

        filtered_data = data[data.index >= cutoff_date]

        
        #filtered_data = data.last(period_map.get(period, "1Y"))
        
        # Rename columns to match yfinance format
        filtered_data = filtered_data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        
        return filtered_data.sort_index()
    
    except Exception as e:
        st.error(f"Alpha Vantage Error: {str(e)}")
        return None

def fix_html_with_embedded_markdown(text):
    """
    Detects markdown sections embedded within mostly-HTML output,
    converts them to HTML, and replaces them in the text.
    """
    if not text:
        return text

    # Don't touch it if it's a fully valid HTML document
    if bool(re.search(r'<html', text, re.IGNORECASE)):
        return text

    # Pattern to detect markdown-style headings, lists, bold, etc.
    markdown_blocks = list(re.finditer(
        r'(?:(^|\n)(\s*)(#{1,6} .+|[-*+] .+|\d+\..+|>\s.+|\*\*.+\*\*|__.+__)([\s\S]+?))(?=\n{2,}|\Z)', 
        text,
        flags=re.MULTILINE
    ))

    # Convert and replace each markdown block
    for match in reversed(markdown_blocks):  # reversed to not break indices when replacing
        md_block = match.group(0).strip()
        # Only convert if not inside an HTML tag already
        if not re.match(r'<[a-z][^>]*>', md_block):
            html_block = markdown2.markdown(md_block)
            # Optionally strip <p> if markdown2 wraps the entire block
            if html_block.startswith('<p>') and html_block.endswith('</p>\n'):
                html_block = html_block[3:-5]
            # Replace markdown block with HTML
            start, end = match.span(0)
            text = text[:start] + html_block + text[end:]

    return text

def clean_embedded_markdown(html: str) -> str:
    if not html:
        return html
    # Skip if it looks like fully valid <html> doc and you trust the model
    # Otherwise, fix inline markdown markers
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html)
    html = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', html)
    html = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<em>\1</em>', html)
    return html


def stock_page():

    st.set_page_config(
        layout="wide"
    )
    

    # Initialize session state
    if "run_analysis_complete" not in st.session_state:
        st.session_state["run_analysis_complete"] = False

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("Market Analysis Dashboard")
        st.markdown("Analyze stock trends using advanced technical indicators powered by AI.")
        
        # Ticker and Company Name Input
        ticker = st.text_input(" Enter Ticker Symbol", "", help="Example: 'AAPL' for Apple Inc.").strip().upper()
        company = st.text_input(" Enter Full Company Name", "", help="Example: 'Apple Inc.'")
        
        # Timeframe Selection
        st.subheader("Select Timeframe for Analysis")
        timeframe = st.radio(
            "Choose timeframe:",
            ("3 Months", "6 Months", "1 Year"),
            index=2,
            help="Select the period of historical data for the stock analysis"
        )
        
        # Analysis Type Selection
        st.subheader("Analysis Options")
        technical_analysis = st.checkbox("Technical Analysis", help="Select to run technical analysis indicators")
        news_and_events = st.checkbox("News and Events", help="Get recent news and event analysis for the company")
        fundamental_analysis = st.checkbox("Fundamental Analysis", help="Select to upload a file for fundamental analysis")

        media_only = False
        twitter_only = False

        if news_and_events:
            selected_sources = st.multiselect(
                "Select sources",
                options=["Media/News", "Twitter"],
                default=["Media/News", "Twitter"],  # preselect both; change if you prefer
                help="Pick one or both sources for analysis"
            )

            media_only = ("Media/News" in selected_sources)
            twitter_only = ("Twitter" in selected_sources)

        selected_types = [
            technical_analysis, 
            fundamental_analysis, 
            news_and_events
        ]

        selected_types = [
            technical_analysis, 
            fundamental_analysis, 
            news_and_events
        ]
        selected_count = sum(selected_types)

        weight_choice = None
        uploaded_file = None
        if technical_analysis:
            weight_choice = st.radio(
                "Weighting Style",
                ("Short Term", "Long Term", "Default"),
                index=1,
                help="Choose analysis style for technical indicators"
            )
        if fundamental_analysis:
            uploaded_file = st.file_uploader("Upload a PDF file for Fundamental Analysis", type="pdf")

        if selected_count > 1:
            st.subheader("Analysis Weightings")
            default_weights = {
                "Technical": 0.33,
                "Fundamental": 0.33,
                "News": 0.34
            }
            weights = {}
            total = 0.0

            if technical_analysis:
                weights["Technical"] = st.slider(
                    "Technical Analysis Weight", 0.0, 1.0, default_weights["Technical"])
                total += weights["Technical"]
            else:
                weights["Technical"] = 0.0

            if fundamental_analysis:
                weights["Fundamental"] = st.slider(
                    "Fundamental Analysis Weight", 0.0, 1.0, default_weights["Fundamental"])
                total += weights["Fundamental"]
            else:
                weights["Fundamental"] = 0.0

            if news_and_events:
                weights["News"] = st.slider(
                    "News Analysis Weight", 0.0, 1.0, default_weights["News"])
                total += weights["News"]
            else:
                weights["News"] = 0.0

            # Normalize weights
            if total > 0:
                for key in weights:
                    weights[key] = weights[key] / total if weights[key] > 0 else 0.0
            else:
                selected_keys = [k for k, v in weights.items() if v > 0]
                for key in weights:
                    weights[key] = 1.0 / len(selected_keys) if key in selected_keys else 0.0

            tech_weight = weights["Technical"]
            fund_weight = weights["Fundamental"]
            news_weight = weights["News"]
        else:

            tech_weight = 1.0 if technical_analysis else 0.0
            fund_weight = 1.0 if fundamental_analysis else 0.0
            news_weight = 1.0 if news_and_events else 0.0

        # Run Button
        run_button = st.button("Run Analysis")
        st.markdown("---")
        st.info("Click 'Run Analysis' after selecting options to start.")

    # --- MAIN PANEL ---
    st.title("Stock Market Analysis with AI-Powered Insights")
    st.markdown("**Gain actionable insights into stock trends with advanced indicators and AI interpretations.**")

    progress_bar = st.progress(0)
    status_text = st.empty()

    if run_button and ticker:
        # Fetch Data
        try:
            status_text.text("Fetching data from Alpha Vantage...")
            progress_bar.progress(30)
    
            data = fetch_alpha_vantage_data(ticker, timeframe)

            if data is not None:
                progress_bar.progress(100)
                status_text.text("Data loaded successfully.")


            else:
                progress_bar.progress(0)

        except Exception as e:
            st.error(f"Error in analysis pipeline: {e}")
            progress_bar.progress(0)
            return

        # --- MUTUALLY EXCLUSIVE ANALYSIS LOGIC ---
        if not technical_analysis and not news_and_events and not fundamental_analysis:
            st.warning("Please select at least one analysis type to proceed.")
        elif data.empty:
            st.warning(f"No data available for {ticker}. Please check the ticker symbol and try again.")
        elif not company:
            st.warning("Please add Name of company.")
        # All three: TA + FA + News
        elif technical_analysis and news_and_events and fundamental_analysis:
            with st.expander("Downloading Data... Click to View Progress"):
                update_progress(progress_bar, 15, 15, "Analyzing Technicals...")
                results, recent_data, availability, score, weighted_score  = calculate_technical_indicators(data, ticker, weight_choice=weight_choice)
                bd_result = results["bd_result"]
                sma_result = results["sma_result"]
                rsi_result = results["rsi_result"]
                macd_result = results["macd_result"]
                obv_result = results["obv_result"]
                adx_result = results["adx_result"]
                summary = SUMMARY(ticker, bd_result, sma_result, rsi_result, macd_result, obv_result, adx_result, weighted_score, weight_choice)
                update_progress(progress_bar, 35, 35, "Technical Analysis complete!")
                update_progress(progress_bar, 45, 45, "Gathering News Data...")    
                txt_summary = generate_company_news_message(company, timeframe)
                update_progress(progress_bar, 75, 75, "Analysing News Data...")
                txt_summary = format_news(txt_summary)
                txt_ovr = txt_conclusion(txt_summary, company)
                update_progress(progress_bar, 80, 80, "Analysing Financial Information...")  
                file_content = uploaded_file
                file_name = uploaded_file.name
                fa_summary = FUNDAMENTAL_ANALYSIS(file_content, company, file_name)
                update_progress(progress_bar, 100, 100, "Analysis Complete...")  
            st.session_state["run_analysis_complete"] = True
            gathered_data = {
                "Ticker": ticker,
                "Company": company,
                "Timeframe": timeframe,
                "Technical Analysis": summary,
                "Fundamental Analysis": fa_summary,
                "News and Events Overall": txt_ovr,
                "News and Events Summary": txt_summary,
                "data": recent_data.to_dict(orient="records"),
                "UserSelectedWeights": {
                    "Technical Analysis Weight": tech_weight,
                    "Fundamental Analysis Weight": fund_weight,
                    "News and Events": news_weight
                },
                "Results": {
                    "SMA Results": sma_result,
                    "RSI Results": rsi_result,
                    "MACD Results": macd_result,
                    "OBV Results": obv_result,
                    "ADX Results": adx_result,
                    "BD Results": bd_result,
                }
            }
            html_text = generate_investment_analysis(gathered_data)
            html_output_no_fix = clean_html_response(html_text)
            html_output = fix_html_with_embedded_markdown(html_output_no_fix)
            st.components.v1.html(html_output, height=700, scrolling=True)
            soup = BeautifulSoup(html_output, "html.parser")
            plain_text = soup.get_text(separator='\n')
            st.session_state["gathered_data"] = gathered_data
            st.session_state["analysis_complete"] = True
            st.success("Stock analysis completed! You can now proceed to the AI Chatbot.")

            # Download buttons
            if "html_output" not in st.session_state:
                st.session_state["html_output"] = html_output
            if "plain_text" not in st.session_state:
                st.session_state["plain_text"] = plain_text
            st.download_button("Download as HTML", st.session_state["html_output"], "stock_analysis_summary.html", "text/html")
            st.download_button("Download as Plain Text", st.session_state["plain_text"], "stock_analysis_summary.txt", "text/plain")
            if st.button("Run Another Stock"):
                st.session_state.technical_analysis = False
                st.session_state.news_and_events = False
                st.session_state["run_analysis_complete"] = False
                st.experimental_rerun()

        # TA + FA only
        elif technical_analysis and fundamental_analysis:
            with st.expander("Downloading Data... Click to View Progress"):
                update_progress(progress_bar, 15, 15, "Analyzing Technicals...")
                results, recent_data, availability, score, weighted_score = calculate_technical_indicators(data, ticker, weight_choice=weight_choice)
                bd_result = results["bd_result"]
                sma_result = results["sma_result"]
                rsi_result = results["rsi_result"]
                macd_result = results["macd_result"]
                obv_result = results["obv_result"]
                adx_result = results["adx_result"]
                summary = SUMMARY(ticker, bd_result, sma_result, rsi_result, macd_result, obv_result, adx_result, weighted_score, weight_choice)
                update_progress(progress_bar, 35, 35, "Technical Analysis complete!")
                file_content = uploaded_file
                file_name = uploaded_file.name
                update_progress(progress_bar, 50, 50, "Analysing Financial Information...")  
                fa_summary = FUNDAMENTAL_ANALYSIS(file_content, company, file_name)
                update_progress(progress_bar, 80, 80, "Finalising...")  
            st.session_state["run_analysis_complete"] = True
            gathered_data = {
                "Ticker": ticker,
                "Company": company,
                "Timeframe": timeframe,
                "Technical Analysis": summary,
                "Fundamental Analysis": fa_summary,
                "data": recent_data.to_dict(orient="records"),
                "UserSelectedWeights": {
                    "Technical Analysis Weight": tech_weight,
                    "Fundamental Analysis Weight": fund_weight,
                    "News and Events": news_weight
                },
                "Results": {
                    "SMA Results": sma_result,
                    "RSI Results": rsi_result,
                    "MACD Results": macd_result,
                    "OBV Results": obv_result,
                    "ADX Results": adx_result,
                    "BD Results": bd_result,
                }
            }
            fa_ta_summary = merge_ta_fa_summary(gathered_data)
            html_output_no_fix = clean_html_response(fa_ta_summary)
            html_output = fix_html_with_embedded_markdown(html_output_no_fix)
            st.components.v1.html(html_output, height=700, scrolling=True)
            soup = BeautifulSoup(html_output, "html.parser")
            plain_text = soup.get_text(separator='\n')
            st.session_state["gathered_data"] = gathered_data
            st.session_state["analysis_complete"] = True
            st.success("Stock analysis completed! You can now proceed to the AI Chatbot.")
            if "html_output" not in st.session_state:
                st.session_state["html_output"] = html_output
            if "plain_text" not in st.session_state:
                st.session_state["plain_text"] = plain_text
            st.download_button("Download as HTML", st.session_state["html_output"], "stock_analysis_summary.html", "text/html")
            st.download_button("Download as Plain Text", st.session_state["plain_text"], "stock_analysis_summary.txt", "text/plain")
            if st.button("Run Another Stock"):
                st.session_state.technical_analysis = False
                st.session_state.news_and_events = False
                st.session_state["run_analysis_complete"] = False
                st.experimental_rerun()

        # TA + News only
        elif technical_analysis and news_and_events:
            with st.expander("Downloading Data... Click to View Progress"):
                update_progress(progress_bar, 15, 15, "Analyzing Technicals...")
                results, recent_data, availability,weighted_score = calculate_technical_indicators(data, ticker, weight_choice=weight_choice)
                bd_result = results["bd_result"]
                sma_result = results["sma_result"]
                rsi_result = results["rsi_result"]
                macd_result = results["macd_result"]
                obv_result = results["obv_result"]
                adx_result = results["adx_result"]
                summary = SUMMARY(ticker, bd_result, sma_result, rsi_result, macd_result, obv_result, adx_result, weighted_score, weight_choice)
                update_progress(progress_bar, 35, 35, "Technical Analysis complete!")
                update_progress(progress_bar, 45, 45, "Gathering News Data...")    
                txt_summary = generate_company_news_message(company, timeframe)
                update_progress(progress_bar, 75, 75, "Analysing News Data...")
                txt_summary = format_news(txt_summary)
                txt_ovr = txt_conclusion(txt_summary, company)
                update_progress(progress_bar, 100, 100, "Finalising...")
            st.session_state["run_analysis_complete"] = True
            gathered_data = {
                "Ticker": ticker,
                "Company": company,
                "Timeframe": timeframe,
                "Technical Analysis": summary,
                "News and Events Overall": txt_ovr,
                "News and Events Summary": txt_summary,
                "data": recent_data.to_dict(orient="records"),
                "UserSelectedWeights": {
                    "Technical Analysis Weight": tech_weight,
                    "Fundamental Analysis Weight": fund_weight,
                    "News and Events": news_weight
                },
                "Results": {
                    "SMA Results": sma_result,
                    "RSI Results": rsi_result,
                    "MACD Results": macd_result,
                    "OBV Results": obv_result,
                    "ADX Results": adx_result,
                    "BD Results": bd_result,
                }
            }
            ovr_summary = merge_news_and_technical_analysis_summary(gathered_data)
            html_output_no_fix = clean_html_response(ovr_summary)
            html_output = fix_html_with_embedded_markdown(html_output_no_fix)
            st.components.v1.html(html_output, height=700, scrolling=True)
            soup = BeautifulSoup(html_output, "html.parser")
            plain_text = soup.get_text(separator='\n')
            st.session_state["gathered_data"] = gathered_data
            st.session_state["analysis_complete"] = True
            st.success("Stock analysis completed! You can now proceed to the AI Chatbot.")
            if "html_output" not in st.session_state:
                st.session_state["html_output"] = html_output
            if "plain_text" not in st.session_state:
                st.session_state["plain_text"] = plain_text
            st.download_button("Download as HTML", st.session_state["html_output"], "stock_analysis_summary.html", "text/html")
            st.download_button("Download as Plain Text", st.session_state["plain_text"], "stock_analysis_summary.txt", "text/plain")
            if st.button("Run Another Stock"):
                st.session_state.technical_analysis = False
                st.session_state.news_and_events = False
                st.session_state["run_analysis_complete"] = False
                st.experimental_rerun()

        # FA + News only
        elif fundamental_analysis and news_and_events:
            with st.expander("Downloading Data..."):
                update_progress(progress_bar, 25, 25, "Gathering News Data...")    
                txt_summary = generate_company_news_message(company, timeframe)
                update_progress(progress_bar, 35, 35, "Analysing News Data...")
                txt_summary = format_news(txt_summary)
                txt_ovr = txt_conclusion(txt_summary, company)
                update_progress(progress_bar, 45, 45, "Finalising News Analysis...")
                file_content = uploaded_file
                file_name = uploaded_file.name
                update_progress(progress_bar, 60, 60, "Starting Fundamental Analysis...")
                fa_summary = FUNDAMENTAL_ANALYSIS(file_content, company, file_name)
                update_progress(progress_bar, 80, 80, "Finalising Analysis...")
            st.session_state["run_analysis_complete"] = True



            gathered_data = {
                "Ticker": ticker,
                "Company": company,
                "Timeframe": timeframe,
                "News and Events Overall": txt_ovr,
                "Fundamental Analysis": fa_summary
            }
            fa_txt_summary = fa_summary_and_news_summary(gathered_data)
            html_output_no_fix = clean_html_response(fa_txt_summary)
            html_output = fix_html_with_embedded_markdown(html_output_no_fix)
            st.components.v1.html(html_output, height=700, scrolling=True)
            soup = BeautifulSoup(html_output, "html.parser")
            plain_text = soup.get_text(separator='\n')
            st.session_state["gathered_data"] = gathered_data
            st.session_state["analysis_complete"] = True
            st.success("Stock analysis completed! You can now proceed to the AI Chatbot.")
            if "html_output" not in st.session_state:
                st.session_state["html_output"] = html_output
            if "plain_text" not in st.session_state:
                st.session_state["plain_text"] = plain_text
            st.download_button("Download as HTML", st.session_state["html_output"], "stock_analysis_summary.html", "text/html")
            st.download_button("Download as Plain Text", st.session_state["plain_text"], "stock_analysis_summary.txt", "text/plain")
            if st.button("Run Another Stock"):
                st.session_state.technical_analysis = False
                st.session_state.news_and_events = False
                st.session_state["run_analysis_complete"] = False
                st.experimental_rerun()

        # Only Technical
        elif technical_analysis:
            with st.expander("Downloading Data... Click to View Progress"):
                update_progress(progress_bar, 50, 50, "Analyzing...")
                results, recent_data, availability, weighted_score = calculate_technical_indicators(data, ticker, weight_choice=weight_choice)
                bd_result = results["bd_result"]
                sma_result = results["sma_result"]
                rsi_result = results["rsi_result"]
                macd_result = results["macd_result"]
                obv_result = results["obv_result"]
                adx_result = results["adx_result"]
            gathered_data = {
                "Ticker": ticker,
                "Company": company,
                "Timeframe": timeframe,
                "data": recent_data.to_dict(orient="records"),
                "Position_type": weight_choice,
                "weighted_score": weighted_score,
                "Results": {
                    "SMA Results": sma_result,
                    "RSI Results": rsi_result,
                    "MACD Results": macd_result,
                    "OBV Results": obv_result,
                    "BD Results": bd_result,
                    "ADX Results": adx_result
                }
            }
            summary = SUMMARY2(gathered_data)
            html_output_no_fix = clean_html_response(summary)
            html_output = fix_html_with_embedded_markdown(html_output_no_fix)
            update_progress(progress_bar, 100, 100, "Finished...")
            st.components.v1.html(html_output, height=700, width=1400, scrolling=True)
            soup = BeautifulSoup(html_output, "html.parser")
            plain_text = soup.get_text(separator='\n')
            #print(gathered_data)
            st.session_state["gathered_data"] = gathered_data
            st.session_state["analysis_complete"] = True
            st.success("Stock analysis completed! You can now proceed to the AI Chatbot.")
            if "html_output" not in st.session_state:
                st.session_state["html_output"] = html_output
            if "plain_text" not in st.session_state:
                st.session_state["plain_text"] = plain_text
            st.download_button("Download as HTML", st.session_state["html_output"], "stock_analysis_summary.html", "text/html")
            st.download_button("Download as Plain Text", st.session_state["plain_text"], "stock_analysis_summary.txt", "text/plain")
            if st.button("Run Another Stock"):
                st.session_state.technical_analysis = False
                st.session_state.news_and_events = False
                st.session_state["run_analysis_complete"] = False
                st.experimental_rerun()

        # Only Fundamental
        elif fundamental_analysis:
            with st.expander("Downloading Data"): 
                update_progress(progress_bar, 25, 25, "Analysis Started...")  
                file_content = uploaded_file
                file_name = uploaded_file.name
                update_progress(progress_bar, 50, 50, "Analysing Financial Information...")  
                fa_summary = FUNDAMENTAL_ANALYSIS2(file_content, company, file_name)
                update_progress(progress_bar, 100, 100, "Analysis Complete...")  
            gathered_data = {
                "Ticker": ticker,
                "Company": company,
                "Timeframe": timeframe,
                "Fundamental Analysis": fa_summary
            }
            html_output_no_fix = clean_html_response(fa_summary)
            html_output = fix_html_with_embedded_markdown(html_output_no_fix)
            st.components.v1.html(html_output, height=700, scrolling=True)
            soup = BeautifulSoup(html_output, "html.parser")
            plain_text = soup.get_text(separator='\n')
            st.session_state["gathered_data"] = gathered_data
            st.session_state["analysis_complete"] = True
            st.success("Stock analysis completed! You can now proceed to the AI Chatbot.")
            if "html_output" not in st.session_state:
                st.session_state["html_output"] = html_output
            if "plain_text" not in st.session_state:
                st.session_state["plain_text"] = plain_text
            st.download_button("Download as HTML", st.session_state["html_output"], "stock_analysis_summary.html", "text/html")
            st.download_button("Download as Plain Text", st.session_state["plain_text"], "stock_analysis_summary.txt", "text/plain")
            if st.button("Run Another Stock"):
                st.session_state.technical_analysis = False
                st.session_state.news_and_events = False
                st.session_state["run_analysis_complete"] = False
                st.experimental_rerun()

        # --- inside your News-only block ---
# Only News
        elif news_and_events:
            if twitter_only:
                with st.expander("Downloading Data"):
                    # Add your code here, for example:
                    st.write("Downloading data, please wait...")
                    update_progress(progress_bar, 40, 40, "Gathering Twitter/X Data...")
                    try:
                        twitter_result = analyze_company_tweets(
                            company_name=company,
                            ticker=ticker,
                            timeframe=timeframe
                        )
                    except Exception as e:
                        st.warning(f"Twitter/X fetch skipped: {e}")
                        twitter_result = {
                            "company": company,
                            "ticker": ticker,
                            "timeframe": timeframe,
                            "monthly_summary": {},
                            "top10_likes": [],
                            "top10_views": [],
                            "top10_retweets": [],
                            "overall_sentiment_score": 0.0,
                            "total_positive": 0,
                            "total_negative": 0,
                            "total_neutral": 0,
                            "total_tweets": 0,
                        }

                    # Twitter-only gathered_data
                    gathered_data = {
                        "Ticker": twitter_result.get("ticker"),
                        "Company": twitter_result.get("company"),
                        "Timeframe": twitter_result.get("timeframe"),
                        "Twitter Total Hits": twitter_result.get("total_tweets", 0),
                        "Monthly Data": twitter_result.get("monthly_summary", {}),
                        "Overall Sentiment Score": twitter_result.get("overall_sentiment_score", 0.0),
                        "Positive Tweets": twitter_result.get("total_positive", 0),
                        "Negative Tweets": twitter_result.get("total_negative", 0),
                        "Neutral Tweets": twitter_result.get("total_neutral", 0),
                        "Top Tweets by Likes": [
                            {
                                "likes": t.get("likeCount"),
                                "createdAt": t.get("createdAt"),
                                "author": t.get("author", {}).get("userName"),
                                "text": t.get("text"),
                                "url": t.get("url"),
                                "sentiment": t.get("sentiment"),
                            }
                            for t in twitter_result.get("top10_likes", [])[:10]
                        ],
                        "Top Tweets by Retweets": [
                            {
                                "retweets": t.get("retweetCount"),
                                "createdAt": t.get("createdAt"),
                                "author": t.get("author", {}).get("userName"),
                                "text": t.get("text"),
                                "url": t.get("url"),
                                "sentiment": t.get("sentiment"),
                            }
                            for t in twitter_result.get("top10_retweets", [])[:10]
                        ],
                        "Top Tweets by Views": [
                            {
                                "views": t.get("viewCount"),
                                "createdAt": t.get("createdAt"),
                                "author": t.get("author", {}).get("userName"),
                                "text": t.get("text"),
                                "url": t.get("url"),
                                "sentiment": t.get("sentiment"),
                            }
                            for t in twitter_result.get("top10_views", [])[:10]
                        ],
                    }

                txt_ovr = twitter_summary(gathered_data)
                html_output_no_fix = clean_html_response(txt_ovr)
                html_output = clean_embedded_markdown(html_output_no_fix)
                st.components.v1.html(html_output, height=700, width=1400, scrolling=True)
            
            if media_only:
                timeframe_map = {
                    "3 Months": "3m",
                    "6 Months": "6m",
                    "1 Year": "1y"
                }

                timeframe_key = timeframe_map.get(timeframe)
                with st.expander("Downloading Data"):
                    update_progress(progress_bar, 30, 30, "Gathering News Data...")
                    news_data = get_news_sentiment_gathered_data(
                        ticker=ticker,
                        period=timeframe_key,
                        company_name=company,
                        alpha_vantage_api_key=st.secrets["ALPHA_VANTAGE_API_KEY"],
                        openai_api_key=st.secrets["OPENAI_API_KEY"],
                    )
                    update_progress(progress_bar, 50, 50, "Analysing News Data...")

                news_html = generate_news_html(news_data)
                html_output_no_fix = clean_html_response(news_html)
                html_output = clean_embedded_markdown(html_output_no_fix)
                update_progress(progress_bar, 100, 100, "")
                st.components.v1.html(html_output, height=700, width=1400, scrolling=True)

            if twitter_only and media_only:
                with st.expander("Downloading Data"):
                    update_progress(progress_bar, 30, 30, "Gathering News Data...")
                    txt_summary = generate_company_news_message(company, timeframe)

                    # === NEW: Pull Twitter/X data for the same timeframe ===
                    update_progress(progress_bar, 40, 40, "Gathering Twitter/X Data...")
                    # Map your UI timeframe to the function's period format
                    tf_map = {
                        "7 Days": "7d",
                        "14 Days": "14d",
                        "30 Days": "30d",
                        "1 Month": "1m",
                        "3 Months": "3m",
                        "6 Months": "6m",
                        "1 Year": "1y",
                    }
                    period = tf_map.get(timeframe, "30d")

                    twitter_result = None
                    try:
                        twitter_result = fetch_ticker_tweets(
                            token=st.secrets["APIFY_API_TOKEN"],
                            ticker=company,
                            time_period=period,
                            max_items=300,          # adjust as you like
                            sort="Latest",
                            tweet_language="en",
                        )
                    except Exception as e:
                        st.warning(f"Twitter/X fetch skipped: {e}")
                        twitter_result = {"total_hits": 0, "tweets": [], "top_likes": [], "top_retweets": []}

                    # Optional: brief text summary snippet about tweets to blend into your narrative
                    tweets_hits = twitter_result.get("total_hits", 0)
                    top_like = twitter_result.get("top_likes", [])[:1]
                    top_rt = twitter_result.get("top_retweets", [])[:1]
                    twitter_blurb = f"\n\n**Twitter/X pulse** for ${ticker} in the last *{timeframe}*: {tweets_hits} tweets found."
                    if top_like:
                        twitter_blurb += f" Most-liked tweet had **{top_like[0].get('likeCount', 0)}** likes."
                    if top_rt:
                        twitter_blurb += f" Most-retweeted tweet had **{top_rt[0].get('retweetCount', 0)}** retweets."

                    # Fold that into your news text before formatting
                    txt_summary = txt_summary + "\n" + twitter_blurb

                    update_progress(progress_bar, 50, 50, "Analysing News Data...")
                    txt_summary = format_news(txt_summary)

                # Include Twitter/X results in gathered_data for downstream use/exports
                gathered_data = {
                    "Ticker": ticker,
                    "Company": company,
                    "Timeframe": timeframe,
                    "News and Events Summary": txt_summary,
                    "Twitter Total Hits": twitter_result.get("total_hits", 0),
                    "Top Tweets by Likes": [
                        {
                            "likes": t.get("likeCount"),
                            "createdAt": t.get("createdAt"),
                            "author": t.get("authorHandle"),
                            "text": t.get("text"),
                            "url": t.get("url"),
                        }
                        for t in twitter_result.get("top_likes", [])[:10]
                    ],
                    "Top Tweets by Retweets": [
                        {
                            "retweets": t.get("retweetCount"),
                            "createdAt": t.get("createdAt"),
                            "author": t.get("authorHandle"),
                            "text": t.get("text"),
                            "url": t.get("url"),
                        }
                        for t in twitter_result.get("top_retweets", [])[:10]
                    ],
                }

                # === Render your existing HTML summary ===
                txt_ovr = txt_conclusion2(gathered_data)
                html_output_no_fix = clean_html_response(txt_ovr)
                html_output = clean_embedded_markdown(html_output_no_fix)
                st.components.v1.html(html_output, height=700, scrolling=True)

                # keep your existing export + session state handling
                soup = BeautifulSoup(html_output, "html.parser")
                plain_text = soup.get_text(separator='\n')
                st.session_state["gathered_data"] = gathered_data
                st.session_state["analysis_complete"] = True
                st.success("Stock analysis completed! You can now proceed to the AI Chatbot.")
                if "html_output" not in st.session_state:
                    st.session_state["html_output"] = html_output
                if "plain_text" not in st.session_state:
                    st.session_state["plain_text"] = plain_text
                st.download_button("Download as HTML", st.session_state["html_output"], "stock_analysis_summary.html", "text/html")
                st.download_button("Download as Plain Text", st.session_state["plain_text"], "stock_analysis_summary.txt", "text/plain")
                if st.button("Run Another Stock"):
                    st.session_state.technical_analysis = False
                    st.session_state.news_and_events = False
                    st.session_state["run_analysis_complete"] = False
                    st.experimental_rerun()

def generate_news_html(gathered_data):
    # Use the system prompt and gathered data to generate the news HTML
    news_system_prompt = system_prompt_html

    user_message = f"The data to analyse: {json.dumps(gathered_data)}"
    
    # Call Claude API to generate the HTML with progress indicator
    with st.spinner("Generating investment analysis..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",  # Use the appropriate Claude model
                messages=[
                    {"role": "system", "content": news_system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Extract the response content
            html_content = response.choices[0].message.content
            return html_content
            
        except Exception as e:
            st.error(f"Error generating analysis: {e}")
            return None

def convert_to_raw_text(text):
    # Remove markdown headers (e.g., ###, ##, #)
    text = re.sub(r'\$', '', text)

    return text

def generate_investment_analysis(gathered_data):
    today = date.today()
    formatted = today.strftime('%Y-%m-%d')

    
    system_prompt = """
    You are an AI model designed to provide technical, fundamental, and news/events-based analysis to deliver actionable, long-term investment insights. Your role is to integrate financial health, competitive positioning, market trends, technical indicators, and relevant news/events into cohesive, data-driven recommendations for strategic, long-term investment strategies.

    The user will provide a JSON object containing all the data needed for analysis, including:
    - Ticker: The stock ticker symbol
    - Company: The company name
    - Timeframe: The analysis timeframe
    - Technical Analysis: Summary of technical analysis
    - Fundamental Analysis: Summary of fundamental analysis
    - News data: News summaries for the company and related companies/sectors
    - Results: Technical indicator results (Summary, SMA, RSI, MACD, OBV, ADX)
    - UserSelectedWeights: An object containing the user-selected weights for Technical, Fundamental, and News analyses, with each value between 0 and 1 (all weights sum to 1).

    You must parse this JSON data and use it to create a comprehensive investment report formatted STRICTLY as an HTML document.

    **Formatting Requirements:**
    - DO NOT USE MARKDOWN OR PLAIN TEXT HEADINGS UNDER ANY CIRCUMSTANCES.
    - DO NOT use Markdown syntax such as #, ##, or bullet points. 
    - DO NOT use Markdown lists. 
    - DO NOT output any content outside of the provided HTML template.
    - RESPOND ONLY IN VALID HTML using the provided template, filling all placeholders with the relevant data.
    - **NEWS SOURCES:** In the News and Events Analysis section, for every news item or event, you MUST HYPERLINK the source by making the news title or summary a clickable <a href="URL">link</a>. If multiple sources are mentioned, each must be separately hyperlinked. The same applies to news summaries in any section—always hyperlink sources.

    **Key instructions for the Financial Metrics section:**
    - For the 'Key Financial Metrics' section, you MUST ALWAYS display the following metrics as individual metric-cards within a flexbox layout:
        - Revenues
        - Sales
        - Income (Net Income or Earnings)
        - Gross Margins
        - Liabilities
        - Cash Flow (Operating Cash Flow)
        - Debt
        - Assets
    - If any value is missing, display "N/A".
    - Each metric must appear as a “metric-card” styled <div> (see CSS below) inside a parent <div class="metrics">, with <div class="metric-title"> for the name and <div class="metric-value"> for the value.
    - You may add additional metric-cards below the required ones if there are more available.

    **STRICT INSTRUCTION:**  
    Your response MUST BE A SINGLE VALID HTML DOCUMENT that fully follows the template below. DO NOT output anything in Markdown or plain text before, after, or within the template.

    Parse the provided JSON data and use it to replace the placeholders in the HTML template. Make sure to:

        1. Extract the Ticker and Company information for the title.
        2. Extract the Timeframe for the timeframe display.
        3. Extract the Technical Analysis summary for the technical analysis section.
        4. Extract Technical indicator results (SMA, RSI, MACD, OBV, ADX) for their dedicated sections.
        5. Extract the Fundamental Analysis for the fundamental analysis section.
        6. Extract News data for the news analysis section.
        - **CRITICAL:** For every news summary or event, render the news item as a clickable HTML link (using the actual source URL as <a href="URL">[headline or summary]</a>). Never show raw URLs; always use hyperlinks.
        7. Extract the user-selected weightings for each analysis type (e.g., Fundamental, Technical, News/Events). Clearly display these weights in the "User-Selected Analysis Weights" section under Investment Recommendation.
        8. When generating the overall investment recommendation, weigh the influence of each analysis type (Fundamental, Technical, News/Events) according to the user-selected weights. The final recommendation (BUY, HOLD, or SELL) should be determined by a weighted synthesis of these three components, based on their assigned importance. Clearly communicate if the result is driven more by one analysis type due to a higher weighting.

    **Recommendation Logic:**

    When generating the “Recommendation” (Buy, Hold, Sell), synthesize and weigh the findings from the Technical Analysis, Fundamental Analysis, and News/Events Analysis according to the provided weights.

    If one weighting is clearly dominant (e.g., Fundamental Analysis = 0.7), emphasize in the summary and recommendation that the final decision is mainly driven by that analysis type.

    The “Integrated Analysis” and “Alignment Assessment” sections should explicitly note which analysis types had the greatest influence on the final recommendation, based on the weights.

    **Justification:**

    The justification text for the recommendation must refer to the weights. For example:
    “Given the user’s preference to weigh Fundamental Analysis at 60%, the final recommendation relies primarily on the company’s strong balance sheet and valuation ratios, despite short-term volatility in technical indicators.”

    **IMPORTANT:**  
    - Return the complete HTML document as your response.  
    - Do not output any Markdown, plain text, or explanation before or after the HTML.  
    - Only output valid HTML using the supplied template and placeholder replacements.

    **FUNDAMENTAL ANALYSIS STRUCTURE:**  
    Within the <div id="fundamental-analysis">, ALWAYS structure the output using the following five bolded subheadings, each as a <strong> label, followed by a short, clear paragraph for each:
    
    <strong>Business Highlights:</strong> [Summary of major achievements, operations, or developments impacting the business.]
    <strong>Financial Health:</strong> [Assessment of the balance sheet, key ratios, liquidity, solvency, and overall financial stability.]
    <strong>Risk Factors:</strong> [Identification and commentary on key risks from filings or market perception.]
    <strong>Competitive & Industry Position:</strong> [Analysis of market standing, competition, and industry trends.]
    <strong>Valuation & Investment Case:</strong> [Insight on valuation multiples, growth prospects, and investment rationale.]
    
    You MUST follow this order and format for every report.

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive Investment Analysis</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0px;
                background-color: transparent;
            }
            .container {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                padding: 30px;
                margin-bottom: 30px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }
            h2 {
                color: #2c3e50;
                border-left: 5px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
                background-color: #f8f9fa;
                padding: 10px 15px;
                border-radius: 0 5px 5px 0;
            }
            h3 {
                color: #2c3e50;
                margin-top: 20px;
                border-bottom: 1px dashed #ddd;
                padding-bottom: 5px;
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            ul, ol {
                padding-left: 25px;
            }
            ul li, ol li {
                margin-bottom: 8px;
            }
            .recommendation {
                font-weight: bold;
                font-size: 1.1em;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                text-align: center;
            }
            .buy {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .hold {
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
            }
            .sell {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .metrics {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin: 20px 0;
            }
            .metric-card {
                background-color: #f0f7ff;
                border-radius: 5px;
                padding: 15px;
                flex: 1;
                min-width: 200px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .metric-title {
                font-weight: bold;
                color: #2980b9;
                margin-bottom: 5px;
            }
            .metric-value {
                font-size: 1.2em;
                font-weight: bold;
            }
            .chart-container {
                margin: 20px 0;
                text-align: center;
            }
            .footnote {
                font-size: 0.9em;
                font-style: italic;
                color: #6c757d;
                margin-top: 30px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
            }
            strong {
                color: #2980b9;
            }
            .highlight {
                background-color: #ffeaa7;
                padding: 2px 4px;
                border-radius: 3px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .summary-box {
                background-color: #e8f4fd;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }
            .indicator {
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 5px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
            }
            .indicator h4 {
                margin-top: 0;
                color: #2980b9;
            }
            .timeframe {
                font-weight: bold;
                color: #2c3e50;
                background-color: #e8f4fd;
                padding: 5px 10px;
                border-radius: 3px;
                display: inline-block;
                margin-bottom: 15px;
            }
            .weights-section {
                background-color: #f0f4f9;
                border-left: 4px solid #2980b9;
                margin-bottom: 30px;
                padding: 15px;
                border-radius: 0 5px 5px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Comprehensive Investment Analysis: [TICKER_PLACEHOLDER] - [COMPANY_PLACEHOLDER]</h1>
            
            <div class="timeframe">Analysis Timeframe: [TIMEFRAME_PLACEHOLDER]</div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p>[SUMMARY_PLACEHOLDER]</p>
                </div>
                
                <div class="recommendation [RECOMMENDATION_CLASS_PLACEHOLDER]">
                    RECOMMENDATION: [RECOMMENDATION_PLACEHOLDER]
                    <br>
                    <span style="font-size:0.95em; font-weight:normal;">
                    <strong>Note:</strong> This recommendation is primarily driven by 
                    <span class="highlight">
                        [DOMINANT_ANALYSIS_TYPE_PLACEHOLDER]
                    </span> 
                    analysis, as selected by the user’s weightings.
                    </span>
                </div>
            </div>

            <div class="section">
                <h2>Fundamental Analysis</h2>
                <div id="fundamental-analysis">
                    [FUNDAMENTAL_ANALYSIS_PLACEHOLDER]
                </div>
                
                <h3>Key Financial Metrics</h3>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">Revenues</div>
                        <div class="metric-value">[REVENUES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Sales</div>
                        <div class="metric-value">[SALES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Income</div>
                        <div class="metric-value">[INCOME_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Gross Margins</div>
                        <div class="metric-value">[GROSS_MARGINS_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Liabilities</div>
                        <div class="metric-value">[LIABILITIES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Cash Flow</div>
                        <div class="metric-value">[CASH_FLOW_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Debt</div>
                        <div class="metric-value">[DEBT_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Assets</div>
                        <div class="metric-value">[ASSETS_PLACEHOLDER]</div>
                    </div>
                    [OPTIONAL_ADDITIONAL_METRIC_CARDS_PLACEHOLDER]
                </div>
                
                <h3>Valuation Analysis</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Industry Average</th>
                        <th>Assessment</th>
                    </tr>
                    [VALUATION_METRICS_PLACEHOLDER]
                </table>
            </div>
            
            <div class="section">
                <h2>Technical Analysis</h2>
                <div id="technical-analysis">
                    [TECHNICAL_ANALYSIS_PLACEHOLDER]
                </div>
                
                <h3>Technical Indicators</h3>
                
                <div class="indicator">
                    <h4>SMA (Simple Moving Average)</h4>
                    <p>[SMA_ANALYSIS_PLACEHOLDER]</p>
                </div>
                
                <div class="indicator">
                    <h4>RSI (Relative Strength Index)</h4>
                    <p>[RSI_ANALYSIS_PLACEHOLDER]</p>
                </div>
                
                <div class="indicator">
                    <h4>MACD (Moving Average Convergence Divergence)</h4>
                    <p>[MACD_ANALYSIS_PLACEHOLDER]</p>
                </div>
                
                <div class="indicator">
                    <h4>OBV (On-Balance Volume)</h4>
                    <p>[OBV_ANALYSIS_PLACEHOLDER]</p>
                </div>
                
                <div class="indicator">
                    <h4>ADX (Average Directional Index)</h4>
                    <p>[ADX_ANALYSIS_PLACEHOLDER]</p>
                </div>

                <div class="indicator">
                    <h4>BD (Bollinger Bands)</h4>
                    <p>[BD_ANALYSIS_PLACEHOLDER]</p>
                </div>
                
                <h3>Support and Resistance Levels</h3>
                <table>
                    <tr>
                        <th>Level Type</th>
                        <th>Price Point</th>
                        <th>Strength</th>
                    </tr>
                    [SUPPORT_RESISTANCE_PLACEHOLDER]
                </table>
            </div>
            
            <div class="section">
                <h2>News and Events Analysis</h2>
                <div id="news-analysis">
                    [NEWS_ANALYSIS_PLACEHOLDER]
                </div>
                
                <h3>Recent Significant Events</h3>
                <ul>
                    [SIGNIFICANT_EVENTS_PLACEHOLDER]
                </ul>
            </div>
            
            <div class="section">
                <h2>Integrated Analysis</h2>
                <p>[INTEGRATED_ANALYSIS_PLACEHOLDER]</p>
                
                <div class="weights-section">
                    <h3>User-Selected Analysis Weights</h3>
                    <ul>
                        <li><strong>Fundamental Analysis Weight:</strong> [FUNDAMENTAL_WEIGHT_PLACEHOLDER]</li>
                        <li><strong>Technical Analysis Weight:</strong> [TECHNICAL_WEIGHT_PLACEHOLDER]</li>
                        <li><strong>News and Events Weight:</strong> [NEWS_WEIGHT_PLACEHOLDER]</li>
                    </ul>
                    <p>
                    These weights determined the overall influence of each analysis type on the final investment recommendation.
                    The report will highlight which analysis category most influenced the recommendation, based on the user’s preferences.
                    </p>
                </div>
                
                <h3>Alignment Assessment</h3>
                <table>
                    <tr>
                        <th>Analysis Type</th>
                        <th>Outlook</th>
                        <th>Confidence</th>
                    </tr>
                    <tr>
                        <td>Fundamental</td>
                        <td>[FUNDAMENTAL_OUTLOOK_PLACEHOLDER]</td>
                        <td>[FUNDAMENTAL_CONFIDENCE_PLACEHOLDER]</td>
                    </tr>
                    <tr>
                        <td>Technical</td>
                        <td>[TECHNICAL_OUTLOOK_PLACEHOLDER]</td>
                        <td>[TECHNICAL_CONFIDENCE_PLACEHOLDER]</td>
                    </tr>
                    <tr>
                        <td>News/Events</td>
                        <td>[NEWS_OUTLOOK_PLACEHOLDER]</td>
                        <td>[NEWS_CONFIDENCE_PLACEHOLDER]</td>
                    </tr>
                    <tr>
                        <td><strong>Overall</strong></td>
                        <td><strong>[OVERALL_OUTLOOK_PLACEHOLDER]</strong></td>
                        <td><strong>[OVERALL_CONFIDENCE_PLACEHOLDER]</strong></td>
                    </tr>
                </table>
                
                <h3>Investment Recommendation</h3>
                <div class="summary-box">
                    <p><strong>Recommendation:</strong> [DETAILED_RECOMMENDATION_PLACEHOLDER]</p>
                    
                    <p><strong>Entry Points:</strong> [ENTRY_POINTS_PLACEHOLDER]</p>
                    
                    <p><strong>Exit Strategy:</strong> [EXIT_STRATEGY_PLACEHOLDER]</p>
                    
                    <p><strong>Risk Management:</strong> [RISK_MANAGEMENT_PLACEHOLDER]</p>
                </div>
            </div>
            
            <div class="footnote">""" f"""
                <p>This investment analysis was generated on {formatted}, and incorporates available data as of this date. All investment decisions should be made in conjunction with personal financial advice and risk tolerance assessments.</p>
            </div>
        </div>
    </body>
    </html>
    """


    
    # Simply pass the entire gathered_data dictionary as JSON
    user_message = f"The data to analyse: {json.dumps(gathered_data)}"
    
    # Call Claude API to generate the HTML with progress indicator
    with st.spinner("Generating investment analysis..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",  # Use the appropriate Claude model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Extract the response content
            html_content = response.choices[0].message.content
            return html_content
            
        except Exception as e:
            st.error(f"Error generating analysis: {e}")
            return None

def fa_summary_and_news_summary(gathered_data):
    today = date.today()
    formatted = today.strftime('%Y-%m-%d')


    system_prompt = """
    As an AI assistant dedicated to supporting traders and investors, your task is to produce a structured, detailed market analysis in valid HTML format. Focus exclusively on synthesizing recent news/events and fundamental analysis related to the selected stock. Do not include any technical analysis or technical indicator results.

    The user will provide a JSON object containing all the data needed for analysis, including:
    - Ticker: The stock ticker symbol
    - Company: The company name
    - Timeframe: The analysis timeframe
    - FundamentalAnalysis: A comprehensive summary of the company’s fundamental position, including key financials, ratios, valuation, and management/industry factors.
    - NewsData: A summary of all significant news and events for the company and its sector.
    - SignificantEvents: A list of recent, impactful events affecting the company or its market environment.

    You must parse this JSON data and use it to create a comprehensive investment report formatted as HTML.

    **Instructions:**
    - Parse the provided JSON data and use it to replace the placeholders in the HTML template below.
    - Extract the Ticker and Company information for the title.
    - Extract the Timeframe for the timeframe display.
    - Extract the FundamentalAnalysis summary for the Fundamental Analysis section.
    - Extract the NewsData for the News and Events Analysis section.
    - Extract the SignificantEvents list for the 'Recent Significant Events' section.
    - **Always include a 'Key Financial Metrics' section within the Fundamental Analysis. Present the following metrics as responsive metric-cards within a flex container, each with its own card, even if the value is "N/A": Revenues, Sales, Income (Net Income or Earnings), Gross Margins, Liabilities, Cash Flow (Operating), Debt, Assets. Additional metrics can be added as extra cards.**
    - Generate a summary and investment recommendation (BUY, HOLD, or SELL) based on the synthesis of fundamental analysis and news/events. Justify your reasoning by referring to the financial fundamentals and the reported news/events.
    - The 'Integrated Analysis' section should synthesize all insights from both fundamental and news/event signals into a final outlook and recommendation.
    - Return the complete HTML document as your response. Do not include any Markdown or plaintext. Do not leave out any required section, even if some are brief or data is missing.

    Your output must use <section>, <h2>, <h3>, <ul>, <li>, <table>, and <p> tags as appropriate. Use <strong> for key points.

    **Follow this professional HTML template exactly, replacing the placeholders with values parsed from the provided JSON:**

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive Fundamental & News Investment Analysis</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0px;
                background-color: transparent;
            }
            .container {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                padding: 30px;
                margin-bottom: 30px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }
            h2 {
                color: #2c3e50;
                border-left: 5px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
                background-color: #f8f9fa;
                padding: 10px 15px;
                border-radius: 0 5px 5px 0;
            }
            h3 {
                color: #2c3e50;
                margin-top: 20px;
                border-bottom: 1px dashed #ddd;
                padding-bottom: 5px;
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .metrics {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin: 20px 0;
            }
            .metric-card {
                background-color: #f0f7ff;
                border-radius: 5px;
                padding: 15px;
                flex: 1;
                min-width: 200px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .metric-title {
                font-weight: bold;
                color: #2980b9;
                margin-bottom: 5px;
            }
            .metric-value {
                font-size: 1.2em;
                font-weight: bold;
            }
            .recommendation {
                font-weight: bold;
                font-size: 1.1em;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                text-align: center;
            }
            .buy {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .hold {
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
            }
            .sell {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .summary-box {
                background-color: #e8f4fd;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }
            .timeframe {
                font-weight: bold;
                color: #2c3e50;
                background-color: #e8f4fd;
                padding: 5px 10px;
                border-radius: 3px;
                display: inline-block;
                margin-bottom: 15px;
            }
            .footnote {
                font-size: 0.9em;
                font-style: italic;
                color: #6c757d;
                margin-top: 30px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
            }
            .highlight {
                background-color: #ffeaa7;
                padding: 2px 4px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Comprehensive Fundamental & News Investment Analysis: [TICKER_PLACEHOLDER] - [COMPANY_PLACEHOLDER]</h1>
            <div class="timeframe">Analysis Timeframe: [TIMEFRAME_PLACEHOLDER]</div>
            
            <section class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p>[SUMMARY_PLACEHOLDER]</p>
                </div>
                <div class="recommendation [RECOMMENDATION_CLASS_PLACEHOLDER]">
                    RECOMMENDATION: [RECOMMENDATION_PLACEHOLDER]
                    <br>
                    <span style="font-size:0.95em; font-weight:normal;">
                    <strong>Note:</strong> This recommendation is based on a synthesis of fundamental data and recent news/events.
                    </span>
                </div>
            </section>
            
            <section class="section">
                <h2>Fundamental Analysis</h2>
                <div id="fundamental-analysis">
                    [FUNDAMENTAL_ANALYSIS_PLACEHOLDER]
                </div>
                <h3>Key Financial Metrics</h3>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">Revenues</div>
                        <div class="metric-value">[REVENUES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Sales</div>
                        <div class="metric-value">[SALES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Income</div>
                        <div class="metric-value">[INCOME_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Gross Margins</div>
                        <div class="metric-value">[GROSS_MARGINS_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Liabilities</div>
                        <div class="metric-value">[LIABILITIES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Cash Flow</div>
                        <div class="metric-value">[CASH_FLOW_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Debt</div>
                        <div class="metric-value">[DEBT_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Assets</div>
                        <div class="metric-value">[ASSETS_PLACEHOLDER]</div>
                    </div>
                    [OPTIONAL_ADDITIONAL_METRIC_CARDS_PLACEHOLDER]
                </div>
            </section>
            
            <section class="section">
                <h2>News and Events Analysis</h2>
                <div id="news-analysis">
                    [NEWS_ANALYSIS_PLACEHOLDER]
                </div>
                <h3>Recent Significant Events</h3>
                <ul>
                    [SIGNIFICANT_EVENTS_PLACEHOLDER]
                </ul>
            </section>
            
            <section class="section">
                <h2>Integrated Analysis</h2>
                <p>[INTEGRATED_ANALYSIS_PLACEHOLDER]</p>
                <h3>Investment Recommendation</h3>
                <div class="summary-box">
                    <p><strong>Recommendation:</strong> [DETAILED_RECOMMENDATION_PLACEHOLDER]</p>
                    <p><strong>Entry Points:</strong> [ENTRY_POINTS_PLACEHOLDER]</p>
                    <p><strong>Exit Strategy:</strong> [EXIT_STRATEGY_PLACEHOLDER]</p>
                    <p><strong>Risk Management:</strong> [RISK_MANAGEMENT_PLACEHOLDER]</p>
                </div>
            </section>
            
            <div class="footnote"> """ f"""
                <p>This investment analysis was generated on {formatted}, and incorporates available financial fundamentals, news, and event data as of this date. All investment decisions should be made in conjunction with personal financial advice and risk tolerance assessments.</p>
            </div>
        </div>
    </body>
    </html>

    Recommendation Logic:

    When generating the “Recommendation” (Buy, Hold, Sell), synthesize and weigh the findings from the Fundamental Analysis, and News/Events Analysis according to the provided weights.

    - **Key Financial Metrics Requirement:**  
    In the 'Fundamental Analysis' section, always display the following metrics as individual metric-cards in a flex container: Revenues, Sales, Income, Gross Margins, Liabilities, Cash Flow, Debt, Assets (with "N/A" if missing). Additional metrics may be added as more cards.

    If one weighting is clearly dominant (e.g., Fundamental Analysis = 0.7), emphasize in the summary and recommendation that the final decision is mainly driven by that analysis type.

    The “Integrated Analysis” and “Alignment Assessment” sections should explicitly note which analysis types had the greatest influence on the final recommendation, based on the weights.

    Justification:

    The justification text for the recommendation must refer to the weights. For example:
    “Given the user’s preference to weigh Fundamental Analysis at 60%, the final recommendation relies primarily on the company’s strong balance sheet and valuation ratios, despite any recent negative news events.”

    Return the complete HTML document as your response.
    """

    user_message = f"The data to analyse: {json.dumps(gathered_data)}"

           
    chat_completion = client.chat.completions.create(
        model="gpt-4.1",  # Ensure that you use a model available in your OpenAI subscription
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            },
        ]
    )

    response = chat_completion.choices[0].message.content
    return response




                
def merge_ta_fa_na_summary(fa_summary,ta_summary,na_summary):
     
    chat_completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI model designed to provide technical, fundamental, and news/events-based analysis to deliver actionable, long-term investment insights. Your role is to integrate financial health, competitive positioning, market trends, technical indicators, and relevant news/events into cohesive, data-driven recommendations for strategic, long-term investment strategies. Follow the specified structure and formatting guidelines to ensure clarity, professionalism, and usability."
                    "Formatting Requirements"
                    "Organized Headings and Subheadings: Separate sections clearly with headings (e.g., “Financial Overview,” “Technical Trends Analysis,” “Investment Recommendations”).Use descriptive subheadings for detailed insights (e.g., “Key Financial Metrics,” “Market Sentiment Analysis”)."
                    "Bullet Points and Numbered Lists:Use bullet points for comprehensive lists and numbered lists for prioritized recommendations."
                    "Formatting for Key Metrics and Indicators:Bold key financial terms (e.g., Earnings Per Share (EPS), Relative Strength Index (RSI))."
                    "Structure Guidelines:"
                    "Introduction"
                    "Briefly describe the asset’s profile, market sector, and its significance for long-term investors."
                    "Highlight the objective: integrating fundamental, technical, and events analysis for well-rounded investment decisions."
                    "Fundamental Analysis:"
                    """Fundamental Analysis
                    Financial Performance and Stability:

                    Review financial statements to assess profitability, solvency, and growth.
                    Focus on metrics such as revenue growth, net margins, and debt ratios.
                    Valuation Metrics:

                    Compare ratios like P/E, P/B, and Dividend Yield to industry norms.
                    Competitive Position and Risks:

                    Analyze market share, competitive advantages, and challenges.
                    Include a SWOT analysis for clarity on growth drivers and risks."""
                    """Technical Analysis
                    Long-Term Indicators:
                    MACD: Identify trends using signal line crossovers and price divergence.
                    ADX: Measure trend strength (e.g., readings > 20 = strong trend).
                    Bollinger Bands: Analyze volatility for entry/exit opportunities.
                    RSI: Use extended RSI values to determine overbought/oversold conditions.
                    SMA Crossovers: Monitor trends (e.g., "golden cross" patterns).
                    News and Events Analysis
                    Market Events and Macroeconomic Trends:

                    Summarize key news/events impacting the asset or sector (e.g., earnings releases, regulatory changes).
                    Highlight implications for long-term investment strategies.
                    Sector-Specific Developments:

                    Address sector-wide disruptions or opportunities (e.g., technological advances, geopolitical risks).
                    Integrated Analysis
                    Correlation of Insights:

                    Combine fundamental, technical, and event-based analysis to determine alignment or divergence.
                    Assess the impact of news/events on intrinsic value and technical trends.
                    Market Sentiment and Timing:

                    Evaluate whether market sentiment aligns with fundamental strengths or highlights discrepancies.
                    Long-Term Actionable Recommendations
                    Investment Decision:

                    Provide a clear Buy, Hold, or Sell recommendation, supported by key findings.
                    Entry and Exit Points:

                    Specify ideal entry/exit points using long-term technical indicators (e.g., SMA crossovers, RSI levels).
                    Risk Management:

                    Suggest risk mitigation strategies (e.g., stop-loss levels, portfolio diversification).
                    Performance Monitoring:

                    Highlight key updates (e.g., quarterly earnings) and technical changes (e.g., MACD signals) for regular review.
                    Style Requirements
                    Maintain a professional, analytical tone, avoiding personal opinions.
                    Use clear, concise language to ensure readability.
                    Minimize jargon; explain technical terms for clarity where needed."""
                ),

            },
            {
                "role": "user",
                "content": (
                    f"From and merge these texts, Technical Analysis: {ta_summary}, Fundamental Analysis: {fa_summary} and News/events: {na_summary}"
                ),
            },
        ]
     )

    response = chat_completion.choices[0].message.content
    return response



                
                



def merge_ta_fa_summary(gathered_data):
    today = date.today()
    formatted = today.strftime('%Y-%m-%d')  


    system_prompt = """
    As an AI assistant dedicated to supporting traders and investors, your task is to produce a structured, detailed market analysis in valid HTML format.

    The user will provide a JSON object containing all the data needed for analysis, including:
    - Ticker: The stock ticker symbol
    - Company: The company name
    - Timeframe: The analysis timeframe
    - Technical Analysis: Summary of technical analysis
    - Fundamental Analysis: Summary of fundamental analysis
    - Results: Technical indicator results (Summary, SMA, RSI, MACD, OBV, ADX)
    - UserSelectedWeights: An object containing the user-selected weights for Technical and Fundamental analyses, with each value between 0 and 1 (all weights sum to 1). Example:
    {
        "Technical Analysis": 0.6,
        "Fundamental Analysis": 0.4
    }

    You must parse this JSON data and use it to create a comprehensive investment report formatted as HTML.

    **Instructions:**
    - Parse the provided JSON data and use it to replace the placeholders in the HTML template below.
    - Extract the Ticker and Company information for the title.
    - Extract the Timeframe for the timeframe display.
    - Extract the Technical Analysis summary for the technical analysis section.
    - Extract Technical indicator results (SMA, RSI, MACD, OBV, ADX) for their dedicated sections.
    - Extract the Fundamental Analysis summary for the fundamental analysis section.
    - **For the 'Key Financial Metrics' section, you MUST ALWAYS display the following as responsive metric cards in a flex container. The core metrics to display as separate cards (even if a value is missing, show 'N/A'):**
        - Revenues
        - Sales
        - Income (Net Income or Earnings)
        - Gross Margins
        - Liabilities
        - Cash Flow (Operating Cash Flow)
        - Debt
        - Assets
    You may add additional metric cards if relevant, but these are required.
    - Each metric should appear in a “metric-card” styled <div> inside a parent <div class="metrics">, with a <div class="metric-title"> for the name and <div class="metric-value"> for the value.
    - Extract the user-selected weightings for Technical and Fundamental Analysis. Clearly display these weights in the 'User-Selected Analysis Weights' section.
    - When generating the overall investment recommendation, weigh the influence of Technical and Fundamental analyses according to the user-selected weights. The final recommendation (BUY, HOLD, or SELL) should be determined by a weighted synthesis of these two components, based on their assigned importance. Clearly communicate if the result is driven more by one analysis type due to a higher weighting.
    - The 'Integrated Analysis' and 'Alignment Assessment' sections should explicitly note which analysis type(s) had the greatest influence on the final recommendation, based on the weights.
    - The justification text for the recommendation must refer to the weights. For example: “Given the user’s preference to weigh Technical Analysis at 70%, the final recommendation relies primarily on chart patterns and indicator signals, despite mixed fundamentals.”
    - Return the complete HTML document as your response. Do not include any Markdown or plaintext. Do not leave out any required section, even if some are brief or data is missing.

    Your output must use <section>, <h2>, <h3>, <ul>, <li>, <table>, and <p> tags as appropriate. Use <strong> for key points.

    **Follow this professional HTML template exactly, replacing the placeholders with values parsed from the provided JSON:**

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive Investment Analysis</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0px;
                background-color: transparent;
            }
            .container {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                padding: 30px;
                margin-bottom: 30px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }
            h2 {
                color: #2c3e50;
                border-left: 5px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
                background-color: #f8f9fa;
                padding: 10px 15px;
                border-radius: 0 5px 5px 0;
            }
            h3 {
                color: #2c3e50;
                margin-top: 20px;
                border-bottom: 1px dashed #ddd;
                padding-bottom: 5px;
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .metrics {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin: 20px 0;
            }
            .metric-card {
                background-color: #f0f7ff;
                border-radius: 5px;
                padding: 15px;
                flex: 1;
                min-width: 200px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .metric-title {
                font-weight: bold;
                color: #2980b9;
                margin-bottom: 5px;
            }
            .metric-value {
                font-size: 1.2em;
                font-weight: bold;
            }
            .recommendation {
                font-weight: bold;
                font-size: 1.1em;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                text-align: center;
            }
            .buy {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .hold {
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
            }
            .sell {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .indicator {
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 5px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
            }
            .indicator h4 {
                margin-top: 0;
                color: #2980b9;
            }
            .summary-box {
                background-color: #e8f4fd;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }
            .timeframe {
                font-weight: bold;
                color: #2c3e50;
                background-color: #e8f4fd;
                padding: 5px 10px;
                border-radius: 3px;
                display: inline-block;
                margin-bottom: 15px;
            }
            .weights-section {
                background-color: #f0f4f9;
                border-left: 4px solid #2980b9;
                margin-bottom: 30px;
                padding: 15px;
                border-radius: 0 5px 5px 0;
            }
            .footnote {
                font-size: 0.9em;
                font-style: italic;
                color: #6c757d;
                margin-top: 30px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
            }
            .highlight {
                background-color: #ffeaa7;
                padding: 2px 4px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Comprehensive Investment Analysis: [TICKER_PLACEHOLDER] - [COMPANY_PLACEHOLDER]</h1>
            <div class="timeframe">Analysis Timeframe: [TIMEFRAME_PLACEHOLDER]</div>
            
            <section class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p>[SUMMARY_PLACEHOLDER]</p>
                </div>
                <div class="recommendation [RECOMMENDATION_CLASS_PLACEHOLDER]">
                    RECOMMENDATION: [RECOMMENDATION_PLACEHOLDER]
                    <br>
                    <span style="font-size:0.95em; font-weight:normal;">
                    <strong>Note:</strong> This recommendation is primarily driven by 
                    <span class="highlight">
                        [DOMINANT_ANALYSIS_TYPE_PLACEHOLDER]
                    </span>
                    analysis, as selected by the user’s weightings.
                    </span>
                </div>
            </section>

            <section class="section">
                <h2>Fundamental Analysis</h2>
                <div id="fundamental-analysis">
                    [FUNDAMENTAL_ANALYSIS_PLACEHOLDER]
                </div>
                <h3>Key Financial Metrics</h3>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">Revenues</div>
                        <div class="metric-value">[REVENUES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Sales</div>
                        <div class="metric-value">[SALES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Income</div>
                        <div class="metric-value">[INCOME_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Gross Margins</div>
                        <div class="metric-value">[GROSS_MARGINS_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Liabilities</div>
                        <div class="metric-value">[LIABILITIES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Cash Flow</div>
                        <div class="metric-value">[CASH_FLOW_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Debt</div>
                        <div class="metric-value">[DEBT_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Assets</div>
                        <div class="metric-value">[ASSETS_PLACEHOLDER]</div>
                    </div>
                    [OPTIONAL_ADDITIONAL_METRIC_CARDS_PLACEHOLDER]
                </div>
                <h3>Valuation Analysis</h3>
                <div class="indicator">
                    <h4>Summary</h4>
                    <p>[VALUATION_METRICS_PLACEHOLDER]</p>
                </div>
            </section>

            <section class="section">
                <h2>Technical Analysis</h2>
                <div id="technical-analysis">
                    [TECHNICAL_ANALYSIS_PLACEHOLDER]
                </div>
                <h3>Technical Indicators</h3>
                <div class="indicator">
                    <h4>SMA (Simple Moving Average)</h4>
                    <p>[SMA_ANALYSIS_PLACEHOLDER]</p>
                </div>
                <div class="indicator">
                    <h4>RSI (Relative Strength Index)</h4>
                    <p>[RSI_ANALYSIS_PLACEHOLDER]</p>
                </div>
                <div class="indicator">
                    <h4>MACD (Moving Average Convergence Divergence)</h4>
                    <p>[MACD_ANALYSIS_PLACEHOLDER]</p>
                </div>
                <div class="indicator">
                    <h4>OBV (On-Balance Volume)</h4>
                    <p>[OBV_ANALYSIS_PLACEHOLDER]</p>
                </div>
                <div class="indicator">
                    <h4>ADX (Average Directional Index)</h4>
                    <p>[ADX_ANALYSIS_PLACEHOLDER]</p>
                </div>
            </section>

            <section class="section">
                <h2>Integrated Analysis</h2>
                <p>[INTEGRATED_ANALYSIS_PLACEHOLDER]</p>
                <div class="weights-section">
                    <h3>User-Selected Analysis Weights</h3>
                    <ul>
                        <li><strong>Technical Analysis Weight:</strong> [TECHNICAL_WEIGHT_PLACEHOLDER]</li>
                        <li><strong>Fundamental Analysis Weight:</strong> [FUNDAMENTAL_WEIGHT_PLACEHOLDER]</li>
                    </ul>
                    <p>
                    These weights determined the overall influence of each analysis type on the final investment recommendation.
                    The report will highlight which analysis category most influenced the recommendation, based on the user’s preferences.
                    </p>
                </div>
                <h3>Alignment Assessment</h3>
                <table>
                    <tr>
                        <th>Analysis Type</th>
                        <th>Outlook</th>
                        <th>Confidence</th>
                    </tr>
                    <tr>
                        <td>Technical</td>
                        <td>[TECHNICAL_OUTLOOK_PLACEHOLDER]</td>
                        <td>[TECHNICAL_CONFIDENCE_PLACEHOLDER]</td>
                    </tr>
                    <tr>
                        <td>Fundamental</td>
                        <td>[FUNDAMENTAL_OUTLOOK_PLACEHOLDER]</td>
                        <td>[FUNDAMENTAL_CONFIDENCE_PLACEHOLDER]</td>
                    </tr>
                    <tr>
                        <td><strong>Overall</strong></td>
                        <td><strong>[OVERALL_OUTLOOK_PLACEHOLDER]</strong></td>
                        <td><strong>[OVERALL_CONFIDENCE_PLACEHOLDER]</strong></td>
                    </tr>
                </table>
                <h3>Investment Recommendation</h3>
                <div class="summary-box">
                    <p><strong>Recommendation:</strong> [DETAILED_RECOMMENDATION_PLACEHOLDER]</p>
                    <p><strong>Entry Points:</strong> [ENTRY_POINTS_PLACEHOLDER]</p>
                    <p><strong>Exit Strategy:</strong> [EXIT_STRATEGY_PLACEHOLDER]</p>
                    <p><strong>Risk Management:</strong> [RISK_MANAGEMENT_PLACEHOLDER]</p>
                </div>
            </section>
            <div class="footnote">""" f"""
                <p>This investment analysis was generated on {formatted}, and incorporates available data as of this date. All investment decisions should be made in conjunction with personal financial advice and risk tolerance assessments.</p>
            </div>
        </div>
    </body>
    </html>

    Recommendation Logic:

    When generating the “Recommendation” (Buy, Hold, Sell), synthesize and weigh the findings from the Fundamental Analysis, and Technical Indicator Analysis according to the provided weights.

    - **Key Financial Metrics Requirement:**  
    In the 'Fundamental Analysis' section, always display the following metrics as individual metric-cards in a flex container: Revenues, Sales, Income, Gross Margins, Liabilities, Cash Flow, Debt, Assets (with "N/A" if missing). Additional metrics may be added as more cards.

    If one weighting is clearly dominant (e.g., Fundamental Analysis = 0.7), emphasize in the summary and recommendation that the final decision is mainly driven by that analysis type.

    The “Integrated Analysis” and “Alignment Assessment” sections should explicitly note which analysis types had the greatest influence on the final recommendation, based on the weights.

    Justification:

    The justification text for the recommendation must refer to the weights. For example:
    “Given the user’s preference to weigh Fundamental Analysis at 60%, the final recommendation relies primarily on the company’s strong balance sheet and valuation ratios, despite any recent negative news events.”

    Return the complete HTML document as your response.


    """
    user_message = f"The data to analyse: {json.dumps(gathered_data)}"

    chat_completion = client.chat.completions.create(
        model="gpt-4.1",  # Ensure that you use a model available in your OpenAI subscription
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            },
        ]
    )

    # Extract and return the AI-generated response
    response = chat_completion.choices[0].message.content
    return response

                        
                

        #if t_col1.button("Technical Analysis"):
            #analysis_type = "Technical Analysis"
        #elif n_col2.button("News and Events"):
            #analysis_type = "News and Events"
def clean_html_response(response):
    # Remove markdown formatting from response
    if response.startswith("```html"):
        response = response.lstrip("```html").strip()
    if response.endswith("```"):
        response = response.rstrip("```").strip()
    return response

def txt_conclusion(news_summary,company_name):
    # OpenAI API call to create a merged summary
    chat_completion = client.chat.completions.create(
        model="gpt-4.1",  # Ensure that you use a model available in your OpenAI subscription
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI model specializing in investment insights, tasked with analyzing recent news and events about a specified company and providing recommendations for investors. Your goal is to review relevant data, including press releases, market trends, earnings reports, and industry events, to assess the companys financial health, growth prospects, and potential risks. From this data, you will determine an ideal investor position (e.g., buy, hold, or sell)."
                    "Instructions:"
                    "Data Collection: Search for and analyze recent press releases, earnings reports, regulatory filings, and news articles regarding the specified company. Focus on the following:"
                    "Financial Performance: Look for quarterly or annual earnings, revenue, and profit trends."
                    "Product & Service Developments: Identify any new product launches, service expansions, or market innovations."
                    "Management Statements: Note key statements from executives or significant personnel changes that might impact the companys direction."
                    "Industry Events & Competitor Actions: Examine news of industry-wide developments, competitor performance, and market conditions."
                    "Regulatory & Legal News: Assess any legal challenges, regulatory updates, or policy changes impacting the company."
                    "Sentiment Analysis: Evaluate the tone and sentiment of the news data—whether positive, neutral, or negative. Gauge investor confidence and sentiment trends as reflected in the media."
                    "Market Impact: Summarize any immediate or anticipated effects of recent events on the companys stock price, including short-term volatility, potential growth indicators, or risk factors that could affect long-term performance."
                    "Investor Recommendation:"
                    "Buy: Recommend if positive news, strong financial performance, and promising growth potential outweigh risks."
                    "Hold: Suggest if there are mixed indicators, with potential growth tempered by risks or uncertain factors."
                    "Sell: Advise if significant risks, declining performance, or negative news dominate, suggesting potential for downturn."
                    "Final Conclusion: Provide a clear summary and reasoning behind the recommended position, addressing key data points and highlighting the rationale for an investor's action."
                    "Additional Sources: A separate section listing sources like press releases and opinions from the mentioned platforms, ensuring proper citations."
                    #Add Press releases, investor oppinions (X), First World Pharma, Bloomberg, Market Watch, seperate segment,add sources, add graphs
                    
                ),
            },
            {
                "role": "user",
                "content": (
                    f"News and Events Summary for {company_name}:\n{news_summary}\n\n"   
                ),
            },
        ]
    )

# Extract and return the AI-generated response
    response = chat_completion.choices[0].message.content
    return response 

def twitter_summary(gathered_data):
    system_prompt = twitter_system_prompt

    user_message = f"The data to analyse: {json.dumps(gathered_data)}"

    chat_completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_message }
        ]
        
    )

    response = chat_completion.choices[0].message.content
    return response

def txt_conclusion2(gathered_data):
    today = date.today()
    formatted = today.strftime('%Y-%m-%d')

    # OpenAI API call to create a merged summary
    system_prompt = """ As an AI assistant dedicated to supporting traders and investors, your task is to produce a structured, detailed market analysis in valid HTML format. Focus exclusively on synthesizing recent news and events related to the selected stock and its sector. Do not include technical or fundamental analysis.

    The user will provide a JSON object containing all the data needed for analysis, including:
    - Ticker: The stock ticker symbol
    - Company: The company name
    - Timeframe: The analysis timeframe
    - NewsData: A summary of all significant news and events for the company and its sector.
    - SignificantEvents: A list of recent, impactful events affecting the company or its market environment.

    You must parse this JSON data and use it to create a comprehensive investment report formatted as HTML.

    **Instructions:**
    - Parse the provided JSON data and use it to replace the placeholders in the HTML template below.
    - Extract the Ticker and Company information for the title.
    - Extract the Timeframe for the timeframe display.
    - Extract the NewsData for the News and Events Analysis section.
    - Extract the SignificantEvents list for the 'Recent Significant Events' section.
    - Generate a summary and investment recommendation (BUY, HOLD, or SELL) based exclusively on news and events. Clearly justify your reasoning by referencing the reported news and events.
    - The 'Integrated Analysis' section should synthesize all news and event signals into a final outlook and recommendation.
    - Return the complete HTML document as your response. Do not include any Markdown or plaintext. Do not leave out any required section, even if some are brief or data is missing.

    Your output must use <section>, <h2>, <h3>, <ul>, <li>, <table>, and <p> tags as appropriate. Use <strong> for key points.

    **Follow this professional HTML template exactly, replacing the placeholders with values parsed from the provided JSON:**

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive News & Events Investment Analysis</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0px;
                background-color: transparent;
            }
            .container {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                padding: 30px;
                margin-bottom: 30px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }
            h2 {
                color: #2c3e50;
                border-left: 5px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
                background-color: #f8f9fa;
                padding: 10px 15px;
                border-radius: 0 5px 5px 0;
            }
            h3 {
                color: #2c3e50;
                margin-top: 20px;
                border-bottom: 1px dashed #ddd;
                padding-bottom: 5px;
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .recommendation {
                font-weight: bold;
                font-size: 1.1em;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                text-align: center;
            }
            .buy {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .hold {
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
            }
            .sell {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .summary-box {
                background-color: #e8f4fd;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }
            .timeframe {
                font-weight: bold;
                color: #2c3e50;
                background-color: #e8f4fd;
                padding: 5px 10px;
                border-radius: 3px;
                display: inline-block;
                margin-bottom: 15px;
            }
            .footnote {
                font-size: 0.9em;
                font-style: italic;
                color: #6c757d;
                margin-top: 30px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
            }
            .highlight {
                background-color: #ffeaa7;
                padding: 2px 4px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Comprehensive News & Events Investment Analysis: [TICKER_PLACEHOLDER] - [COMPANY_PLACEHOLDER]</h1>
            <div class="timeframe">Analysis Timeframe: [TIMEFRAME_PLACEHOLDER]</div>
            
            <section class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p>[SUMMARY_PLACEHOLDER]</p>
                </div>
                <div class="recommendation [RECOMMENDATION_CLASS_PLACEHOLDER]">
                    RECOMMENDATION: [RECOMMENDATION_PLACEHOLDER]
                    <br>
                    <span style="font-size:0.95em; font-weight:normal;">
                    <strong>Note:</strong> This recommendation is based solely on the latest news and event signals.
                    </span>
                </div>
            </section>
            
            <section class="section">
                <h2>News and Events Analysis</h2>
                <div id="news-analysis">
                    [NEWS_ANALYSIS_PLACEHOLDER]
                </div>
                
                <h3>Recent Significant Events</h3>
                <ul>
                    [SIGNIFICANT_EVENTS_PLACEHOLDER]
                </ul>
            </section>
            
            <section class="section">
                <h2>Integrated Analysis</h2>
                <p>[INTEGRATED_ANALYSIS_PLACEHOLDER]</p>
                <h3>Investment Recommendation</h3>
                <div class="summary-box">
                    <p><strong>Recommendation:</strong> [DETAILED_RECOMMENDATION_PLACEHOLDER]</p>
                    <p><strong>Entry Points:</strong> [ENTRY_POINTS_PLACEHOLDER]</p>
                    <p><strong>Exit Strategy:</strong> [EXIT_STRATEGY_PLACEHOLDER]</p>
                    <p><strong>Risk Management:</strong> [RISK_MANAGEMENT_PLACEHOLDER]</p>
                </div>
            </section>
            
            <div class="footnote"> """ f"""
                <p>This investment analysis was generated on {formatted}, and incorporates available news and event data as of this date. All investment decisions should be made in conjunction with personal financial advice and risk tolerance assessments.</p>
            </div>
        </div>
    </body>
    </html>
    """

    user_message = f"The data to analyse: {json.dumps(gathered_data)}"

    chat_completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

# Extract and return the AI-generated response
    response = chat_completion.choices[0].message.content
    return response 

def txt_twitter_conclusion(gathered_data):
    today = date.today()
    formatted = today.strftime('%Y-%m-%d')

    # If you don't have a global `client`, uncomment:
    # if openai_api_key:
    #     client = OpenAI(api_key=openai_api_key)
    # else:
    #     raise ValueError("OpenAI API key required if no global client is set.")

    system_prompt = """ As an AI assistant dedicated to supporting traders and investors, your task is to produce a structured, detailed market analysis in valid HTML format. Focus exclusively on synthesizing recent news and events related to the selected stock and its sector. Do not include technical or fundamental analysis.

    The user will provide a JSON object containing all the data needed for analysis, including:
    - Ticker: The stock ticker symbol
    - Company: The company name
    - Timeframe: The analysis timeframe
    - News and Events Summary (NewsData): A summary of all significant news and events for the company and its sector.
    - SignificantEvents: A list of recent, impactful events affecting the company or its market environment.
    - Twitter Total Hits: Total tweets found in the timeframe (optional)
    - Top Tweets by Likes: Array of objects with fields {likes, retweets, createdAt, author, text, url, sentiment?, reason?} (optional)
    - Top Tweets by Retweets: Same shape as above (optional)
    - Twitter Sentiment Summary: Object with {Positive, Neutral, Negative, Analyzed Count} (optional)

    You must parse this JSON data and use it to create a comprehensive investment report formatted as HTML.

    **Instructions:**
    - Parse the provided JSON and use it to replace placeholders in the HTML template below.
    - Extract Ticker, Company, and Timeframe.
    - Use News and Events Summary for the News and Events Analysis section.
    - Use SignificantEvents for the 'Recent Significant Events' list.
    - If Twitter/X data is provided, render the 'Twitter/X Pulse' section:
        - Show a one-line summary including total hits and the sentiment roll-up (Positive / Neutral / Negative).
        - Build two compact tables:
            1) Top Tweets by Likes
            2) Top Tweets by Retweets
          Each table should include: Created At, Author, Likes, Retweets, Sentiment (if provided), a short snippet (truncate long text), and a link (url).
        - If any Twitter fields are missing, include a brief note indicating data was unavailable.
    - Generate a summary and investment recommendation (BUY, HOLD, or SELL) based exclusively on news and events (and Twitter/X investor sentiment if present). Justify clearly by referencing the reported news, events, and sentiment signals.
    - The 'Integrated Analysis' should synthesize all signals into a final outlook and recommendation.
    - Return the complete HTML document. No Markdown or plaintext. Do not omit required sections even if data is sparse.

    Your output must use <section>, <h2>, <h3>, <ul>, <li>, <table>, <thead>, <tbody>, <tr>, <th>, <td>, and <p> tags where appropriate. Use <strong> for key points.

    **Follow this professional HTML template exactly, replacing the placeholders with values parsed from the provided JSON:**

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive News & Events Investment Analysis</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0px;
                background-color: transparent;
            }
            .container {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                padding: 30px;
                margin-bottom: 30px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }
            h2 {
                color: #2c3e50;
                border-left: 5px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
                background-color: #f8f9fa;
                padding: 10px 15px;
                border-radius: 0 5px 5px 0;
            }
            h3 {
                color: #2c3e50;
                margin-top: 20px;
                border-bottom: 1px dashed #ddd;
                padding-bottom: 5px;
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .recommendation {
                font-weight: bold;
                font-size: 1.1em;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                text-align: center;
            }
            .buy {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .hold {
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
            }
            .sell {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .summary-box {
                background-color: #e8f4fd;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }
            .timeframe {
                font-weight: bold;
                color: #2c3e50;
                background-color: #e8f4fd;
                padding: 5px 10px;
                border-radius: 3px;
                display: inline-block;
                margin-bottom: 15px;
            }
            .footnote {
                font-size: 0.9em;
                font-style: italic;
                color: #6c757d;
                margin-top: 30px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
            }
            .highlight {
                background-color: #ffeaa7;
                padding: 2px 4px;
                border-radius: 3px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            th, td {
                border: 1px solid #e2e2e2;
                padding: 8px;
                text-align: left;
                vertical-align: top;
            }
            thead th {
                background-color: #f1f5f9;
            }
            .muted { color: #6c757d; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Comprehensive News & Events Investment Analysis: [TICKER_PLACEHOLDER] - [COMPANY_PLACEHOLDER]</h1>
            <div class="timeframe">Analysis Timeframe: [TIMEFRAME_PLACEHOLDER]</div>
            
            <section class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p>[SUMMARY_PLACEHOLDER]</p>
                </div>
                <div class="recommendation [RECOMMENDATION_CLASS_PLACEHOLDER]">
                    RECOMMENDATION: [RECOMMENDATION_PLACEHOLDER]
                    <br>
                    <span style="font-size:0.95em; font-weight:normal;">
                    <strong>Note:</strong> This recommendation is based solely on the latest news, event signals, and (if provided) investor sentiment from Twitter/X.
                    </span>
                </div>
            </section>
            
            <section class="section">
                <h2>News and Events Analysis</h2>
                <div id="news-analysis">
                    [NEWS_ANALYSIS_PLACEHOLDER]
                </div>
                
                <h3>Recent Significant Events</h3>
                <ul>
                    [SIGNIFICANT_EVENTS_PLACEHOLDER]
                </ul>
            </section>

            <section class="section">
                <h2>Twitter/X Pulse</h2>
                <p>[TWITTER_SUMMARY_PLACEHOLDER]</p>

                <h3>Top Tweets by Likes</h3>
                <div>
                    [TWITTER_TABLE_LIKES_PLACEHOLDER]
                </div>

                <h3>Top Tweets by Retweets</h3>
                <div>
                    [TWITTER_TABLE_RETWEETS_PLACEHOLDER]
                </div>

                <p class="muted">[TWITTER_SENTIMENT_SUMMARY_PLACEHOLDER]</p>
            </section>
            
            <section class="section">
                <h2>Integrated Analysis</h2>
                <p>[INTEGRATED_ANALYSIS_PLACEHOLDER]</p>
                <h3>Investment Recommendation</h3>
                <div class="summary-box">
                    <p><strong>Recommendation:</strong> [DETAILED_RECOMMENDATION_PLACEHOLDER]</p>
                    <p><strong>Entry Points:</strong> [ENTRY_POINTS_PLACEHOLDER]</p>
                    <p><strong>Exit Strategy:</strong> [EXIT_STRATEGY_PLACEHOLDER]</p>
                    <p><strong>Risk Management:</strong> [RISK_MANAGEMENT_PLACEHOLDER]</p>
                </div>
            </section>
            
            <div class="footnote"> """ f"""
                <p>This investment analysis was generated on {formatted}, and incorporates available news, event, and investor sentiment data (if provided) as of this date. All investment decisions should be made in conjunction with personal financial advice and risk tolerance assessments.</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Give the model the entire gathered_data JSON as the single user message
    user_message = f"The data to analyse: {json.dumps(gathered_data)}"

    chat_completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_message }
        ]
        
    )

    response = chat_completion.choices[0].message.content
    return response
    

def merge_news_and_technical_analysis_summary(gathered_data):
    today = date.today()
    formatted = today.strftime('%Y-%m-%d')


    system_prompt = """
    As an AI assistant dedicated to supporting traders and investors, your task is to produce a structured, detailed market analysis in valid HTML format.
    Merge the latest news and technical analysis for the selected stock, ensuring all insights are clearly organized.

    The user will provide a JSON object containing all the data needed for analysis, including:
    - Ticker: The stock ticker symbol
    - Company: The company name
    - Timeframe: The analysis timeframe
    - Technical Analysis: Summary of technical analysis
    - News data: News summaries for the company and related companies/sectors
    - Results: Technical indicator results (Summary, SMA, RSI, MACD, OBV, ADX)
    - UserSelectedWeights: An object containing the user-selected weights for Technical, Fundamental, and News analyses, with each value between 0 and 1 (all weights sum to 1). Example:
    {
        "Technical Analysis": 0.4,
        "Fundamental Analysis": 0.4,
        "News and Events": 0.2
    }

    You must parse this JSON data and use it to create a comprehensive investment report formatted as HTML.

    Follow this HTML template exactly, replacing the placeholder content with information derived from the JSON data:

    Your output must use <section>, <h2>, <h3>, <ul>, <li>, <table>, and <p> tags as appropriate. Use <strong> for key points.
    Follow this professional HTML template exactly, replacing the placeholders with values parsed from the provided JSON:

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive Investment Analysis</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0px;
                background-color: transparent;
            }
            .container {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                padding: 30px;
                margin-bottom: 30px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }
            h2 {
                color: #2c3e50;
                border-left: 5px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
                background-color: #f8f9fa;
                padding: 10px 15px;
                border-radius: 0 5px 5px 0;
            }
            h3 {
                color: #2c3e50;
                margin-top: 20px;
                border-bottom: 1px dashed #ddd;
                padding-bottom: 5px;
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            ul, ol {
                padding-left: 25px;
            }
            ul li, ol li {
                margin-bottom: 8px;
            }
            .recommendation {
                font-weight: bold;
                font-size: 1.1em;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                text-align: center;
            }
            .buy {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .hold {
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
            }
            .sell {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .metrics {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin: 20px 0;
            }
            .metric-card {
                background-color: #f0f7ff;
                border-radius: 5px;
                padding: 15px;
                flex: 1;
                min-width: 200px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .metric-title {
                font-weight: bold;
                color: #2980b9;
                margin-bottom: 5px;
            }
            .metric-value {
                font-size: 1.2em;
                font-weight: bold;
            }
            .chart-container {
                margin: 20px 0;
                text-align: center;
            }
            .footnote {
                font-size: 0.9em;
                font-style: italic;
                color: #6c757d;
                margin-top: 30px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
            }
            strong {
                color: #2980b9;
            }
            .highlight {
                background-color: #ffeaa7;
                padding: 2px 4px;
                border-radius: 3px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .summary-box {
                background-color: #e8f4fd;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }
            .indicator {
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 5px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
            }
            .indicator h4 {
                margin-top: 0;
                color: #2980b9;
            }
            .timeframe {
                font-weight: bold;
                color: #2c3e50;
                background-color: #e8f4fd;
                padding: 5px 10px;
                border-radius: 3px;
                display: inline-block;
                margin-bottom: 15px;
            }
            .weights-section {
                background-color: #f0f4f9;
                border-left: 4px solid #2980b9;
                margin-bottom: 30px;
                padding: 15px;
                border-radius: 0 5px 5px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Comprehensive Investment Analysis: [TICKER_PLACEHOLDER] - [COMPANY_PLACEHOLDER]</h1>
            
            <div class="timeframe">Analysis Timeframe: [TIMEFRAME_PLACEHOLDER]</div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p>[SUMMARY_PLACEHOLDER]</p>
                </div>
                
                <div class="recommendation [RECOMMENDATION_CLASS_PLACEHOLDER]">
                    RECOMMENDATION: [RECOMMENDATION_PLACEHOLDER]
                    <br>
                    <span style="font-size:0.95em; font-weight:normal;">
                    <strong>Note:</strong> This recommendation is primarily driven by 
                    <span class="highlight">
                        [DOMINANT_ANALYSIS_TYPE_PLACEHOLDER]
                    </span> 
                    analysis, as selected by the user’s weightings.
                    </span>
                </div>
            </div>
            
            <div class="section">
                <h2>Technical Analysis</h2>
                <div id="technical-analysis">
                    [TECHNICAL_ANALYSIS_PLACEHOLDER]
                </div>
                
                <h3>Technical Indicators</h3>
                
                <div class="indicator">
                    <h4>SMA (Simple Moving Average)</h4>
                    <p>[SMA_ANALYSIS_PLACEHOLDER]</p>
                </div>
                
                <div class="indicator">
                    <h4>RSI (Relative Strength Index)</h4>
                    <p>[RSI_ANALYSIS_PLACEHOLDER]</p>
                </div>
                
                <div class="indicator">
                    <h4>MACD (Moving Average Convergence Divergence)</h4>
                    <p>[MACD_ANALYSIS_PLACEHOLDER]</p>
                </div>
                
                <div class="indicator">
                    <h4>OBV (On-Balance Volume)</h4>
                    <p>[OBV_ANALYSIS_PLACEHOLDER]</p>
                </div>
                
                <div class="indicator">
                    <h4>ADX (Average Directional Index)</h4>
                    <p>[ADX_ANALYSIS_PLACEHOLDER]</p>
                </div>
                
                <h3>Support and Resistance Levels</h3>
                <table>
                    <tr>
                        <th>Level Type</th>
                        <th>Price Point</th>
                        <th>Strength</th>
                    </tr>
                    [SUPPORT_RESISTANCE_PLACEHOLDER]
                </table>
            </div>
            
            <div class="section">
                <h2>News and Events Analysis</h2>
                <div id="news-analysis">
                    [NEWS_ANALYSIS_PLACEHOLDER]
                </div>
                
                <h3>Recent Significant Events</h3>
                <ul>
                    [SIGNIFICANT_EVENTS_PLACEHOLDER]
                </ul>
            </div>
            
            <div class="section">
                <h2>Integrated Analysis</h2>
                <p>[INTEGRATED_ANALYSIS_PLACEHOLDER]</p>
                
                <div class="weights-section">
                    <h3>User-Selected Analysis Weights</h3>
                    <ul>
                        <li><strong>Technical Analysis Weight:</strong> [TECHNICAL_WEIGHT_PLACEHOLDER]</li>
                        <li><strong>News and Events Weight:</strong> [NEWS_WEIGHT_PLACEHOLDER]</li>
                    </ul>
                    <p>
                    These weights determined the overall influence of each analysis type on the final investment recommendation.
                    The report will highlight which analysis category most influenced the recommendation, based on the user’s preferences.
                    </p>
                </div>
                
                <h3>Alignment Assessment</h3>
                <table>
                    <tr>
                        <th>Analysis Type</th>
                        <th>Outlook</th>
                        <th>Confidence</th>
                    </tr>
                    <tr>
                        <td>Technical</td>
                        <td>[TECHNICAL_OUTLOOK_PLACEHOLDER]</td>
                        <td>[TECHNICAL_CONFIDENCE_PLACEHOLDER]</td>
                    </tr>
                    <tr>
                        <td>News/Events</td>
                        <td>[NEWS_OUTLOOK_PLACEHOLDER]</td>
                        <td>[NEWS_CONFIDENCE_PLACEHOLDER]</td>
                    </tr>
                    <tr>
                        <td><strong>Overall</strong></td>
                        <td><strong>[OVERALL_OUTLOOK_PLACEHOLDER]</strong></td>
                        <td><strong>[OVERALL_CONFIDENCE_PLACEHOLDER]</strong></td>
                    </tr>
                </table>
                
                <h3>Investment Recommendation</h3>
                <div class="summary-box">
                    <p><strong>Recommendation:</strong> [DETAILED_RECOMMENDATION_PLACEHOLDER]</p>
                    
                    <p><strong>Entry Points:</strong> [ENTRY_POINTS_PLACEHOLDER]</p>
                    
                    <p><strong>Exit Strategy:</strong> [EXIT_STRATEGY_PLACEHOLDER]</p>
                    
                    <p><strong>Risk Management:</strong> [RISK_MANAGEMENT_PLACEHOLDER]</p>
                </div>
            </div>
            
            <div class="footnote">""" f"""
                <p>This investment analysis was generated on {formatted}, and incorporates available data as of this date. All investment decisions should be made in conjunction with personal financial advice and risk tolerance assessments.</p>
            </div>
        </div>
    </body>
    </html>

    Instructions:
    - Parse the provided JSON data and use it to replace the placeholders in the HTML template.
    - Extract the Ticker and Company information for the title.
    - Extract the Timeframe for the timeframe display.
    - Extract the Technical Analysis summary for the technical analysis section.
    - Extract Technical indicator results (SMA, RSI, MACD, OBV, ADX) for their dedicated sections.
    - Extract News data for the news analysis section.
    - Extract the user-selected weightings for Technical and News/Events. Clearly display these weights in the 'User-Selected Analysis Weights' section.
    - When generating the overall investment recommendation, weigh the influence of Technical and News/Events analyses according to the user-selected weights. The final recommendation (BUY, HOLD, or SELL) should be determined by a weighted synthesis of these two components, based on their assigned importance. Clearly communicate if the result is driven more by one analysis type due to a higher weighting.
    - The 'Integrated Analysis' and 'Alignment Assessment' sections should explicitly note which analysis type(s) had the greatest influence on the final recommendation, based on the weights.
    - The justification text for the recommendation must refer to the weights. For example: “Given the user’s preference to weigh Technical Analysis at 70%, the final recommendation relies primarily on chart patterns and indicator signals, despite mixed news sentiment.”

    Recommendation Logic:

    When generating the “Recommendation” (Buy, Hold, Sell), synthesize and weigh the findings from the Technical Analysis, and News/Events Analysis according to the provided weights.

    If one weighting is clearly dominant (e.g., Technical Analysis = 0.7), emphasize in the summary and recommendation that the final decision is mainly driven by that analysis type.

    The “Integrated Analysis” and “Alignment Assessment” sections should explicitly note which analysis types had the greatest influence on the final recommendation, based on the weights.

    Justification:

    The justification text for the recommendation must refer to the weights. For example:
    “Given the user’s preference to weigh Fundamental Analysis at 60%, the final recommendation relies primarily on the company’s strong balance sheet and valuation ratios, despite any recent negative news events.”

    Return the complete HTML document as your response.
    """

    user_message = f"The data to analyse: {json.dumps(gathered_data)}"
    # OpenAI API call to create a merged summary
    chat_completion = client.chat.completions.create(
        model="gpt-4.1",  # Ensure that you use a model available in your OpenAI subscription
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            },
        ]
    )

    # Extract and return the AI-generated response
    response = chat_completion.choices[0].message.content
    return response

def get_start_date(timeframe: str) -> str:
    today = datetime.today()
    
    if timeframe == "3 Months":
        start_date = today - relativedelta(months=3)
    elif timeframe == "6 Months":
        start_date = today - relativedelta(months=6)
    elif timeframe == "1 Year":
        start_date = today - relativedelta(years=1)
    else:
        raise ValueError("Invalid timeframe")

    return start_date.strftime('%Y-%m-%d')

def generate_company_news_message(company_name, time_period):
    # Define the messages for different time periods 
    start_date = get_start_date(time_period)
    query = f'"{company_name}" (news OR tweet OR earnings OR downgrade OR acquisition) after:{start_date}'

    
    params = {
        "q": query,
        "api_key": "6bbbb0268f96b1336ac50343fe6ef93a286a74d0f64c3d09fca848c5d62c9cce"
    }

    print(f"\n🔍 Searching SerpAPI with query: {query}")
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        print(results)
        print("✅ SerpAPI search completed")
    except Exception as e:
        print(f"❌ SerpAPI error: {e}")
        return
    
    news = []
    for item in results.get("organic_results", []):
        title = item.get("title", "")
        date = item.get("date", "")
        link = item.get("link", "")
        print(f"\n📄 Scraping: {title}")
        content = extract_diffbot_data(link)

        news.append({
            "title": title,
            "date": date,
            "link": link,
            "content": content
        })

        time.sleep(4)

    #Webhook payload
    payload = {
        "news": news,
        "company": company_name,
        "time_frame": time_period
    }

    chat_completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": f"""You will receive a set of news items, each including the following:

                Company name

                Date of publication

                News title

                Full news content

                URL link to the original source

                Your task is to analyze all news articles published over the past{time_period} . From these, extract and summarize the most relevant events that could potentially affect {company_name}'s stock price, either positively or negatively. Pay particular attention to financial updates, regulatory developments, strategic decisions, product milestones, partnerships, or public sentiment shifts or tweets.

                Present your output in the following format:

                [Company Name] Announcements – Key Events Timeline

                [Date]: [Event Title]
                [Provide a brief description of the event, emphasizing its relevance, such as its impact on stock performance, investor confidence, financial results, or industry positioning.]
                [Include the URL to the original source.]

                [Date]: [Event Title]
                [Summarize another key event, noting details such as earnings announcements, trial outcomes, leadership changes, or market expansions.]
                [Include the URL to the original source.]

                [Date]: [Event Title]
                [Describe the event’s importance, including aspects like M&A activity, regulatory updates, or attendance at industry conferences.]
                [Include the URL to the original source.]

                """
            },
            {
                "role": "user",
                "content": f"News: {payload} Company: {company_name} Time Frame: {time_period}"
            },
        ]
    )
    response = chat_completion.choices[0].message.content
    return response

    # print("\n📤 Sending to Make.com webhook...")
    # webhook_url = "https://hook.eu2.make.com/s4xsnimg9v87rrrckcwo88d9k57186q6"
    # try:
    #     response = requests.post(webhook_url, json=payload)
    #     if response.status_code == 200:
    #         print("✅ Successfully posted to the webhook.")
    #     else:
    #         print(f"❌ Webhook error: {response.status_code} - {response.text}")
    # except Exception as e:
    #     print(f"❌ Error posting to webhook: {e}")


 
    # print(response.text)

    # time.sleep(65)

    # credentials_dict = {
    #     "type": type_sa,
    #     "project_id": project_id,
    #     "private_key_id": private_key_id,
    #     "private_key": private_key,
    #     "client_email": client_email,
    #     "client_id": client_id,
    #     "auth_uri": auth_uri,
    #     "token_uri": token_uri,
    #     "auth_provider_x509_cert_url": auth_provider_x509_cert_url,
    #     "client_x509_cert_url": client_x509_cert_url,
    #     "universe_domain": universe_domain
    # }
    # credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, ["https://www.googleapis.com/auth/spreadsheets"])

    # gc = gspread.authorize(credentials)
    # sh = gc.open_by_url(google_sheet_url)
    # previous = sh.sheet1.get('A2')
    # future = sh.sheet1.get('B2')
          
    # # chats = client.chat.completions.create(
    # #     model="gpt-4o",
    # #     messages=[
    # #         {
    # #             "role": "system",
    # #             "content": "You are an artificial intelligence assistant, and your role is to "
    # #                 f"present the latest news and updates along with the future news and update for {company_name} in a detailed, organized, and engaging manner."
    # #         },
    # #         {
    # #             "role": "user",
    # #             "content": f"Present the news and events aswell {company_name} over the past {time_period} retatining all the Dates aswell as the future news and events: Latest News and Updates text {previous}, Future News and Updates text {future}?"
    # #         },
    # #     ]
    # # )
    # # response = chats.choices[0].message.content
    # return previous

def extract_diffbot_data(link):
    url = f"https://api.diffbot.com/v3/analyze?url={link}&token=fdbc63a153d0d8da7c0dfb7ccef69945"
    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        article = data.get("objects", [])[0]  # Take the first object

        title = article.get("title", "N/A")
        date = article.get("date", "N/A")
        link = article.get("pageUrl", "N/A")
        content = article.get("text", "N/A")

        return content

        #print("🔹 Title:", title)
        #print("📅 Date:", date)
        #print("🔗 Link:", link)
        #print("\n📄 Content:\n", content[:1000], "...")  # Print first 1000 chars for brevity

    except Exception as e:
        print(f"❌ Failed to extract Diffbot data: {e}")
        print(link)

def bollingerbands(company_name, data_text):
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an AI model designed to assist long-term day traders in analyzing stock market data. "
                    "Your primary task is to interpret stock trading data, especially focusing on Bollinger Bands, "
                    "to identify key market trends. When provided with relevant data you will: "
                    "Analyze the stock's current position relative to its Bollinger Bands (upper, middle, or lower bands) and provide insights."
            },
            {
                "role": "user",
                "content": f"Please analyze the stock data for {company_name}, here is the data {data_text}, What insights can you provide from observing the Bollinger Bands?"
            },
        ]
    )
    response = chat_completion.choices[0].message.content
    return response
def SMA(company_name,data_text):
    
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # System message to define the assistant's behavior
            {
                "role": "system",
                "content":"You are an AI model designed to assist long-term day traders in analyzing stock market data."
                    "Your primary task is to interpret stock trading data, especially focusing on 20, 50, and 200 Simple Moving Averages (SMA),"
                    "to identify key market trends. When provided with relevant data you will:"
                    "\n- Analyze the stock's current position relative to its 20, 50, and 200 SMAs."
                    "\n- Assess if the stock is in an uptrend, downtrend, or nearing a breakout based on the relationships between the SMAs."
                    "\n- Determine if the stock is prone to a reversal by analyzing price movements, SMA crossovers, and the stock's position relative to key SMAs."
                    "\n- Provide a concise, expert-level explanation of your analysis, including how specific SMA characteristics (e.g., crossovers, price deviation from SMAs, trend strength)"
                    "indicate potential market moves."
                    "\n\nEnsure that your explanations are clear and easy to understand, even for users with little to no trading experience, avoiding complex jargon or offering simple definitions where necessary."
                    "Your output should balance depth and simplicity, offering actionable insights for traders while being accessible to non-traders."
                
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": f"Please analyze the stock data for {company_name}, here is the data {data_text}, What insights can you provide from observing SMA?"
                
            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response


def RSI(company_name,data_text):
    
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # System message to define the assistant's behavior
            {
                "role": "system",
                "content":"You are an AI model designed to assist long-term day traders in analyzing stock market data."
                    "Your primary task is to interpret stock trading data, especially focusing on the Relative Strength Index (RSI),"
                    "to identify key market trends. When provided with relevant data you will:"

                    "\n- Analyze the stock's current RSI values to determine if it is overbought, oversold, or in a neutral range."
                    "\n- Assess if the stock is in an uptrend, downtrend, or nearing a potential reversal based on RSI levels and patterns."
                    "\n- Determine if the stock is prone to a reversal by analyzing RSI divergences (bullish or bearish), overbought/oversold conditions, and the stock's momentum."
                    "\n- Provide a concise, expert-level explanation of your analysis, including how specific RSI characteristics (e.g., divergence, trend strength, threshold breaches)"
                    "indicate potential market moves."
                
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": f"Please analyze the stock data for {company_name}, here is the data {data_text}, What insights can you provide from observing RSI?"
                
            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response

def MACD(company_name,data_text):
    
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # System message to define the assistant's behavior
            {
                "role": "system",
                "content":"You are an AI model designed to assist long-term day traders in analyzing stock market data."
                    "Your primary task is to interpret stock trading data, especially focusing on the MACD (Moving Average Convergence Divergence), MACD Signal Line, and MACD Histogram,"
                    "to identify key market trends. When provided with relevant data you will:"
                    "\n- Analyze the stock's MACD line, Signal Line, and Histogram to assess trend strength and potential price direction."
                    "\n- Assess if the stock is in an uptrend, downtrend, or nearing a crossover by analyzing the MACD line relative to the Signal Line."
                    "\n- Determine if the stock is prone to a reversal by examining MACD crossovers, divergences, and changes in the MACD Histogram."
                    "\n- Provide a concise, expert-level explanation of your analysis, including how specific MACD characteristics (e.g., crossover points, divergence, histogram changes)"
                    "indicate potential market moves."
                    "\n\nEnsure that your explanations are clear and easy to understand, even for users with little to no trading experience, avoiding complex jargon or offering simple definitions where necessary."
                    "Your output should balance depth and simplicity, offering actionable insights for traders while being accessible to non-traders."
                
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": f"Please analyze the stock data for {company_name}, here is the data {data_text}, What insights can you provide from observing MACD?"
                
            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response


def OBV(company_name,data_text):
    
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # System message to define the assistant's behavior
            {
                "role": "system",
                "content":"You are an AI model designed to assist long-term day traders in analyzing stock market data."
                    "Your primary task is to interpret stock trading data, especially focusing on On-Balance Volume (OBV),"
                    "to identify key market trends. When provided with relevant data you will:"

                    "\n\n- Read and extract relevant data from PDF and CSV files."
                    "\n- Analyze the stock's OBV to assess the relationship between volume and price movement."
                    "\n- Assess if the stock is in an uptrend, downtrend, or nearing a breakout by evaluating OBV trends and volume momentum."
                    "\n- Determine if the stock is prone to a reversal by analyzing OBV divergences (where OBV moves in the opposite direction of price), which can signal potential trend changes."
                    "\n- Provide a concise, expert-level explanation of your analysis, including how specific OBV characteristics (e.g., divergence, volume spikes, confirmation of price moves)"
                    "indicate potential market moves."

                    "\n\nEnsure that your explanations are clear and easy to understand, even for users with little to no trading experience, avoiding complex jargon or offering simple definitions where necessary."
                    "Your output should balance depth and simplicity, offering actionable insights for traders while being accessible to non-traders."
                
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": f"Please analyze the stock data for {company_name}, here is the data {data_text}, What insights can you provide from observing the OBV?"
                
            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response


def ADX(company_name,data_text):
    
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # System message to define the assistant's behavior
            {
                "role": "system",
                "content":"You are an AI model designed to assist long-term day traders in analyzing stock market data."
                    "Your primary task is to interpret stock trading data, especially focusing on the Average Directional Index (ADX),"
                    "to identify key market trends. When provided with relevant data you will:"

                    "\n- Analyze the stock's ADX values to assess the strength of the current trend, regardless of its direction."
                    "\n- Assess if the stock is in a strong or weak trend based on ADX levels, with particular attention to rising or falling ADX values."
                    "\n- Determine if the stock is prone to a trend reversal by analyzing ADX indicating whether the market is gaining or losing trend strength."
                    "\n- Provide a concise, expert-level explanation of your analysis, including how specific ADX characteristics (e.g., ADX crossovers, trend strength, or weakening trends)"
                    "indicate potential market moves."

                    "\n\nEnsure that your explanations are clear and easy to understand, even for users with little to no trading experience, avoiding complex jargon or offering simple definitions where necessary."
                    "Your output should balance depth and simplicity, offering actionable insights for traders while being accessible to non-traders."
                
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": f"Please analyze the stock data for {company_name}, here is the data {data_text}, What insights can you provide from observing ADX?"
                
            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response

def FUNDAMENTAL_ANALYSIS(file_name, company_name, file):

    temp_file_path = os.path.join(tempfile.gettempdir(), file)

# Write the contents to the temporary file
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(file_name.read())

    system_prompt = """
    You are an AI assistant specializing in financial analysis and long-term investment insights. Your task is to thoroughly analyze a 10-K or 10-Q filing or similar financial document for a public company, using only the content provided below.

    **Instructions:**
    - Do not ask any follow-up questions or request clarifications. Do not prompt the user for any additional input or context. Use only the information in the document provided.
    - **ALWAYS prioritize** the following financial metrics in your analysis: **revenue, income, sales, debt, liabilities, gross margins, and assets**. Ensure these areas are reviewed and reported on first, with maximum detail and supporting data.
    - Analyze the financial information as follows:
        1. Begin your analysis with a focused breakdown of the company's revenue, income, sales, debt, liabilities, gross margins, and assets, referencing all available figures and trends for each. Discuss how these metrics have changed over recent years and what they imply for the company's financial stability, operational performance, and growth prospects.
        2. Review the remaining key financial statements (income statement, balance sheet, cash flow). Assess profitability margins, cash flow trends, and capital expenditures, supplementing your initial breakdown where relevant.
        3. Evaluate Management’s Discussion and Analysis (MD&A) for narrative on financial performance, operational challenges, and future outlook. Identify major shifts, cost measures, or growth initiatives.
        4. Analyze the risk factors section for industry, regulatory, operational, or market risks, distinguishing between ongoing and potentially mitigated risks.
        5. Evaluate the company’s competitive position, market share, and relevant industry trends (including technological/economic changes) that may affect long-term prospects.
        6. Calculate and interpret financial ratios (P/E, debt-to-equity, ROE, free cash flow yield, etc.) and compare to industry peers where possible.
        7. Provide actionable investment recommendations based solely on the findings, including whether the company is under/overvalued, and any suggested entry/exit points. Emphasize potential returns and risks for long-term investors.

    **Output:**  
    Your analysis should be comprehensive, detailed, and data-driven, to the standard of a senior investment analyst. Clearly explain all recommendations and highlight supporting data/metrics. Do not output anything outside this analysis or request further information.
    """


    # 1. Extract full text
    full_text = ""
    with pdfplumber.open(temp_file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"
    

    tables_dict = {}
    with pdfplumber.open(temp_file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for idx, table in enumerate(tables):
                df = pd.DataFrame(table)
                # Only keep tables with real content
                if df.shape[0] > 1 and df.shape[1] > 1:
                    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
                    key = f"page_{page_number}_table_{idx+1}"
                    tables_dict[key] = df

    tables_text_dict = {}
    for key, df in tables_dict.items():
        tables_text_dict[key] = df.to_string(index=False)

    print("bitch")
    
    user_prompt = f"""
    Conduct a comprehensive Fundamental Analysis of the following financial document for Apple Inc. Retrieve all relevant financial data, including ratios and calculations, and provide a robust, broken-down analysis to the level of a very senior investment analyst. Do not ask any questions—just analyze and report.

    ---
    [Full Narrative Text]
    {full_text}

    ---
    [Extracted Tables]
    {tables_text_dict}
    """
        
    output_response = client.chat.completions.create(
    model="gpt-4.1",  # or "gpt-4-turbo", "gpt-4.1", etc.
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
       
    )

    response = output_response.choices[0].message.content

# Write the contents to the temporary file
    # with open(temp_file_path, 'wb') as temp_file:
    #     temp_file.write(file_name.read())
    
    # message_file = client.files.create(
    # file=open(temp_file_path, "rb"), purpose="assistants"
    # )

    

    # file_id = message_file.id

    # file_name_ai = message_file.filename

    # vector_store = client.vector_stores.create(name=f"{company_name} Store")
    # client.vector_stores.files.create(
    #     vector_store_id=vector_store.id,
    #     file_id=file_id
    # )

    # vector_store_id = vector_store.id

    # response = client.responses.create(
    # model="gpt-4.1",
    # instructions=""" 
    # You are an AI assistant specializing in financial analysis and long-term investment insights. Your task is to thoroughly analyze a 10-K filing or similar financial document for a public company, using only the content provided from the vector store. 

    # **Instructions:**
    # - Do not ask any follow-up questions or request clarifications. Do not prompt the user for any additional input or context. Use only the information in the document provided.
    # - Analyze the financial information as follows:
    #     1. Review and break down the key financial statements (income statement, balance sheet, cash flow). Assess revenue growth, profitability margins, debt, cash flow trends, and capital expenditures over recent years for financial stability and growth potential.
    #     2. Evaluate Management’s Discussion and Analysis (MD&A) for narrative on financial performance, operational challenges, and future outlook. Identify major shifts, cost measures, or growth initiatives.
    #     3. Analyze the risk factors section for industry, regulatory, operational, or market risks, distinguishing between ongoing and potentially mitigated risks.
    #     4. Evaluate the company’s competitive position, market share, and relevant industry trends (including technological/economic changes) that may affect long-term prospects.
    #     5. Calculate and interpret financial ratios (P/E, debt-to-equity, ROE, free cash flow yield, etc.) and compare to industry peers where possible.
    #     6. Provide actionable investment recommendations based solely on the findings, including whether the company is under/overvalued, and any suggested entry/exit points. Emphasize potential returns and risks for long-term investors.

    # **Output:**  
    # Your analysis should be comprehensive, detailed, and data-driven, to the standard of a senior investment analyst. Clearly explain all recommendations and highlight supporting data/metrics. Do not output anything outside this analysis or request further information.

    # """,
    # input=f"Conduct Fundamental Analysis on the only document, which is {company_name}'s financial information. Retrieve all relevant financial data, including ratios and calculations, and provide a robust, broken-down analysis to the level of a very senior investment analyst. Do not ask any questions—just analyze and report.",
    # tools=[{
    #     "type": "file_search",
    #     "vector_store_ids": [vector_store_id]
    #     }]
    # )

    # response_id = response.id

    # fetched = client.responses.retrieve(response_id)
    # text = fetched.output_text
    # print("bitch")
    # print(text)

    # chat_completion = client.chat.completions.create(
    #     model="gpt-4.1",  # Ensure that you use a model available in your OpenAI subscription
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": (
    #             "You are an AI model trained to format text for fundamental analysis of financial assets, delivering actionable recommendations. "
    #             "You must output only valid, structured HTML, using semantic tags such as <section>, <h2>, <h3>, <ul>, <ol>, <li>, <p>, and <strong> for clarity and readability. "
    #             "Do not use Markdown or plain text—output only HTML.\n"
    #             "\n"
    #             "Format your analysis with these sections and formatting standards:\n"
    #             "\n"
    #             "<section id='introduction'>\n"
    #             "  <h2>Introduction</h2>\n"
    #             "  <p>Provide a concise overview of the asset, including its industry context and the main purpose of the analysis.</p>\n"
    #             "</section>\n"
    #             "\n"
    #             "<section id='financial-analysis'>\n"
    #             "  <h2>Financial Analysis</h2>\n"
    #             "  <h3>Income Statement</h3>\n"
    #             "  <ul>\n"
    #             "    <li>Summarize trends in <strong>Revenue</strong>, <strong>Cost of Goods Sold</strong>, <strong>Operating Income</strong>, and <strong>Net Income</strong>. Highlight significant changes or growth patterns.</li>\n"
    #             "  </ul>\n"
    #             "  <h3>Balance Sheet</h3>\n"
    #             "  <ul>\n"
    #             "    <li>Summarize <strong>Assets</strong>, <strong>Liabilities</strong>, and <strong>Equity</strong>, focusing on liquidity and leverage metrics.</li>\n"
    #             "  </ul>\n"
    #             "  <h3>Cash Flow Statement</h3>\n"
    #             "  <ul>\n"
    #             "    <li>Highlight <strong>Cash Flow from Operating</strong>, <strong>Investing</strong>, and <strong>Financing Activities</strong>, emphasizing cash generation and any unusual patterns.</li>\n"
    #             "  </ul>\n"
    #             "  <h3>Key Ratios and Metrics</h3>\n"
    #             "  <ul>\n"
    #             "    <li><strong>Profitability Ratios</strong> (e.g., <strong>Gross Margin</strong>, <strong>Return on Assets</strong>)</li>\n"
    #             "    <li><strong>Liquidity Ratios</strong> (e.g., <strong>Current Ratio</strong>, <strong>Quick Ratio</strong>)</li>\n"
    #             "    <li><strong>Leverage Ratios</strong> (e.g., <strong>Debt-to-Equity Ratio</strong>)</li>\n"
    #             "    <li><strong>Valuation Ratios</strong> (e.g., <strong>Price-to-Earnings Ratio (P/E)</strong>, <strong>Price-to-Book Ratio (P/B)</strong>)</li>\n"
    #             "  </ul>\n"
    #             "</section>\n"
    #             "\n"
    #             "<section id='competitive-market-analysis'>\n"
    #             "  <h2>Competitive Positioning and Market Analysis</h2>\n"
    #             "  <ul>\n"
    #             "    <li>Overview of the asset’s competitive position, market share, and primary competitors.</li>\n"
    #             "    <li>Summary of industry trends and a concise <strong>SWOT analysis</strong> (strengths, weaknesses, opportunities, threats).</li>\n"
    #             "  </ul>\n"
    #             "</section>\n"
    #             "\n"
    #             "<section id='management-governance'>\n"
    #             "  <h2>Management and Governance</h2>\n"
    #             "  <ul>\n"
    #             "    <li>Describe the executive team and board structure, noting experience, past performance, and recent changes.</li>\n"
    #             "    <li>Mention recent strategic decisions (e.g., acquisitions, new product lines) that have impacted performance.</li>\n"
    #             "  </ul>\n"
    #             "</section>\n"
    #             "\n"
    #             "<section id='conclusion-outlook'>\n"
    #             "  <h2>Conclusion and Outlook</h2>\n"
    #             "  <ul>\n"
    #             "    <li>Concise summary of strengths and potential risks based on financial and strategic positioning.</li>\n"
    #             "    <li>Outlook considering financial stability, industry conditions, and management’s strategic direction.</li>\n"
    #             "  </ul>\n"
    #             "</section>\n"
    #             "\n"
    #             "<section id='actionable-recommendations'>\n"
    #             "  <h2>Actionable Recommendations</h2>\n"
    #             "  <ol>\n"
    #             "    <li><strong>Investment Recommendation:</strong> Clearly state Buy, Hold, or Sell, and justify with reference to valuation, market, or management actions.</li>\n"
    #             "    <li><strong>Risk Management Suggestions:</strong> Outline risk mitigation strategies (e.g., diversification, stop-loss orders).</li>\n"
    #             "    <li><strong>Strategic Suggestions for Management:</strong> If relevant, suggest actions for the company (e.g., explore new markets, reduce debt, optimize costs).</li>\n"
    #             "    <li><strong>Performance Monitoring Tips:</strong> Recommend specific metrics or events (e.g., quarterly earnings, regulatory updates) for ongoing evaluation.</li>\n"
    #             "  </ol>\n"
    #             "</section>\n"


    #             "Style Requirements"
    #             "Maintain a professional, objective tone focused on analysis without personal opinions."
    #             "Avoid excessive jargon; use clear, direct explanations where needed."
    #             "Keep sentences and paragraphs clear and direct for logical flow and easy understanding."
    #             "Include all sections and headings as listed, even if a section is brief. Output only valid HTML."
                    
    #             ),
    #         },
    #         {
    #             "role": "user",
    #             "content": (
    #                 f"fromat this text {text}"   
    #             ),
    #         },
    #     ]
    # )

    # # Extract and return the AI-generated response
    # response = chat_completion.choices[0].message.content

    # deleted_vector_store_file = client.vector_stores.delete(
    #     vector_store_id=vector_store_id
    # )
    
    print("File successfully deleted from vector store.")
    return response
    

def FUNDAMENTAL_ANALYSIS2(file_name, company_name, file):

    today = date.today()
    formatted = today.strftime('%Y-%m-%d')

    system_prompt = f"""
    You are a world-class investment analyst.
    Using the available company data and your financial expertise, produce a comprehensive fundamental analysis report in valid HTML using the template below. 
    Fill in all [PLACEHOLDER] tags with clear, professional analysis—even if some sections are brief. 
    Be explicit in the recommendation (Buy, Hold, or Sell) and use supporting financial evidence throughout.

    Style Requirements:
    - Use clear, objective, and professional language suitable for investors.
    - Avoid unnecessary jargon—explain technical terms where needed.
    - Organize your report with clear sections and subheadings.
    - Include all sections, even if some are brief due to limited data.

    **Key Financial Metrics Requirement:**
    - Always include a "Key Financial Metrics" section at the top of Financial Analysis.
    - Display the following eight metrics as individual metric-cards in a flex container (with "N/A" if missing): Revenues, Sales, Income (Net Income or Earnings), Gross Margins, Liabilities, Cash Flow (Operating), Debt, Assets.
    - Additional metrics may be displayed as extra cards if data is available.
    - Each card uses the <div class="metric-card"><div class="metric-title">[Metric]</div><div class="metric-value">[Value]</div></div> structure within <div class="metrics">.

    HTML Template:
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fundamental Investment Analysis</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0px;
                background-color: transparent;
            }}
            .container {{
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                padding: 30px;
                margin-bottom: 30px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            h2 {{
                color: #2c3e50;
                border-left: 5px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
                background-color: #f8f9fa;
                padding: 10px 15px;
                border-radius: 0 5px 5px 0;
            }}
            h3 {{
                color: #2c3e50;
                margin-top: 20px;
                border-bottom: 1px dashed #ddd;
                padding-bottom: 5px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .metrics {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background-color: #f0f7ff;
                border-radius: 5px;
                padding: 15px;
                flex: 1;
                min-width: 200px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .metric-title {{
                font-weight: bold;
                color: #2980b9;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 1.2em;
                font-weight: bold;
            }}
            .summary-box {{
                background-color: #e8f4fd;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }}
            .recommendation {{
                font-weight: bold;
                font-size: 1.1em;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                text-align: center;
            }}
            .buy {{
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            .hold {{
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
            }}
            .sell {{
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }}
            .footnote {{
                font-size: 0.9em;
                font-style: italic;
                color: #6c757d;
                margin-top: 30px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
            }}
            .highlight {{
                background-color: #ffeaa7;
                padding: 2px 4px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fundamental Investment Analysis: [COMPANY_PLACEHOLDER]</h1>
            
            <section class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p>[SUMMARY_PLACEHOLDER]</p>
                </div>
                <div class="recommendation [RECOMMENDATION_CLASS_PLACEHOLDER]">
                    RECOMMENDATION: [RECOMMENDATION_PLACEHOLDER]
                    <br>
                    <span style="font-size:0.95em; font-weight:normal;">
                    <strong>Note:</strong> This recommendation is based on a thorough review of the company's fundamental financial and strategic data.
                    </span>
                </div>
            </section>

            <section class="section">
                <h2>Financial Analysis</h2>
                <h3>Key Financial Metrics</h3>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">Revenues</div>
                        <div class="metric-value">[REVENUES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Sales</div>
                        <div class="metric-value">[SALES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Income</div>
                        <div class="metric-value">[INCOME_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Gross Margins</div>
                        <div class="metric-value">[GROSS_MARGINS_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Liabilities</div>
                        <div class="metric-value">[LIABILITIES_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Cash Flow</div>
                        <div class="metric-value">[CASH_FLOW_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Debt</div>
                        <div class="metric-value">[DEBT_PLACEHOLDER]</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Assets</div>
                        <div class="metric-value">[ASSETS_PLACEHOLDER]</div>
                    </div>
                    [OPTIONAL_ADDITIONAL_METRIC_CARDS_PLACEHOLDER]
                </div>
                <h3>Income Statement</h3>
                <ul>
                    <li>[INCOME_STATEMENT_PLACEHOLDER]</li>
                </ul>
                <h3>Balance Sheet</h3>
                <ul>
                    <li>[BALANCE_SHEET_PLACEHOLDER]</li>
                </ul>
                <h3>Cash Flow Statement</h3>
                <ul>
                    <li>[CASH_FLOW_PLACEHOLDER]</li>
                </ul>
                <h3>Key Ratios and Metrics</h3>
                <ul>
                    <li>[RATIOS_PLACEHOLDER]</li>
                </ul>
            </section>

            <section class="section">
                <h2>Competitive Positioning and Market Analysis</h2>
                <ul>
                    <li>[COMPETITIVE_ANALYSIS_PLACEHOLDER]</li>
                    <li>[SWOT_PLACEHOLDER]</li>
                </ul>
            </section>

            <section class="section">
                <h2>Management and Governance</h2>
                <ul>
                    <li>[MANAGEMENT_PLACEHOLDER]</li>
                    <li>[STRATEGIC_DECISIONS_PLACEHOLDER]</li>
                </ul>
            </section>

            <section class="section">
                <h2>Conclusion and Outlook</h2>
                <ul>
                    <li>[CONCLUSION_PLACEHOLDER]</li>
                    <li>[OUTLOOK_PLACEHOLDER]</li>
                </ul>
            </section>

            <section class="section">
                <h2>Actionable Recommendations</h2>
                <ol>
                    <li><strong>Investment Recommendation:</strong> [DETAILED_RECOMMENDATION_PLACEHOLDER]</li>
                    <li><strong>Risk Management Suggestions:</strong> [RISK_MANAGEMENT_PLACEHOLDER]</li>
                    <li><strong>Strategic Suggestions for Management:</strong> [MANAGEMENT_SUGGESTIONS_PLACEHOLDER]</li>
                    <li><strong>Performance Monitoring Tips:</strong> [PERFORMANCE_MONITORING_PLACEHOLDER]</li>
                </ol>
            </section>

            <div class="footnote">
                <p>This fundamental analysis was generated on {formatted}. All investment decisions should be made in conjunction with personal financial advice and risk tolerance assessments.</p>
            </div>
        </div>
    </body>
    </html>
    """


    temp_file_path = os.path.join(tempfile.gettempdir(), file)

# Write the contents to the temporary file
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(file_name.read())
    
    message_file = client.files.create(
    file=open(temp_file_path, "rb"), purpose="assistants"
    )

    file_id = message_file.id

    file_name_ai = message_file.filename

    vector_store = client.vector_stores.create(name=f"{company_name} Store")
    client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file_id
    )

    vector_store_id = vector_store.id

    response = client.responses.create(
    model="gpt-4.1",
    instructions=""" You are an AI assistant specializing in financial analysis and long-term investment insights, particularly skilled in reading, interpreting, and analyzing 10-K filings of public companies. Your primary objective is to conduct thorough fundamental analysis based on the financial data, management discussion, and other disclosures within these filings. By doing so, you identify strengths, weaknesses, risks, and opportunities of the company’s financial health, operational performance, and strategic positioning. Your approach includes the following steps: Reviewing Key Financial Statements: Analyze the income statement, balance sheet, and cash flow statement. Assess revenue growth, profitability margins, debt levels, cash flow trends, and capital expenditures over recent years to gauge the company’s financial stability and growth potential. Assessing Management’s Discussion and Analysis (MD&A): Evaluate the management’s narrative on financial performance, operational challenges, and future outlook. Identify any significant shifts in strategy, cost-cutting measures, or growth initiatives that might impact long-term viability. Analyzing Risk Factors: Carefully review the section on risk factors to understand industry-specific, regulatory, operational, and market risks that may affect the company’s future performance. Assess which risks are ongoing versus those that may be temporary or mitigated through strategic actions. Evaluating Competitive Position and Industry Trends: Examine the company’s competitive positioning, market share, and any significant developments in its industry. Look for insights on emerging trends, technological changes, or economic factors that may influence long-term prospects. Reviewing Financial Ratios and Key Metrics: Calculate and interpret relevant financial ratios—such as the price-to-earnings (P/E) ratio, debt-to-equity ratio, return on equity (ROE), and free cash flow yield. These metrics help gauge valuation, efficiency, leverage, and profitability relative to industry peers. Providing Actionable Investment Recommendations: Based on your findings, formulate long-term investment recommendations. Consider if the company appears undervalued or overvalued, and outline potential entry or exit points for investment. Your recommendations should emphasize a balanced view of potential returns and risks for long-term investors, aligning with value, growth, or income-based investment objectives. Your goal is to offer a comprehensive, data-driven perspective that enables users to make informed decisions about including the company in a long-term investment portfolio. Ensure all recommendations are clearly explained, with relevant data and metrics highlighted to support your conclusions.""",
    input= f"Conduct Fundamental Analysis of {company_name}'s finacial statements, using the document in the vector store: {file_name_ai} to retrieve all the information",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vector_store_id]
    }]
    )
    


    # data = {"File_id": file_id, "Company Name": company_name, "File_name": file}

    # webhook_url = "https://hook.eu2.make.com/d68cwl3ujkpqmgrnbpgy9mx3d06vs198"
    # if webhook_url:
    #     response = requests.post(webhook_url,data)
    # else: 
    #     print("Error")

    # time.sleep(65)

    # credentials_dict = {
    #     "type": type_sa,
    #     "project_id": project_id,
    #     "private_key_id": private_key_id,
    #     "private_key": private_key,
    #     "client_email": client_email,
    #     "client_id": client_id,
    #     "auth_uri": auth_uri,
    #     "token_uri": token_uri,
    #     "auth_provider_x509_cert_url": auth_provider_x509_cert_url,
    #     "client_x509_cert_url": client_x509_cert_url,
    #     "universe_domain": universe_domain
    # }
    # credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, ["https://www.googleapis.com/auth/spreadsheets"])
    # gc = gspread.authorize(credentials)
    # sh = gc.open_by_url(google_sheet_url)
    # anaylsis = sh.sheet1.get('C2')

    chat_completion = client.chat.completions.create(
        model="gpt-4.1",  # Ensure that you use a model available in your OpenAI subscription
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"fromat this text {response}"   
                ),
            },
        ]
    )

    # Extract and return the AI-generated response
    response = chat_completion.choices[0].message.content

    deleted_vector_store_file = client.vector_stores.delete(
        vector_store_id=vector_store_id
    )
    
    print("File successfully deleted from vector store.")
    return response 
    
   
def SUMMARY(company_name, BD, SMA, RSI, MACD, OBV, ADX, weighted_score, weight_choice):
    chat_completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI model designed to assist long-term day traders in analyzing stock market data using a weighted approach."
                    " Your primary role is to synthesize data from multiple technical indicators—both lagging (MACD, SMA) and leading (ADX, RSI, OBV, Bollinger Bands)—"
                    " and deliver a single, clear, actionable conclusion about the stock's long-term trend."
                    "\n\n"
                    "When analyzing the indicators provided, you must:"
                    "\n- Extract and interpret the key signals from each indicator summary."
                    "\n- Weigh the importance of each indicator according to the selected weighting style (for example, 'Long Term' prioritizes slow-moving trends, 'Short Term' prioritizes fast-moving ones, and 'Default' is balanced)."
                    "\n- Calculate or use the provided *weighted score* to support your conclusion (the weighted score is a summary value reflecting the overall strength and direction of the combined indicators, weighted according to the chosen style)."
                    "\n- Make your final advice based on this weighted approach, ensuring that the recommendation aligns with the weighted score and selected style."
                    "\n\n"
                    "Guidelines for your output:"
                    "\n- Limit your response to ONE concise paragraph."
                    "\n- Clearly state the overall trend (e.g., strengthening, weakening, reversal) and the recommended action."
                    "\n- Bold your suggested position (e.g., **Strong Buy**, **Hold**, **Sell**), and mention whether the weighted score and weighting style support this choice."
                    "\n- Do NOT output individual indicator details or jargon—focus on the summary and recommendation."
                    "\n- Ensure your response is simple, actionable, and understandable for a non-trader."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Summarize the stock data for {company_name}. "
                    f"Bollinger Bands: {BD}, "
                    f"Simple Moving Averages: {SMA}, "
                    f"Relative Strength Index: {RSI}, "
                    f"MACD: {MACD}, "
                    f"OBV: {OBV}, "
                    f"ADX: {ADX}."
                    f" Use the weighting style: {weight_choice}. The combined weighted score is: {weighted_score}"
                )
            },
        ]
    )

    response = chat_completion.choices[0].message.content
    return response

    
    

def SUMMARY2(gathered_data):
    today = date.today()
    formatted = today.strftime('%Y-%m-%d')

    weighted_score = gathered_data.get("weighted_score")
    print("weighted_score =", float(weighted_score))

    # --- New momentum logic ---
    if weighted_score > 0.05:
        momentum = "Upward"
        momentum_class = "buy"  # reuse the style, or create new CSS
    elif weighted_score < -0.05:
        momentum = "Downward"
        momentum_class = "sell"
    else:
        momentum = "Neutral"
        momentum_class = "hold"


    system_prompt = f"""As an AI assistant for traders and investors, your task is to produce a structured technical market analysis in valid HTML format.

    Parse the provided JSON, replacing these placeholders:
    - Ticker, Company, Timeframe, Technical Analysis summary, and Technical Indicator summaries.
    - Instead of an investment recommendation, show only the current **momentum** as Upward, Downward, or Neutral, based solely on the weighted technical score.

    **Special Instructions:**  
    - The analysis and all technical indicator values must be grouped by weeks. Clearly highlight which week(s) or date range are used for the analysis.
    - In the Executive Summary section, include a visible note stating:
        1. The analysis is grouped by weeks using the most recent available data.
        2. Which weeks or date range the analysis covers (e.g. "Week of 2025-06-03 to 2025-07-21").
        3. The technical momentum and indicators reflect weekly trends to filter out short-term noise.
    - In the "Technical Analysis" section, provide only a concise and detailed summary of the technical analysis. Do not include charts, extended commentary, or breakdowns—just main insights in a few sentences.

    **Indicator Formatting Instructions:**  
    For **each Technical Indicator** (SMA, RSI, MACD, OBV, ADX, Bollinger Bands), use this structure for the text:
    1. State the date(s) or week(s) the analysis covers.
    2. Clearly present the most recent indicator values (e.g., “As of July 27, 2025, the 20-day SMA is $208.67, the 50-day SMA is $205.01, and the closing price is $214.40”).
    3. Use bullet points or numbered lists to highlight:
        - Position of the price relative to indicator(s)
        - Trend/crossover events
        - Any significant market moves, shifts, or signals
        - A concise summary/conclusion (bold or strong tag) at the end of each indicator’s section

    **Example (for SMA):**
    <ol>
    <li><strong>Current Position:</strong> As of July 27, 2025, the closing price ($214.40) is above both the 20-day SMA ($208.67) and 50-day SMA ($205.01).</li>
    <li><strong>Trend Analysis:</strong> Both SMAs are rising, indicating a strong uptrend.</li>
    <li><strong>SMA Crossover:</strong> The 20-day SMA is above the 50-day, signaling continued bullish momentum.</li>
    <li><strong>Summary:</strong> <b>AAPL is in a robust uptrend; further gains are possible unless resistance emerges.</b></li>
    </ol>

    Repeat this clear, analytic, bullet/numbered style for each indicator (SMA, RSI, MACD, OBV, ADX, BD/Bollinger Bands), always referencing the latest weekly values and their implications for trend, strength, or reversals.

    Your output must use the following HTML structure and tags, omitting any references to BUY/HOLD/SELL:

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Technical Momentum Analysis</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0px;
                background-color: transparent;
            }}
            .container {{
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                padding: 30px;
                margin-bottom: 30px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            h2 {{
                color: #2c3e50;
                border-left: 5px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
                background-color: #f8f9fa;
                padding: 10px 15px;
                border-radius: 0 5px 5px 0;
            }}
            h3 {{
                color: #2c3e50;
                margin-top: 20px;
                border-bottom: 1px dashed #ddd;
                padding-bottom: 5px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .momentum {{
                font-weight: bold;
                font-size: 1.1em;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                text-align: center;
            }}
            .buy {{
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            .hold {{
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
            }}
            .sell {{
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }}
            .summary-box {{
                background-color: #e8f4fd;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }}
            .timeframe {{
                font-weight: bold;
                color: #2c3e50;
                background-color: #e8f4fd;
                padding: 5px 10px;
                border-radius: 3px;
                display: inline-block;
                margin-bottom: 15px;
            }}
            .indicator {{
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 5px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
            }}
            .indicator h4 {{
                margin-top: 0;
                color: #2980b9;
            }}
            .footnote {{
                font-size: 0.9em;
                font-style: italic;
                color: #6c757d;
                margin-top: 30px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
            }}
            .highlight {{
                background-color: #ffeaa7;
                padding: 8px 12px;
                border-radius: 5px;
                display: block;
                margin: 15px 0 0 0;
                font-size: 1em;
            }}
            /* Responsive Design */
            @media (max-width: 768px) {{
                .container {{
                    padding: 10px;
                }}
                h1, h2 {{
                    font-size: 1.3em;
                    padding-left: 8px;
                    padding-right: 8px;
                }}
                .section {{
                    padding: 10px;
                }}
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 16px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f8f9fa;
                color: #2c3e50;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Technical Momentum Analysis: [TICKER_PLACEHOLDER] - [COMPANY_PLACEHOLDER]</h1>
            <div class="timeframe">Analysis Timeframe: [TIMEFRAME_PLACEHOLDER]</div>

            <section class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p>[SUMMARY_PLACEHOLDER]</p>
                    <div class="highlight">
                        <em>
                            Note: This analysis is grouped by weeks, using the most recent available weekly data.<br>
                            <strong>Weeks Analyzed:</strong> [WEEK_RANGE_PLACEHOLDER]<br>
                            The technical momentum and indicators reflect weekly trends, offering a clearer view of medium-term market movements and filtering out short-term fluctuations.
                        </em>
                    </div>
                </div>
                <div class="momentum {momentum_class}">
                    <strong>Momentum:</strong> {momentum}
                </div>
            </section>

            <section class="section">
                <h2>Technical Analysis</h2>
                <div id="technical-analysis">
                    [TECHNICAL_ANALYSIS_SUMMARY_PLACEHOLDER]
                </div>
                <h3>Technical Indicators</h3>
                <div class="indicator">
                    <h4>SMA (Simple Moving Average)</h4>
                    <ol>
                        [SMA_ANALYSIS_PLACEHOLDER]
                    </ol>
                </div>
                <div class="indicator">
                    <h4>RSI (Relative Strength Index)</h4>
                    <ol>
                        [RSI_ANALYSIS_PLACEHOLDER]
                    </ol>
                </div>
                <div class="indicator">
                    <h4>MACD (Moving Average Convergence Divergence)</h4>
                    <ol>
                        [MACD_ANALYSIS_PLACEHOLDER]
                    </ol>
                </div>
                <div class="indicator">
                    <h4>OBV (On-Balance Volume)</h4>
                    <ol>
                        [OBV_ANALYSIS_PLACEHOLDER]
                    </ol>
                </div>
                <div class="indicator">
                    <h4>ADX (Average Directional Index)</h4>
                    <ol>
                        [ADX_ANALYSIS_PLACEHOLDER]
                    </ol>
                </div>
                <div class="indicator">
                    <h4>BD (Bollinger Bands)</h4>
                    <ol>
                        [BD_ANALYSIS_PLACEHOLDER]
                    </ol>
                </div>
            </section>

            <div class="footnote">
                <p>This technical momentum analysis was generated on {formatted}, using available market data as of this date. For investing, always consider multiple sources and your personal risk tolerance.</p>
            </div>
        </div>
    </body>
    </html>
    """


    user_message = f"The data to analyse: {json.dumps(gathered_data)}"
    chat_completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                # System message to define the assistant's behavior
            {
                    "role": "system",
                    "content":  system_prompt
                                    
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": user_message
                    
            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response

def format_news(txt_summary):
    chat_completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an AI model designed to convert recent event data into a clean, professionally formatted HTML summary.\n\n"
                "Your task is to transform each event into a structured HTML entry that is easy to read and suitable for use in reports, dashboards, or investor updates.\n\n"
                "The user will provide a list of events in structured text or JSON format. Each event contains:\n"
                "- Date\n"
                "- Title\n"
                "- Overview\n"
                "- Impact\n"
                "- Source\n\n"
                "You must format this information using the HTML structure below:\n\n"
                "<div style=\"font-size:12pt; margin-bottom:20px; font-family:Arial, sans-serif;\">\n"
                "  <strong>Date: [DATE] — Event: [TITLE]</strong><br/>\n"
                "  <p><strong>Overview:</strong> [OVERVIEW]</p>\n"
                "  <p><strong>Impact:</strong> [IMPACT]</p>\n"
                "  <p><strong>Source:</strong> [SOURCE]</p>\n"
                "</div>\n\n"
                "Formatting Guidelines:\n"
                "- Use 12pt font consistently.\n"
                "- Separate sections with <p> tags.\n"
                "- Replace all [PLACEHOLDER] entries with actual event content.\n"
                "- Output valid, clean HTML only — no extra narrative, no markdown.\n\n"
                "Return one complete <div> block per event."
            )
        },
        {
            "role": "user",
            "content": f"text to format {txt_summary}"
        }
    ]
)


# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response



def calculate_technical_indicators(data, ticker, weight_choice=None):
    """
    Calculate various technical indicators, prepare them for AI analysis,
    and compute a weighted technical score.

    Args:
        data (pd.DataFrame): The input financial data.
        ticker (str): The stock ticker.
        weights (dict): Optional dict of weights for each indicator.

    Returns:
        Tuple: (results dict, recent_data, availability, scores, weighted_score)
    """
    short_term_weights = {
    "sma": 0.1,
    "rsi": 0.3,
    "macd": 0.3,
    "obv": 0.1,
    "adx": 0.1,
    "bbands": 0.1
    }
    long_term_weights = {
        "sma": 0.4,
        "rsi": 0.1,
        "macd": 0.15,
        "obv": 0.15,
        "adx": 0.2,
        "bbands": 0.0
    }

    weights = {
            "sma": 0.2,
            "rsi": 0.2,
            "macd": 0.2,
            "obv": 0.2,
            "adx": 0.2,
            "bbands": 0.0  # Set to 0 if not using
        }

# Choose the right weights
    if weight_choice == "Short Term":
        weights = short_term_weights
    if weight_choice == "Long Term":
        weights = long_term_weights
    if weight_choice == "Default":
        weights = weights

    # --- Default Weights if not provided ---

    # Initialize availability flags
    sma_available = False
    rsi_available = False
    macd_available = False
    obv_available = False
    adx_available = False
    bbands_available = False

    # --- Calculate SMA ---
    if 'Close' in data.columns:
        data['SMA_20'] = ta.sma(data['Close'], length=20)
        data['SMA_50'] = ta.sma(data['Close'], length=50)
        data['SMA_200'] = ta.sma(data['Close'], length=200)
        sma_available = data[['SMA_20', 'SMA_50', 'SMA_200']].notna().any().any()

    # --- Calculate RSI ---
    if 'Close' in data.columns:
        data['RSI'] = ta.rsi(data['Close'], length=14)
        rsi_available = 'RSI' in data.columns and data['RSI'].notna().any()

    # --- Calculate MACD ---
    macd = ta.macd(data['Close'])
    if macd is not None and all(col in macd.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']):
        data['MACD'] = macd['MACD_12_26_9']
        data['MACD_signal'] = macd['MACDs_12_26_9']
        data['MACD_hist'] = macd['MACDh_12_26_9']
        macd_available = True

    # --- Calculate OBV ---
    if 'Close' in data.columns and 'Volume' in data.columns:
        data['OBV'] = ta.obv(data['Close'], data['Volume'])
        obv_available = 'OBV' in data.columns and data['OBV'].notna().any()

    # --- Calculate ADX ---
    adx = ta.adx(data['High'], data['Low'], data['Close'])
    if adx is not None and 'ADX_14' in adx.columns:
        data['ADX'] = adx['ADX_14']
        adx_available = True

    # --- Calculate Bollinger Bands ---
    bbands = ta.bbands(data['Close'], length=20, std=2)
    if bbands is not None and all(col in bbands.columns for col in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']):
        data['upper_band'] = bbands['BBU_20_2.0']
        data['middle_band'] = bbands['BBM_20_2.0']
        data['lower_band'] = bbands['BBL_20_2.0']
        bbands_available = True

    # --- Resample data weekly ---
    data = data.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'SMA_20': 'last',
        'SMA_50': 'last',
        'SMA_200': 'last',
        'RSI': 'last',
        'MACD': 'last',
        'MACD_signal': 'last',
        'MACD_hist': 'last',
        'OBV': 'last',
        'ADX': 'last',
        'upper_band': 'last',
        'middle_band': 'last',
        'lower_band': 'last'
    })

    # --- Prepare data for analysis ---
    recent_data = data

    # --- Run your original analysis functions (these return text) ---
    results = {
        "bd_result": bollingerbands(ticker, recent_data[["Open", "High", "Low", "Close", "Volume", "upper_band", "middle_band", "lower_band"]].to_markdown()),
        "sma_result": SMA(ticker, recent_data[["Open", "High", "Low", "Close", "SMA_20", "SMA_50", "SMA_200"]].to_markdown()) if sma_available else "SMA analysis not available.",
        "rsi_result": RSI(ticker, recent_data[["Open", "High", "Low", "Close", "RSI"]].to_markdown()) if rsi_available else "RSI analysis not available.",
        "macd_result": MACD(ticker, recent_data[["Open", "High", "Low", "Close", "MACD", "MACD_signal", "MACD_hist"]].to_markdown()) if macd_available else "MACD analysis not available.",
        "obv_result": OBV(ticker, recent_data[["Open", "High", "Low", "Close", "Volume", "OBV"]].to_markdown()) if obv_available else "OBV analysis not available.",
        "adx_result": ADX(ticker, recent_data[["Open", "High", "Low", "Close", "ADX"]].to_markdown()) if adx_available else "ADX analysis not available."
    }

    availability = {
        "sma_available": sma_available,
        "rsi_available": rsi_available,
        "macd_available": macd_available,
        "obv_available": obv_available,
        "adx_available": adx_available,
        "bbands_available": bbands_available
    }

    indicator_scores = {k: [] for k in weights}

    for _, week in data.iterrows():
    # SMA
        if availability['sma_available'] and pd.notna(week['Close']) and pd.notna(week['SMA_20']):
            score = 1 if week['Close'] > week['SMA_20'] else -1
            indicator_scores['sma'].append(score)
        # RSI
        if availability['rsi_available'] and pd.notna(week['RSI']):
            if week['RSI'] > 55:
                score = 1
            elif week['RSI'] < 45:
                score = -1
            else:
                score = 0
            indicator_scores['rsi'].append(score)
        # MACD
        if availability['macd_available'] and pd.notna(week['MACD']) and pd.notna(week['MACD_signal']):
            score = 1 if week['MACD'] > week['MACD_signal'] else -1
            indicator_scores['macd'].append(score)
        # OBV
        if availability['obv_available'] and pd.notna(week['OBV']):
            if week['OBV'] > 0:
                score = 1
            elif week['OBV'] < 0:
                score = -1
            else:
                score = 0
            indicator_scores['obv'].append(score)
        # ADX
        if availability['adx_available'] and pd.notna(week['ADX']):
            score = 1 if week['ADX'] > 20 else -1
            indicator_scores['adx'].append(score)
        # BBands
        if availability['bbands_available'] and pd.notna(week['Close']) and pd.notna(week['middle_band']):
            score = 1 if week['Close'] > week['middle_band'] else -1
            indicator_scores['bbands'].append(score)

    # Aggregate: take the mean (average) score for each indicator
    import numpy as np
    final_scores = {}
    for k in indicator_scores:
        if indicator_scores[k]:  # If there are scores for that indicator
            final_scores[k] = np.mean(indicator_scores[k])
        else:
            final_scores[k] = 0

    # Now calculate weighted score as before, but using averages over weeks
    total_weight = sum(weights[k] for k in final_scores if availability.get(f"{k}_available", False))
    weighted_score = (
        sum(final_scores[k] * weights[k] for k in final_scores if availability.get(f"{k}_available", False)) / total_weight
        if total_weight > 0 else 0
    )

    print("Final Indicator Averages:", final_scores)
    print("Weighted Score:", weighted_score)
    

    # --- RETURN everything ---
    return results, recent_data, availability, weighted_score



def update_progress(progress_bar, stage, progress, message):
    progress_bar.progress(progress)
    st.text(message)
    st.empty()

def plot_sma(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='SMA 200', line=dict(color='green', dash='dash')))
    return fig

# Function to plot Bollinger Bands
def plot_bbands(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['upper_band'], mode='lines', name='Upper Band', line=dict(color='cyan', dash='dot')))
    fig.add_trace(go.Scatter(x=data.index, y=data['middle_band'], mode='lines', name='Middle Band', line=dict(color='magenta', dash='dot')))
    fig.add_trace(go.Scatter(x=data.index, y=data['lower_band'], mode='lines', name='Lower Band', line=dict(color='cyan', dash='dot')))
    return fig

# Function to plot RSI
def plot_rsi(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
    fig.add_hline(y=70, line=dict(color='red', dash='dash'))
    fig.add_hline(y=30, line=dict(color='green', dash='dash'))
    return fig

# Function to plot MACD
def plot_macd(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], mode='lines', name='MACD Signal', line=dict(color='red')))
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_hist'], name='MACD Histogram', marker_color='gray', opacity=0.5))
    return fig

# Function to plot OBV
def plot_obv(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], mode='lines', name='OBV', line=dict(color='brown')))
    return fig

# Function to plot ADX
def plot_adx(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], mode='lines', name='ADX', line=dict(color='orange')))
    return fig


def main():
    stock_page()

if __name__ == "__main__":
    main()

