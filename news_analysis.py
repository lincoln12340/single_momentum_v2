# news_sentiment_gather.py
# ------------------------------------------------------------
# Usage:
#   from news_sentiment_gather import get_news_sentiment_gathered_data
#   data = get_news_sentiment_gathered_data(
#       ticker="MRNA",
#       period="3m",  # "3m" | "6m" | "1y"
#       alpha_vantage_api_key="YOUR_AV_KEY",
#       openai_api_key="YOUR_OPENAI_KEY",
#       topics=None,             # e.g. ["life_sciences","earnings"]
#       min_relevance=0.20,
#       use_recency_decay=True
#   )
#   print(data)
#
# Returns:
#   gathered_data: dict with Context, Sentiment, Articles (incl. LLM fields for top/bottom weighted),
#   and Synthesis for the top-3 positive/negative sets.
# ------------------------------------------------------------

import math
import time
import json
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

# =============== Time & scoring utilities ===============

def _parse_time(ts: str) -> Optional[datetime]:
    """Alpha Vantage timestamp format: YYYYMMDDTHHMM (UTC)."""
    try:
        return datetime.strptime(ts, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def _recency_weight(time_published: Optional[str], half_life_days: float = 14.0) -> float:
    """Exponential decay by age (days)."""
    dt = _parse_time(time_published) if time_published else None
    if not dt:
        return 1.0
    age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    lam = math.log(2) / half_life_days
    return math.exp(-lam * age_days)

def _label_from_score(score: float) -> str:
    if score <= -0.35: return "Bearish"
    if score <= -0.15: return "Somewhat Bearish"
    if score <  0.15:  return "Neutral"
    if score <  0.35:  return "Somewhat Bullish"
    return "Bullish"

def _time_from_for_period(period: str) -> str:
    """Return time_from in YYYYMMDDTHHMM for '3m'|'6m'|'1y' (UTC, midnight)."""
    period = period.lower().strip()
    days_map = {"3m": 90, "6m": 182, "1y": 365}
    days = days_map.get(period)
    if days is None:
        raise ValueError("period must be one of: '3m', '6m', '1y'")
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt.strftime("%Y%m%dT%H%M")

# =============== Alpha Vantage: URL build, fetch, scoring ===============

def _build_news_url(
    apikey: str,
    ticker: Optional[str] = None,
    topics: Optional[List[str]] = None,
    period: str = "3m",
    limit: int = 1000,
    sort: str = "LATEST"
) -> str:
    base = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": apikey,
        "time_from": _time_from_for_period(period),
        "limit": str(limit),
        "sort": sort
    }
    if ticker:
        params["tickers"] = ticker.upper()
    if topics:
        params["topics"] = ",".join(topics)

    from urllib.parse import urlencode
    return f"{base}?{urlencode(params)}"

def _fetch_news(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def _summarize_ticker_sentiment(
    payload: Dict[str, Any],
    ticker: str,
    min_relevance: float = 0.20,
    use_recency_decay: bool = True,
    half_life_days: float = 14.0
) -> Dict[str, Any]:
    """
    Compute relevance×(optional)recency weighted average sentiment for a ticker.
    Returns dict with weighted_avg, label, n_articles, total_weight, and articles list.
    """
    if not payload or "feed" not in payload:
        return {
            "ticker": ticker.upper(),
            "weighted_avg": 0.0,
            "label": "Neutral",
            "n_articles": 0,
            "total_weight": 0.0,
            "articles": []
        }

    tkr = ticker.upper()
    rows: List[Dict[str, Any]] = []

    for item in payload.get("feed", []):
        for ts in (item.get("ticker_sentiment") or []):
            if ts.get("ticker", "").upper() != tkr:
                continue
            try:
                rel = float(ts.get("relevance_score", 0))
                s = float(ts.get("ticker_sentiment_score", 0))
            except (TypeError, ValueError):
                rel, s = 0.0, 0.0

            if rel <= min_relevance:
                continue

            s = max(-1.0, min(1.0, s))   # clip to [-1, 1]
            tp = item.get("time_published")
            w_time = _recency_weight(tp, half_life_days) if use_recency_decay else 1.0
            w = rel * w_time

            rows.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "time_published": tp,
                "source": item.get("source"),
                "relevance_score": rel,
                "ticker_sentiment_score": s,
                "weight": w
            })
            break  # one matching ticker per article

    total_weight = sum(r["weight"] for r in rows)
    weighted_avg = (
        sum(r["ticker_sentiment_score"] * r["weight"] for r in rows) / total_weight
        if total_weight > 0 else 0.0
    )
    label = _label_from_score(weighted_avg)

    # Sort newest first, then weight
    rows.sort(key=lambda r: (r["time_published"] or "", r["weight"]), reverse=True)
    return {
        "ticker": tkr,
        "weighted_avg": weighted_avg,
        "label": label,
        "n_articles": len(rows),
        "total_weight": total_weight,
        "articles": rows
    }

def _get_sentiment_for_ticker(
    apikey: str,
    ticker: str,
    period: str = "3m",
    topics: Optional[List[str]] = None,
    min_relevance: float = 0.20,
    use_recency_decay: bool = True
) -> Dict[str, Any]:
    url = _build_news_url(
        apikey=apikey,
        ticker=ticker,
        topics=topics,
        period=period,
        limit=1000,
        sort="LATEST"
    )
    data = _fetch_news(url)
    return _summarize_ticker_sentiment(
        data,
        ticker=ticker,
        min_relevance=min_relevance,
        use_recency_decay=use_recency_decay
    )

# =============== OpenAI LLM helpers (Responses API) ===============

from openai import OpenAI

def _extract_url_text_with_llm(client: OpenAI, url: str, model: str = "gpt-4.1") -> str:
    """Get main article text from a URL via web_search_preview."""
    try:
        resp = client.responses.create(
            model=model,
            tools=[{"type": "web_search_preview"}],
            input=(
                "Extract and return ONLY the main article text (no boilerplate, navigation, or ads) "
                "from this exact URL. No commentary—just the text:\n"
                f"{url}"
            ),
        )
        return (getattr(resp, "output_text", "") or "").strip()
    except Exception as e:
        return f"[ERROR extracting text for {url}: {e}]"

def _summarize_text_for_company(client: OpenAI, text: str, company: str, model: str = "gpt-4.1") -> str:
    """Summarize article text specifically for the given company."""
    if not text or text.startswith("[ERROR"):
        return text
    try:
        prompt = (
            f"Summarize the following article specifically in relation to {company}.\n"
            f"Focus on: business impact, catalysts (e.g., trials/approvals, earnings, M&A), "
            f"and overall tone (bullish/bearish/neutral) with a one-line justification.\n\n"
            f"Article:\n{text}"
        )
        resp = client.responses.create(model=model, input=prompt)
        return (getattr(resp, "output_text", "") or "").strip()
    except Exception as e:
        return f"[ERROR summarizing for {company}: {e}]"

def _explain_article_weighted_sentiment(
    client: OpenAI,
    article: Dict[str, Any],
    company: str,
    period: str,
    model: str = "gpt-4.1"
) -> str:
    """Explain why this article affects the period's weighted sentiment and how significant it is."""
    title = article.get("title", "")
    url = article.get("url", "")
    time_published = article.get("time_published", "")
    rel = article.get("relevance_score", 0.0)
    s = article.get("ticker_sentiment_score", 0.0)
    w = article.get("weight", 0.0)
    impact = s * w
    summary = article.get("llm_summary", "")
    text = article.get("llm_text", "")
    context_text = summary if summary else text

    prompt = f"""
You are an equity news analyst.

Goal:
Explain why THIS single article influences the overall {period} weighted sentiment for {company} the way it does,
and assess how significant that influence is.

Article metadata:
- Title: {title}
- URL: {url}
- Published: {time_published}
- Ticker sentiment score (s): {s:+.3f}  (range ~[-1,1])
- Relevance (r): {rel:.3f}  (importance of company within article)
- Weight (≈ relevance × recency): {w:.3f}
- Weighted impact (s × weight): {impact:+.3f}

Article (company-focused) summary or text:
{context_text}

Instructions:
- Start with a one-line verdict: (Bullish/Bearish/Neutral) + (High/Medium/Low) significance.
- Then 2–4 short bullets:
  • Why the tone is positive/negative/neutral for {company}.
  • What concrete drivers (e.g., earnings, trials/approvals, guidance, M&A, litigation) the article cites.
  • Why its weight is high/low (relevance to {company}, recency).
- Keep it crisp. Avoid fluff. Do not repeat the raw numbers—refer to them conceptually.
"""
    try:
        resp = client.responses.create(model=model, input=prompt)
        return (getattr(resp, "output_text", "") or "").strip()
    except Exception as e:
        return f"[ERROR generating article reason: {e}]"

def _attach_llm_content_for_articles(
    client: OpenAI,
    articles: List[Dict[str, Any]],
    company: str,
    period: str,
    delay_s: float = 0.0
) -> List[Dict[str, Any]]:
    """
    For each article dict with a 'url', attach:
      - 'llm_text'    : extracted main text from the URL
      - 'llm_summary' : summary focused on the specified company
      - 'llm_reason'  : why this article has its weighted sentiment impact in this timeframe
    """
    enriched = []
    for a in articles:
        url = a.get("url", "")

        text = _extract_url_text_with_llm(client, url)
        if delay_s: time.sleep(delay_s)

        summary = _summarize_text_for_company(client, text, company)
        if delay_s: time.sleep(delay_s)

        with_context = {**a, "llm_text": text, "llm_summary": summary}
        reason = _explain_article_weighted_sentiment(client, with_context, company, period)

        aa = dict(a)
        aa["llm_text"] = text
        aa["llm_summary"] = summary
        aa["llm_reason"] = reason
        enriched.append(aa)

        if delay_s: time.sleep(delay_s)
    return enriched

def _synthesize_overall_analysis(
    client: OpenAI,
    summaries: List[str],
    company: str,
    period: str,
    polarity: str,  # "positive" or "negative"
    model: str = "gpt-4.1"
) -> str:
    """Combine multiple article-level summaries into one concise overall analysis."""
    summaries = [s for s in summaries if s and not s.startswith("[ERROR")]
    if not summaries:
        return f"[No valid summaries available to synthesize for {polarity} set]"

    joined = "\n\n".join(f"- {s}" for s in summaries[:3])  # top 3
    prompt = f"""
You are an equity news analyst.

Task:
Produce ONE concise overall analysis that synthesizes the top 3 {polarity} (by weighted impact) article summaries
for {company} covering the {period} period.

Input (3 summaries):
{joined}

Instructions:
- Start with a one-line verdict for {company} over {period} (Bullish/Bearish/Neutral) + confidence (High/Medium/Low).
- Then provide 3–5 bullets:
  • Common themes and drivers (earnings, trials/approvals, guidance, M&A, litigation, partnerships).
  • Net effect on company outlook (near-term vs mid-term).
  • Any contradictions across the articles and how to interpret them.
  • Key risks or watch-outs that could flip sentiment.
- Be concise and avoid repeating the same sentence structure. No fluff.
"""
    try:
        resp = client.responses.create(model=model, input=prompt)
        return (getattr(resp, "output_text", "") or "").strip()
    except Exception as e:
        return f"[ERROR synthesizing overall analysis for {polarity}: {e}]"

# =============== Public function: returns gathered_data ===============

def get_news_sentiment_gathered_data(
    ticker: str,
    period: str,
    company_name: str,
    alpha_vantage_api_key: str,
    openai_api_key: str,
    topics: Optional[List[str]] = None,
    min_relevance: float = 0.20,
    use_recency_decay: bool = True,
    llm_delay_s: float = 0.0   # increase if you hit rate limits
) -> Dict[str, Any]:
    """
    Fetches Alpha Vantage NEWS_SENTIMENT for a ticker and period, computes weighted sentiment,
    ranks top/bottom articles, enriches top/bottom (weighted) with LLM (text, summary, reason),
    synthesizes overall analyses, and returns a unified gathered_data dict.
    """
    # Step 1: Fetch & compute sentiment
    res = _get_sentiment_for_ticker(
        apikey=alpha_vantage_api_key,
        ticker=ticker,
        period=period,
        topics=topics,
        min_relevance=min_relevance,
        use_recency_decay=use_recency_decay
    )

    company_name = company_name # swap to a nicer mapping if you have one
    articles = res["articles"]

    # Step 2: Rankings
    top3_raw = sorted(articles, key=lambda a: a["ticker_sentiment_score"], reverse=True)[:3]
    bottom3_raw = sorted(articles, key=lambda a: a["ticker_sentiment_score"])[:3]

    top3_weighted = sorted(articles, key=lambda a: a["ticker_sentiment_score"] * a["weight"], reverse=True)[:3]
    bottom3_weighted = sorted(articles, key=lambda a: a["ticker_sentiment_score"] * a["weight"])[:3]

    # Step 3: LLM enrichment for weighted sets
    client = OpenAI(api_key=openai_api_key)
    top3_weighted_enriched = _attach_llm_content_for_articles(
        client=client,
        articles=top3_weighted,
        company=company_name,
        period=period,
        delay_s=llm_delay_s
    )
    bottom3_weighted_enriched = _attach_llm_content_for_articles(
        client=client,
        articles=bottom3_weighted,
        company=company_name,
        period=period,
        delay_s=llm_delay_s
    )

    # Step 4: Synthesize overall analyses from the two sets
    pos_summaries = [a.get("llm_summary", "") for a in top3_weighted_enriched]
    neg_summaries = [a.get("llm_summary", "") for a in bottom3_weighted_enriched]
    overall_positive = _synthesize_overall_analysis(
        client=client,
        summaries=pos_summaries,
        company=company_name,
        period=period,
        polarity="positive"
    )
    overall_negative = _synthesize_overall_analysis(
        client=client,
        summaries=neg_summaries,
        company=company_name,
        period=period,
        polarity="negative"
    )

    # Step 5: Build gathered_data (only what we actually compute)
    def _impact(a: dict) -> float:
        return float(a.get("ticker_sentiment_score", 0)) * float(a.get("weight", 0))

    def _project_minimal(a: dict) -> dict:
        return {
            "title": a.get("title"),
            "url": a.get("url"),
            "time_published": a.get("time_published"),
            "source": a.get("source"),
            "relevance_score": a.get("relevance_score"),
            "ticker_sentiment_score": a.get("ticker_sentiment_score"),
            "weight": a.get("weight"),
            "impact": _impact(a),
        }

    def _project_enriched(a: dict) -> dict:
        base = _project_minimal(a)
        base.update({
            "llm_summary": a.get("llm_summary"),
            "llm_reason": a.get("llm_reason"),
            "llm_text": a.get("llm_text"),
        })
        return base

    all_articles_minimal = [_project_minimal(a) for a in articles]
    top3_raw_min = [_project_minimal(a) for a in top3_raw]
    bottom3_raw_min = [_project_minimal(a) for a in bottom3_raw]
    top3_weighted_full = [_project_enriched(a) for a in top3_weighted_enriched]
    bottom3_weighted_full = [_project_enriched(a) for a in bottom3_weighted_enriched]

    gathered_data = {
        "Context": {
            "Ticker": ticker.upper(),
            "Company": company_name,
            "Timeframe": period,                # "3m" | "6m" | "1y"
            "TopicsFilter": topics or [],
            "MinRelevanceFilter": float(min_relevance),
            "UseRecencyDecay": bool(use_recency_decay),
        },
        "Sentiment": {
            "WeightedAvg": float(round(res["weighted_avg"], 6)),
            "Label": res["label"],
            "NArticles": int(res["n_articles"]),
            "TotalWeight": float(round(res["total_weight"], 6)),
        },
        "Articles": {
            "AllQualifying": all_articles_minimal,
            "Top3RawPositive": top3_raw_min,
            "Top3RawNegative": bottom3_raw_min,
            "Top3WeightedPositive": top3_weighted_full,
            "Top3WeightedNegative": bottom3_weighted_full,
        },
        "Synthesis": {
            "PositiveOverallAnalysis": overall_positive,
            "NegativeOverallAnalysis": overall_negative,
        },
    }

    return gathered_data


# Optional: quick manual test
# if __name__ == "__main__":
#     import os
#     TICKER = os.getenv("TEST_TICKER", "MRNA")
#     PERIOD = os.getenv("TEST_PERIOD", "3m")
#     AV_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "YOUR_AV_KEY")
#     OA_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY")

#     result = get_news_sentiment_gathered_data(
#         ticker=TICKER,
#         period=PERIOD,
#         alpha_vantage_api_key=AV_KEY,
#         openai_api_key=OA_KEY,
#         topics=None,          # or ["life_sciences","earnings"] etc.
#         min_relevance=0.20,
#         use_recency_decay=True,
#         llm_delay_s=0.0
#     )
#     print(json.dumps(result, indent=2))
