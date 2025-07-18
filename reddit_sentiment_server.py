import logging
import json
import anthropic
import asyncpraw
import asyncio
import httpx
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP
from asyncio import TimeoutError, wait_for

mcp = FastMCP("reddit-sentiment-analyzer")

# Note: Replace with your actual credentials or use environment variables
REDDIT_CLIENT_ID = "YOUR_REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT = "SentimentBot/1.0 by YOUR_USERNAME"
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"

@dataclass
class ClaudeSentimentResult:
    sentiment_score: float
    sentiment_label: str
    confidence: float
    reasoning: str
    key_themes: List[str]
    pain_points: List[str]
    feature_requests: List[str]
    customer_intent: str
    urgency_level: str
    business_impact: str

async def setup_reddit_client():
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    reddit.read_only = True
    return reddit

def analyze_sentiment_simple(text: str) -> Tuple[float, str]:
    positive_words = ['love', 'amazing', 'great', 'excellent', 'fantastic', 'wonderful', 'perfect', 'awesome', 'brilliant', 'outstanding', 'helpful', 'useful']
    negative_words = ['hate', 'terrible', 'awful', 'horrible', 'bad', 'poor', 'useless', 'frustrating', 'annoying', 'disappointed', 'broken', 'crash']
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    total_words = len(text_lower.split())
    if total_words == 0:
        return 0.0, "neutral"
    score = (positive_count - negative_count) / max(total_words * 0.1, 1)
    score = max(-1, min(1, score))
    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"
    return score, label

async def analyze_sentiment_with_claude_batch(posts: List[str], product_context: str = "") -> List[ClaudeSentimentResult]:
    prompt = f"""
    You are a product analyst. Analyze the sentiment of each Reddit post below.
    Product context: {product_context if product_context else "(none)"}
    Provide a list of JSON results with the following structure:

    [
        {{
            "score": float,
            "label": "positive|negative|neutral|mixed",
            "confidence": float,
            "reasoning": "...",
            "primary_themes": [...],
            "pain_points": [...],
            "feature_requests": [...],
            "customer_intent": "...",
            "urgency_level": "...",
            "business_impact": "..."
        }},
        ...
    ]

    Posts:
    {json.dumps(posts)}
    """

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-5-sonnet-20241022",
                "temperature": 0.1,
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        text = response.json()["content"][0]["text"]
        json_start = text.find("[")
        json_end = text.rfind("]") + 1
        parsed = json.loads(text[json_start:json_end])

        return [ClaudeSentimentResult(
            sentiment_score=p.get("score", 0.0),
            sentiment_label=p.get("label", "neutral"),
            confidence=p.get("confidence", 0.5),
            reasoning=p.get("reasoning", ""),
            key_themes=p.get("primary_themes", []),
            pain_points=p.get("pain_points", []),
            feature_requests=p.get("feature_requests", []),
            customer_intent=p.get("customer_intent", "unknown"),
            urgency_level=p.get("urgency_level", "medium"),
            business_impact=p.get("business_impact", "medium")
        ) for p in parsed]

def aggregate_claude_results(results: List[ClaudeSentimentResult]) -> Dict:
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    total_sentiment_score = 0
    high_confidence_results = [r for r in results if r.confidence > 0.7]
    for result in high_confidence_results:
        label = result.sentiment_label.lower()
        if label in sentiment_counts:
            sentiment_counts[label] += 1
        else:
            if result.sentiment_score > 0.1:
                sentiment_counts["positive"] += 1
            elif result.sentiment_score < -0.1:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1
        total_sentiment_score += result.sentiment_score
    all_themes = [theme for r in high_confidence_results for theme in r.key_themes]
    theme_frequency = {theme: all_themes.count(theme) for theme in set(all_themes)}
    top_themes = sorted(theme_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    all_pain_points = [pt for r in high_confidence_results for pt in r.pain_points][:20]
    all_feature_requests = [fr for r in high_confidence_results for fr in r.feature_requests][:20]
    urgent_issues = [r for r in high_confidence_results if r.urgency_level == "high"]
    high_impact_feedback = [r for r in high_confidence_results if r.business_impact == "high"]
    return {
        "analysis_summary": {
            "total_posts_analyzed": len(results),
            "high_confidence_posts": len(high_confidence_results),
            "average_sentiment_score": total_sentiment_score / len(high_confidence_results) if high_confidence_results else 0
        },
        "sentiment_distribution": sentiment_counts,
        "top_themes": top_themes,
        "customer_insights": {
            "pain_points": all_pain_points,
            "feature_requests": all_feature_requests,
            "urgent_issues_count": len(urgent_issues),
            "high_impact_feedback_count": len(high_impact_feedback)
        }
    }

def create_summary(results: Dict) -> Dict:
    total_posts = results.get("total_posts", 0)
    sentiment_dist = results.get("sentiment_distribution", {})
    sentiment_percentages = {k: f"{(v / total_posts) * 100:.1f}%" for k, v in sentiment_dist.items()} if total_posts else {}
    insights = results.get("customer_insights", {})
    return {
        "overview": {
            "total_posts_analyzed": total_posts,
            "analysis_method": "claude_ai"
        },
        "sentiment_breakdown": {
            "distribution": sentiment_dist,
            "percentages": sentiment_percentages,
            "overall_sentiment": max(sentiment_dist, key=sentiment_dist.get) if sentiment_dist else "neutral"
        },
        "key_insights": {
            "top_themes": [t[0] for t in results.get("top_themes", [])],
            "main_pain_points": insights.get("pain_points", []),
            "top_feature_requests": insights.get("feature_requests", [])
        },
        "business_metrics": {
            "urgent_issues": insights.get("urgent_issues_count", 0),
            "high_impact_items": insights.get("high_impact_feedback_count", 0)
        },
        "recommendation": "Would you like to see the full detailed analysis with individual post data?"
    }

@mcp.tool()
async def safe_analyze_reddit_sentiment(
    query: str,
    subreddits: Optional[List[str]] = None,
    time_filter: str = "week",
    limit: int = 5,
    use_claude: bool = True,
    product_context: str = "",
    return_full_data: bool = False
) -> Dict:
    """
    Analyze Reddit sentiment for a given query across specified subreddits.
    
    Args:
        query: Search term to analyze
        subreddits: List of subreddit names (default: ["all"])
        time_filter: Time period ("hour", "day", "week", "month", "year", "all")
        limit: Maximum posts per subreddit
        use_claude: Use Claude AI for advanced analysis (default: True)
        product_context: Additional context about the product being analyzed
        return_full_data: Return detailed post-level data (default: False)
    
    Returns:
        Dictionary containing sentiment analysis results
    """
    try:
        return await wait_for(
            analyze_reddit_sentiment(query, subreddits, time_filter, limit, use_claude, product_context, return_full_data),
            timeout=55
        )
    except TimeoutError:
        return {"error": "The tool took too long. Try reducing the number of posts or subreddits."}

async def analyze_reddit_sentiment(query, subreddits, time_filter, limit, use_claude, product_context, return_full_data):
    reddit = await setup_reddit_client()
    all_posts = []
    post_texts = []

    if subreddits is None:
        subreddits = ["all"]

    for sr in subreddits:
        subreddit = await reddit.subreddit(sr)
        async for post in subreddit.search(query, sort="relevance", time_filter=time_filter, limit=limit):
            if post.selftext in ['[removed]', '[deleted]'] or post.author is None:
                continue
            text = f"{post.title} {post.selftext}"
            all_posts.append({"id": post.id, "text": text, "meta": {
                "author": str(post.author),
                "subreddit": str(post.subreddit),
                "url": f"https://reddit.com{post.permalink}"}})
            post_texts.append(text)

    if use_claude:
        results = await analyze_sentiment_with_claude_batch(post_texts, product_context)
    else:
        results = [ClaudeSentimentResult(*analyze_sentiment_simple(p["text"]), 0.5, "simple fallback", [], [], [], "unknown", "medium", "medium") for p in all_posts]

    full = [vars(r) | p["meta"] | {"text": p["text"]} for r, p in zip(results, all_posts)]
    agg = aggregate_claude_results(results)
    agg["total_posts"] = len(results)

    return full if return_full_data else create_summary(agg)

if __name__ == "__main__":
    # FastMCP automatically uses PORT environment variable for HTTP transport
    # No need to manually specify host/port
    mcp.run()
