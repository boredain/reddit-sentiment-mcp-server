import logging
import json
import os
import anthropic
import asyncpraw
import asyncio
import httpx
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from fastmcp import FastMCP
from asyncio import TimeoutError, wait_for

# Configure logging
logging.basicConfig(level=logging.INFO)

# Get port from environment variable (Render sets this automatically)
port = int(os.environ.get("PORT", 8000))

# Initialize MCP server with proper configuration
mcp = FastMCP(
    "reddit-sentiment-analyzer")

# Use environment variables for credentials
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "SentimentBot/1.0")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Validate required environment variables
required_vars = {
    "REDDIT_CLIENT_ID": REDDIT_CLIENT_ID,
    "REDDIT_CLIENT_SECRET": REDDIT_CLIENT_SECRET,
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    logging.warning(f"Missing environment variables: {', '.join(missing_vars)}")
    logging.warning("Server will start but functionality may be limited without proper credentials")

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
    """Initialize Reddit client with credentials"""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        raise Exception("Reddit credentials not configured")
    
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    reddit.read_only = True
    return reddit

def analyze_sentiment_simple(text: str) -> Tuple[float, str]:
    """Simple rule-based sentiment analysis fallback"""
    positive_words = ['love', 'amazing', 'great', 'excellent', 'fantastic', 'wonderful', 
                     'perfect', 'awesome', 'brilliant', 'outstanding', 'helpful', 'useful']
    negative_words = ['hate', 'terrible', 'awful', 'horrible', 'bad', 'poor', 'useless', 
                      'frustrating', 'annoying', 'disappointed', 'broken', 'crash']
    
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
    """Analyze sentiment using Claude AI"""
    if not ANTHROPIC_API_KEY:
        logging.warning("No Anthropic API key configured, falling back to simple sentiment analysis")
        return [ClaudeSentimentResult(
            *analyze_sentiment_simple(post), 0.5, "simple fallback", 
            [], [], [], "unknown", "medium", "medium"
        ) for post in posts]
    
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

    try:
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
            response.raise_for_status()
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
    except Exception as e:
        logging.error(f"Claude API error: {e}")
        return [ClaudeSentimentResult(
            *analyze_sentiment_simple(post), 0.5, f"API error fallback: {str(e)}", 
            [], [], [], "unknown", "medium", "medium"
        ) for post in posts]

def aggregate_claude_results(results: List[ClaudeSentimentResult]) -> Dict:
    """Aggregate individual sentiment results into business insights"""
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
    """Create executive summary from aggregated results"""
    total_posts = results.get("total_posts", 0)
    sentiment_dist = results.get("sentiment_distribution", {})
    sentiment_percentages = {
        k: f"{(v / total_posts) * 100:.1f}%" 
        for k, v in sentiment_dist.items()
    } if total_posts else {}
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
async def analyze_reddit_sentiment(
    query: str,
    subreddits: Optional[List[str]] = None,
    time_filter: str = "week",
    limit: int = 5,
    use_claude: bool = True,
    product_context: str = "",
    return_full_data: bool = False
) -> Dict:

    await asyncio.sleep(2)
    """
    Analyze Reddit sentiment for a given query across specified subreddits.
    
    This tool provides enterprise-grade sentiment analysis with business intelligence
    extraction including pain points, feature requests, and urgency assessment.
    
    Args:
        query: Search term to analyze (e.g., product name, brand, topic)
        subreddits: List of subreddit names to search (default: ["all"])
        time_filter: Time period - "hour", "day", "week", "month", "year", or "all"
        limit: Maximum posts per subreddit (1-25, default: 5)
        use_claude: Use Claude AI for advanced analysis (default: True)
        product_context: Additional business context for better analysis
        return_full_data: Return detailed post-level data (default: False)
    
    Returns:
        Dictionary containing:
        - Sentiment distribution and percentages
        - Key themes and insights
        - Business metrics (urgent issues, high-impact items)
        - Pain points and feature requests
    """
    try:
        # Validate inputs
        if not query:
            return {"error": "Query parameter is required"}
        
        if limit < 1 or limit > 25:
            return {"error": "Limit must be between 1 and 25"}
        
        # Execute with timeout protection
        result = await wait_for(
            _analyze_reddit_sentiment_internal(
                query, subreddits, time_filter, limit, 
                use_claude, product_context, return_full_data
            ),
            timeout=55
        )
        return result
    except TimeoutError:
        return {
            "error": "Analysis timed out. Try reducing the number of posts or subreddits.",
            "suggestion": "Consider using limit=3 or fewer subreddits for faster results"
        }
    except Exception as e:
        logging.error(f"Error in analyze_reddit_sentiment: {e}")
        return {
            "error": f"An error occurred: {str(e)}",
            "suggestion": "Please check your parameters and try again"
        }

async def _analyze_reddit_sentiment_internal(
    query, subreddits, time_filter, limit, 
    use_claude, product_context, return_full_data
):
    """Internal implementation of sentiment analysis"""
    reddit = await setup_reddit_client()
    all_posts = []
    post_texts = []

    if subreddits is None:
        subreddits = ["all"]

    # Collect posts from each subreddit
    for sr in subreddits:
        try:
            subreddit = await reddit.subreddit(sr)
            async for post in subreddit.search(
                query, sort="relevance", 
                time_filter=time_filter, limit=limit
            ):
                if post.selftext in ['[removed]', '[deleted]'] or post.author is None:
                    continue
                
                text = f"{post.title} {post.selftext}"
                all_posts.append({
                    "id": post.id, 
                    "text": text, 
                    "meta": {
                        "author": str(post.author),
                        "subreddit": str(post.subreddit),
                        "url": f"https://reddit.com{post.permalink}"
                    }
                })
                post_texts.append(text)
        except Exception as e:
            logging.warning(f"Error fetching from r/{sr}: {e}")
            continue

    if not all_posts:
        return {
            "error": "No posts found matching your query",
            "suggestion": "Try different subreddits or a broader search term"
        }

    # Analyze sentiment
    if use_claude:
        results = await analyze_sentiment_with_claude_batch(post_texts, product_context)
    else:
        results = [
            ClaudeSentimentResult(
                *analyze_sentiment_simple(p["text"]), 0.5, 
                "simple fallback", [], [], [], 
                "unknown", "medium", "medium"
            ) for p in all_posts
        ]

    # Prepare response
    if return_full_data:
        full = [
            {**vars(r), **p["meta"], "text": p["text"][:500] + "..." if len(p["text"]) > 500 else p["text"]} 
            for r, p in zip(results, all_posts)
        ]
        return {"posts": full, "total": len(full)}
    else:
        agg = aggregate_claude_results(results)
        agg["total_posts"] = len(results)
        return create_summary(agg)

if __name__ == "__main__":
    mcp.run(
        transport="streamable_http",  # ✅ This is supported!
        host="0.0.0.0", 
        port=port,
        path="/mcp/"  # Keep trailing slash to match client requests and avoid redirects
    )




