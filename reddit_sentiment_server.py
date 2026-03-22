#!/usr/bin/env python3
"""
Routes all JSON-RPC responses through SSE stream (not HTTP response body)
"""

import asyncio
import json
import logging
import os
import sys
import socket
import uuid
from typing import Any, Dict, List
from dataclasses import dataclass
import httpx
import threading
from dotenv import load_dotenv

# Official MCP imports
from mcp.server import Server
import mcp.types as types

# FastAPI for SSE transport wrapper
from fastapi import FastAPI, Request, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

 

# Environment variables
# Load variables from a local .env file for local development.
# Existing environment variables (e.g., Render dashboard) take precedence.
load_dotenv()
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "RedditSentimentBot/1.0")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DUST_MCP_BEARER_TOKEN = os.environ.get("DUST_MCP_BEARER_TOKEN", "")

# Create the REAL MCP server instance
mcp_server = Server("reddit-sentiment-analyzer")

# Session management for SSE streams
active_sessions: Dict[str, asyncio.Queue] = {}
session_lock = threading.Lock()

# Request deduplication to prevent duplicate API calls
in_progress_analyses: Dict[str, asyncio.Future] = {}
dedup_lock = asyncio.Lock()

# Prompt caching for external prompt files
_prompt_cache: Dict[str, str] = {}

def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt from the prompts directory with caching.

    Args:
        prompt_name: Name of the prompt file (without .txt extension)

    Returns:
        The prompt content as a string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    if prompt_name in _prompt_cache:
        return _prompt_cache[prompt_name]

    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", f"{prompt_name}.txt")

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        _prompt_cache[prompt_name] = prompt_content
        logger.info(f"Loaded prompt: {prompt_name}")
        return prompt_content
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading prompt {prompt_name}: {e}")
        raise

@dataclass
class SentimentResult:
    post_index: int
    sentiment_score: float
    sentiment_label: str
    confidence: float
    reasoning: str
    key_themes: List[str]
    pain_points: List[str]
    feature_requests: List[str]

# ============================================================================
# REAL MCP SERVER IMPLEMENTATION (Official SDK)
# ============================================================================

@mcp_server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """REAL MCP: List available tools using official MCP types"""
    return [
        types.Tool(
            name="analyze_reddit_sentiment",
            description="Analyze Reddit sentiment for business intelligence with AI-powered insights including pain points, feature requests, and urgency assessment",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term to analyze (e.g., product name, brand, topic)"
                    },
                    "subreddits": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of subreddit names to search",
                        "default": ["all"]
                    },
                    "time_filter": {
                        "type": "string",
                        "enum": ["hour", "day", "week", "month", "year", "all"],
                        "description": "Time period filter",
                        "default": "all"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Ignored in this build; total posts fixed to 50 across all subreddits",
                        "default": 50
                    },
                    "product_context": {
                        "type": "string",
                        "description": "Additional business context for analysis",
                        "default": ""
                    }
                },
                "required": ["query"]
            }
        )
    ]

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """REAL MCP: Handle tool calls using official MCP types"""
    if name != "analyze_reddit_sentiment":
        raise ValueError(f"Unknown tool: {name}")

    try:
        # Extract parameters with validation
        query = arguments.get("query", "")
        if not query:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "Query parameter is required"}, indent=2)
            )]

        subreddits = arguments.get("subreddits", ["all"])
        time_filter = arguments.get("time_filter", "all")
        # Enforce a fixed total of 50 posts across all subreddits, regardless of user input
        limit = 50

        # Debug log to verify enforced total limit and inputs
        logger.info(f"Enforcing total limit=50; subreddits={subreddits}, time_filter={time_filter}")
        product_context = arguments.get("product_context", "")

        # Perform sentiment analysi
        result = await perform_sentiment_analysis(
            query, subreddits, time_filter, limit, product_context
        )

        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "error": f"Analysis failed: {str(e)}",
                "query": arguments.get("query", "unknown"),
                "suggestion": "Please try again with different parameters"
            }, indent=2)
        )]

# ============================================================================
# BUSINESS LOGIC (Sentiment Analysis Implementation)
# ============================================================================

def analyze_sentiment_simple(text: str) -> tuple[float, str]:
    """Simple rule-based sentiment analysis"""
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

def calculate_ranked_insights(items_with_indices: List[tuple[str, int]], total_posts: int, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Count occurrences, calculate percentages, track source indices, and rank by frequency

    Args:
        items_with_indices: List of (theme, post_index) tuples
        total_posts: Total number of posts analyzed
        limit: Maximum number of insights to return (default 10)

    Returns:
        List of dicts with theme, count, percentage, and source_indices, sorted by count descending
    """
    if not items_with_indices:
        return []

    # Build a mapping of theme -> list of post indices
    theme_to_indices = {}
    for theme, post_idx in items_with_indices:
        if theme not in theme_to_indices:
            theme_to_indices[theme] = []
        theme_to_indices[theme].append(post_idx)

    # Build ranked list with percentages and source indices
    insights = []
    for theme, indices in theme_to_indices.items():
        count = len(indices)
        percentage = (count / total_posts) * 100
        # Remove duplicates and sort indices
        unique_indices = sorted(set(indices))
        insights.append({
            "theme": theme,
            "count": count,
            "percentage": round(percentage, 1),
            "source_indices": unique_indices
        })

    # Sort by count descending (most popular first)
    insights.sort(key=lambda x: x["count"], reverse=True)

    # Return top N
    return insights[:limit]

# Batch processing configuration
BATCH_SIZE = 10
MAX_TOKENS = 2000

async def analyze_with_openai(posts: List[str], context: str, query: str = "") -> tuple[List[SentimentResult], str]:
    """Analyze sentiment using OpenAI GPT-4 with batch processing for all posts

    Returns:
        tuple: (list of SentimentResults, analysis_method string)
    """
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not configured")
        return [], "openai_key_missing"

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        logger.error("OpenAI package not installed")
        return [], "openai_import_error"

    all_results = []
    total_batches = (len(posts) + BATCH_SIZE - 1) // BATCH_SIZE

    # Process in batches
    for batch_idx in range(0, len(posts), BATCH_SIZE):
        batch = posts[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1

        logger.info(f"[OpenAI] Processing batch {batch_num}/{total_batches}")

        # Load prompt template from external file and format with variables
        prompt_template = load_prompt("analysis_prompt")
        prompt = prompt_template.format(
            query=query or "a product",
            context=context or f"General discussion about {query or 'the product'}",
            batch=json.dumps(batch)
        )

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing customer feedback and extracting actionable business insights."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=MAX_TOKENS
                ),
                timeout=40.0
            )

            # Check for refusals
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                logger.error(f"[OpenAI] Model refused: {response.choices[0].message.refusal}")
                return [], f"openai_refusal_batch_{batch_num}"

            text = response.choices[0].message.content or ""
            logger.info(f"[OpenAI] Batch {batch_num} API call successful")
            logger.info(f"[OpenAI] Response length: {len(text)} chars")
            logger.info(f"[OpenAI] Response preview: {text[:500] if text else '(empty)'}...")  # Log first 500 chars

            try:
                json_start = text.find("[")
                json_end = text.rfind("]") + 1
                if json_start != -1 and json_end > json_start:
                    parsed = json.loads(text[json_start:json_end])

                    # Validate array length matches batch size; be tolerant by padding/truncating
                    if len(parsed) != len(batch):
                        logger.warning(
                            f"[OpenAI] Batch {batch_num} length mismatch. Expected {len(batch)}, got {len(parsed)}. Adjusting to maintain alignment."
                        )
                        if len(parsed) < len(batch):
                            # Pad with empty results to preserve index alignment
                            pad_count = len(batch) - len(parsed)
                            parsed.extend([{"themes": [], "pain_points": [], "requests": []} for _ in range(pad_count)])
                        else:
                            # Truncate any extra items beyond the batch size
                            parsed = parsed[:len(batch)]

                    for local_idx, p in enumerate(parsed):
                        global_idx = batch_idx + local_idx  # Calculate global post index
                        all_results.append(SentimentResult(
                            global_idx,  # post index
                            0.0,  # score not needed
                            "neutral",  # label not needed
                            1.0,  # confidence not needed
                            "",  # reasoning not needed
                            p.get("themes", []),
                            p.get("pain_points", []),
                            p.get("requests", [])
                        ))
                    logger.info(f"[OpenAI] Batch {batch_num} processed successfully: {len(parsed)} posts")
                else:
                    logger.error(f"[OpenAI] Full response text: {text}")
                    raise ValueError("No JSON array found in response")
            except Exception as parse_error:
                logger.error(f"[OpenAI] Failed to parse batch {batch_num}: {parse_error}")
                logger.error(f"[OpenAI] Response length: {len(text)} chars")
                return [], f"openai_parse_error_batch_{batch_num}"

        except asyncio.TimeoutError:
            logger.error(f"[OpenAI] Batch {batch_num} timeout after 40s")
            return [], f"openai_timeout_batch_{batch_num}"
        except Exception as e:
            logger.error(f"[OpenAI] Batch {batch_num} exception: {e}")
            return [], f"openai_error_batch_{batch_num}"

    logger.info(f"[OpenAI] Total posts analyzed: {len(all_results)}/{len(posts)}")
    return all_results, "openai"

async def analyze_with_claude(posts: List[str], context: str, query: str = "") -> tuple[List[SentimentResult], str]:
    """Analyze sentiment using Claude AI with batch processing for all posts

    Returns:
        tuple: (list of SentimentResults, analysis_method string)
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("Anthropic API key not configured")
        return [], "claude_key_missing"

    all_results = []
    total_batches = (len(posts) + BATCH_SIZE - 1) // BATCH_SIZE

    # Process in batches
    for batch_idx in range(0, len(posts), BATCH_SIZE):
        batch = posts[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1

        logger.info(f"[Claude] Processing batch {batch_num}/{total_batches}")

        # Load prompt template from external file and format with variables
        prompt_template = load_prompt("analysis_prompt")
        prompt = prompt_template.format(
            query=query or "a product",
            context=context or f"General discussion about {query or 'the product'}",
            batch=json.dumps(batch)
        )

        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": MAX_TOKENS,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )

                if response.status_code == 200:
                    body = response.json()
                    text = body.get("content", [{}])[0].get("text", "")
                    logger.info(f"[Claude] Batch {batch_num} API call successful")
                    try:
                        json_start = text.find("[")
                        json_end = text.rfind("]") + 1
                        if json_start != -1 and json_end > json_start:
                            parsed = json.loads(text[json_start:json_end])

                            # Validate array length matches batch size; be tolerant by padding/truncating
                            if len(parsed) != len(batch):
                                logger.warning(
                                    f"[Claude] Batch {batch_num} length mismatch. Expected {len(batch)}, got {len(parsed)}. Adjusting to maintain alignment."
                                )
                                if len(parsed) < len(batch):
                                    pad_count = len(batch) - len(parsed)
                                    parsed.extend([{"themes": [], "pain_points": [], "requests": []} for _ in range(pad_count)])
                                else:
                                    parsed = parsed[:len(batch)]

                            for local_idx, p in enumerate(parsed):
                                global_idx = batch_idx + local_idx  # Calculate global post index
                                all_results.append(SentimentResult(
                                    global_idx,  # post index
                                    0.0,  # score not needed
                                    "neutral",  # label not needed
                                    1.0,  # confidence not needed
                                    "",  # reasoning not needed
                                    p.get("themes", []),
                                    p.get("pain_points", []),
                                    p.get("requests", [])
                                ))
                            logger.info(f"[Claude] Batch {batch_num} processed successfully: {len(parsed)} posts")
                        else:
                            raise ValueError("No JSON array found in response")
                    except Exception as parse_error:
                        logger.error(f"[Claude] Failed to parse batch {batch_num}: {parse_error}")
                        return [], f"claude_parse_error_batch_{batch_num}"
                else:
                    logger.error(f"[Claude] Batch {batch_num} API error: status {response.status_code}")
                    return [], f"claude_api_error_{response.status_code}"
        except asyncio.TimeoutError:
            logger.error(f"[Claude] Batch {batch_num} timeout after 40s")
            return [], f"claude_timeout_batch_{batch_num}"
        except Exception as e:
            logger.error(f"[Claude] Batch {batch_num} exception: {e}")
            return [], f"claude_error_batch_{batch_num}"

    logger.info(f"[Claude] Total posts analyzed: {len(all_results)}/{len(posts)}")
    return all_results, "claude_ai"

def cleanup_bullet(bullet: str) -> str:
    """
    Post-processing cleanup to remove unwanted phrases and ensure format compliance.
    """
    import re

    # Remove unwanted phrases
    unwanted_patterns = [
        r'\s+to\s+\w+',  # "to reduce", "to protect", etc.
        r'\s+for\s+\w+ing',  # "for handling", "for reducing", etc.
        r'\s+causing\s+',
        r'\s+leading\s+to\s+',
        r'\s+impacting\s+',
        r'\s+resulting\s+in\s+',
        r'\s+attracting\s+',
    ]

    cleaned = bullet
    for pattern in unwanted_patterns:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)

    # Clean up extra spaces
    cleaned = ' '.join(cleaned.split())

    return cleaned

def validate_and_correct_sentiment_result(result: SentimentResult) -> SentimentResult:
    """
    Validate sentiment categorization and auto-correct obvious misclassifications.

    Checks themes (positive) for negative keywords and moves them to pain_points if found.
    Logs corrections for debugging and quality monitoring.

    Args:
        result: SentimentResult to validate

    Returns:
        Corrected SentimentResult
    """
    # Negative keywords that should never appear in themes (positive category)
    NEGATIVE_KEYWORDS = [
        'slow', 'broken', 'frustrating', 'frustrated', 'frustration', 'poor', 'low-quality',
        'disappointing', 'disappointed', 'lacking', 'difficult', 'confusing', 'confused',
        'expensive', 'unreliable', 'buggy', 'crash', 'crashing', 'failed', 'failing',
        'terrible', 'awful', 'horrible', 'worst', 'bad', 'worse', 'useless', 'annoying',
        'annoyed', 'anger', 'angry', 'hate', 'hated', 'dislike', 'declined', 'declining',
        'decreased', 'decreasing', 'loss', 'losing', 'lost', 'problem', 'problems', 'issue',
        'issues', 'concern', 'concerns', 'worried', 'worry', 'threat', 'threatening',
        'unstable', 'instability', 'outage', 'down', 'unavailable', 'unreachable',
        'theft', 'stealing', 'stolen', 'scam', 'deceptive', 'misleading', 'unfair'
    ]

    # Check themes for negative keywords and move to pain_points if found
    corrected_themes = []
    moved_to_pain_points = []

    for theme in result.key_themes:
        theme_lower = theme.lower()
        found_negative = False

        # Check if any negative keyword appears in the theme
        for keyword in NEGATIVE_KEYWORDS:
            if keyword in theme_lower:
                found_negative = True
                logger.warning(
                    f"[Validation] MISCLASSIFICATION DETECTED in post {result.post_index}: "
                    f"Theme contains negative keyword '{keyword}' - moving to pain_points"
                )
                logger.warning(f"[Validation] Misclassified theme: '{theme[:100]}...'")
                moved_to_pain_points.append(theme)
                break

        if not found_negative:
            corrected_themes.append(theme)

    # Create corrected result
    corrected_pain_points = list(result.pain_points) + moved_to_pain_points

    if moved_to_pain_points:
        logger.info(
            f"[Validation] Post {result.post_index}: Moved {len(moved_to_pain_points)} "
            f"misclassified theme(s) to pain_points"
        )

    return SentimentResult(
        post_index=result.post_index,
        sentiment_score=result.sentiment_score,
        sentiment_label=result.sentiment_label,
        confidence=result.confidence,
        reasoning=result.reasoning,
        key_themes=corrected_themes,
        pain_points=corrected_pain_points,
        feature_requests=result.feature_requests
    )

def fuzzy_match_theme(source_theme: str, detailed_themes: List[str], threshold: float = 0.6) -> str:
    """
    Find the best matching detailed theme using fuzzy string matching.
    Compares only the heading part (before "–") of detailed themes.

    Args:
        source_theme: Theme name from AI summary
        detailed_themes: List of available detailed theme names
        threshold: Minimum similarity score (0-1) to consider a match

    Returns:
        Best matching theme name or None if no match found
    """
    from difflib import SequenceMatcher

    best_match = None
    best_score = 0.0

    source_clean = source_theme.lower().strip()

    for detailed_theme in detailed_themes:
        # Extract only the heading part (before "–") for comparison
        if "–" in detailed_theme:
            detailed_heading = detailed_theme.split("–")[0].strip().lower()
        else:
            detailed_heading = detailed_theme.lower().strip()

        # Calculate similarity ratio using just the heading
        similarity = SequenceMatcher(None, source_clean, detailed_heading).ratio()

        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = detailed_theme

    return best_match

async def create_key_insights_summary(
    themes: List[Dict[str, Any]],
    pain_points: List[Dict[str, Any]],
    requests: List[Dict[str, Any]],
    query: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create concise, high-level summary bullets from detailed insights.
    Uses AI to semantically group and consolidate similar themes.
    Format: "Heading: Context" with 9-12 words total.
    Maps summary bullets to source post indices.
    """

    # Prepare the detailed insights for summarization
    themes_top_15 = themes[:15]
    pain_points_top_15 = pain_points[:15]
    requests_top_15 = requests[:15]

    themes_text = "\n".join([f"- {item['theme']}" for item in themes_top_15])
    pain_points_text = "\n".join([f"- {item['theme']}" for item in pain_points_top_15])
    requests_text = "\n".join([f"- {item['theme']}" for item in requests_top_15])

    # Create lookup maps: theme_name -> source_indices
    themes_lookup = {item['theme']: item.get('source_indices', []) for item in themes_top_15}
    pain_points_lookup = {item['theme']: item.get('source_indices', []) for item in pain_points_top_15}
    requests_lookup = {item['theme']: item.get('source_indices', []) for item in requests_top_15}

    # Load prompt template from external file and format with variables
    prompt_template = load_prompt("summary_prompt")
    prompt = prompt_template.format(
        query=query,
        themes_text=themes_text if themes_text else "None identified",
        pain_points_text=pain_points_text if pain_points_text else "None identified",
        requests_text=requests_text if requests_text else "None identified"
    )

    # Try OpenAI first
    if OPENAI_API_KEY:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)

            logger.info("[Summary] Using OpenAI to create key insights summary")
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert at synthesizing customer feedback into actionable business insights."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                ),
                timeout=30.0
            )

            text = response.choices[0].message.content
            # Extract JSON from response
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                summary_raw = json.loads(text[json_start:json_end])

                # Map source_themes to source_indices for each category
                summary = {}
                category_lookups = {
                    "what_users_like": themes_lookup,
                    "major_frustrations": pain_points_lookup,
                    "what_users_want": requests_lookup
                }

                for category, lookup in category_lookups.items():
                    summary[category] = []
                    logger.info(f"[Summary] Processing {category}: {len(lookup)} detailed themes available")
                    logger.info(f"[Summary] Available themes: {list(lookup.keys())[:5]}...")  # Show first 5
                    for bullet_obj in summary_raw.get(category, []):
                        if isinstance(bullet_obj, dict):
                            text_cleaned = cleanup_bullet(bullet_obj.get("text", ""))
                            source_themes = bullet_obj.get("source_themes", [])
                            logger.info(f"[Summary] Bullet: '{text_cleaned[:50]}...' has {len(source_themes)} source_themes: {source_themes}")

                            # Map source_themes to source_indices (exact match preferred, fuzzy as fallback)
                            all_indices = []
                            for source_theme in source_themes:
                                # Try exact match first (AI should return verbatim text)
                                if source_theme in lookup:
                                    matched_indices = lookup[source_theme]
                                    all_indices.extend(matched_indices)
                                    heading = source_theme.split("–")[0].strip() if "–" in source_theme else source_theme
                                    logger.info(f"[Summary] ✓ Exact match '{heading[:40]}...' → {len(matched_indices)} posts")
                                else:
                                    # Try fuzzy match
                                    matched = fuzzy_match_theme(source_theme, list(lookup.keys()))
                                    if matched:
                                        matched_indices = lookup[matched]
                                        all_indices.extend(matched_indices)
                                        # Extract heading for logging
                                        matched_heading = matched.split("–")[0].strip() if "–" in matched else matched
                                        logger.info(f"[Summary] Fuzzy match '{source_theme}' → '{matched_heading}' → {len(matched_indices)} posts")
                                    else:
                                        logger.warning(f"[Summary] No match found for '{source_theme}' (tried {len(lookup)} themes)")

                            # Remove duplicates and sort
                            unique_indices = sorted(set(all_indices))
                            logger.info(f"[Summary] Final: {len(unique_indices)} unique posts mapped")
                            summary[category].append({
                                "text": text_cleaned,
                                "source_indices": unique_indices
                            })
                        else:
                            # Fallback for string format (backward compatibility)
                            summary[category].append({
                                "text": cleanup_bullet(bullet_obj),
                                "source_indices": []
                            })

                logger.info("[Summary] OpenAI summarization successful")
                return summary
        except Exception as e:
            logger.warning(f"[Summary] OpenAI failed: {e}, falling back to Claude")

    # Fall back to Claude
    if ANTHROPIC_API_KEY:
        try:
            logger.info("[Summary] Using Claude to create key insights summary")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 1000,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )

                if response.status_code == 200:
                    body = response.json()
                    text = body.get("content", [{}])[0].get("text", "")
                    json_start = text.find("{")
                    json_end = text.rfind("}") + 1
                    if json_start != -1 and json_end > json_start:
                        summary_raw = json.loads(text[json_start:json_end])

                        # Map source_themes to source_indices for each category
                        summary = {}
                        category_lookups = {
                            "what_users_like": themes_lookup,
                            "major_frustrations": pain_points_lookup,
                            "what_users_want": requests_lookup
                        }

                        for category, lookup in category_lookups.items():
                            summary[category] = []
                            logger.info(f"[Summary] Processing {category}: {len(lookup)} detailed themes available")
                            logger.info(f"[Summary] Available themes: {list(lookup.keys())[:5]}...")  # Show first 5
                            for bullet_obj in summary_raw.get(category, []):
                                if isinstance(bullet_obj, dict):
                                    text_cleaned = cleanup_bullet(bullet_obj.get("text", ""))
                                    source_themes = bullet_obj.get("source_themes", [])
                                    logger.info(f"[Summary] Bullet: '{text_cleaned[:50]}...' has {len(source_themes)} source_themes: {source_themes}")

                                    # Map source_themes to source_indices (exact match preferred, fuzzy as fallback)
                                    all_indices = []
                                    for source_theme in source_themes:
                                        # Try exact match first (AI should return verbatim text)
                                        if source_theme in lookup:
                                            matched_indices = lookup[source_theme]
                                            all_indices.extend(matched_indices)
                                            heading = source_theme.split("–")[0].strip() if "–" in source_theme else source_theme
                                            logger.info(f"[Summary] ✓ Exact match '{heading[:40]}...' → {len(matched_indices)} posts")
                                        else:
                                            # Try fuzzy match
                                            matched = fuzzy_match_theme(source_theme, list(lookup.keys()))
                                            if matched:
                                                matched_indices = lookup[matched]
                                                all_indices.extend(matched_indices)
                                                # Extract heading for logging
                                                matched_heading = matched.split("–")[0].strip() if "–" in matched else matched
                                                logger.info(f"[Summary] Fuzzy match '{source_theme}' → '{matched_heading}' → {len(matched_indices)} posts")
                                            else:
                                                logger.warning(f"[Summary] No match found for '{source_theme}' (tried {len(lookup)} themes)")

                                    # Remove duplicates and sort
                                    unique_indices = sorted(set(all_indices))
                                    logger.info(f"[Summary] Final: {len(unique_indices)} unique posts mapped")
                                    summary[category].append({
                                        "text": text_cleaned,
                                        "source_indices": unique_indices
                                    })
                                else:
                                    # Fallback for string format (backward compatibility)
                                    summary[category].append({
                                        "text": cleanup_bullet(bullet_obj),
                                        "source_indices": []
                                    })

                        logger.info("[Summary] Claude summarization successful")
                        return summary
        except Exception as e:
            logger.warning(f"[Summary] Claude failed: {e}")

    # Fallback: create simple summary from top items
    logger.info("[Summary] Using simple fallback summarization")
    return {
        "what_users_like": [
            {"text": item['theme'].split('–')[0].strip(), "source_indices": item.get('source_indices', [])}
            for item in themes[:3]
        ],
        "major_frustrations": [
            {"text": item['theme'].split('–')[0].strip(), "source_indices": item.get('source_indices', [])}
            for item in pain_points[:3]
        ],
        "what_users_want": [
            {"text": item['theme'].split('–')[0].strip(), "source_indices": item.get('source_indices', [])}
            for item in requests[:3]
        ]
    }

async def get_reddit_posts(query: str, subreddits: List[str], time_filter: str, limit: int) -> List[Dict[str, str]]:
    """Get Reddit posts with URLs - using mock data if Reddit API unavailable"""
    try:
        import asyncpraw
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            reddit = asyncpraw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
            reddit.read_only = True

            posts: List[Dict[str, str]] = []
            for sr in subreddits:
                # Respect a global cap across all subreddits
                if len(posts) >= limit:
                    break
                try:
                    subreddit = await reddit.subreddit(sr)
                    remaining = max(0, limit - len(posts))
                    if remaining <= 0:
                        break
                    async for post in subreddit.search(
                        query,
                        sort="relevance",
                        time_filter=time_filter,
                        limit=remaining,
                    ):
                        if getattr(post, "selftext", "") in ['[removed]', '[deleted]']:
                            continue
                        title = getattr(post, "title", "")
                        selftext = getattr(post, "selftext", "")
                        text = f"{title} {selftext}".strip()
                        permalink = getattr(post, "permalink", "")
                        url = f"https://reddit.com{permalink}" if permalink else "URL unavailable"

                        if len(text) > 10:
                            posts.append({
                                "text": text,
                                "title": title,
                                "url": url
                            })
                            if len(posts) >= limit:
                                break
                except Exception:
                    continue

            # Debug log to see total fetched across all subreddits
            logger.info(f"Fetched {len(posts)} posts total across {subreddits}")

            if posts:
                return posts[:limit]
    except ImportError:
        pass

    # Mock data when Reddit API unavailable
    samples = [
        {"text": f"Great experience with {query}! The new features are amazing and really helpful.", "title": f"Great experience with {query}!", "url": "URL unavailable"},
        {"text": f"Having some issues with {query}, hope they can fix the bugs soon.", "title": f"Having some issues with {query}", "url": "URL unavailable"},
        {"text": f"Mixed feelings about {query}. Some parts are good, others need work.", "title": f"Mixed feelings about {query}", "url": "URL unavailable"},
        {"text": f"Love the {query} updates! Best improvement in years.", "title": f"Love the {query} updates!", "url": "URL unavailable"},
        {"text": f"Disappointed with {query} lately. Performance has been declining.", "title": f"Disappointed with {query} lately", "url": "URL unavailable"}
    ]
    return samples[:limit]

async def perform_sentiment_analysis(query: str, subreddits: List[str], time_filter: str, limit: int, context: str) -> Dict[str, Any]:
    """Main sentiment analysis function with OpenAI primary and Claude fallback"""

    # Create a unique hash for this request to detect duplicates
    import hashlib
    query_params = f"{query}|{','.join(sorted(subreddits))}|{time_filter}|{limit}|{context}"
    query_hash = hashlib.md5(query_params.encode()).hexdigest()

    # Check if this exact request is already being processed
    async with dedup_lock:
        if query_hash in in_progress_analyses:
            logger.info("=" * 60)
            logger.info("DUPLICATE REQUEST DETECTED")
            logger.info(f"Query: {query}, Subreddits: {subreddits}")
            logger.info("Waiting for ongoing analysis to complete...")
            logger.info("=" * 60)
            # Wait for the existing analysis to complete and return its result
            result = await in_progress_analyses[query_hash]
            logger.info("Duplicate request resolved - returning cached result")
            return result

        # Create a new future for this analysis
        future = asyncio.Future()
        in_progress_analyses[query_hash] = future

    logger.info(f"Starting new analysis with hash: {query_hash}")

    try:
        # Get posts
        posts = await get_reddit_posts(query, subreddits, time_filter, limit)

        if not posts:
            return {
                "error": "No posts found",
                "suggestion": "Try different subreddits or search terms",
                "query_info": {"query": query, "subreddits": subreddits}
            }

        # Extract text for AI analysis while keeping full post objects
        post_texts = [p["text"] for p in posts]

        results = []
        analysis_method = "simple"

        # Try OpenAI first (with one retry)
        logger.info("=" * 60)
        logger.info("ANALYSIS ATTEMPT: Starting with OpenAI (primary)")
        logger.info("=" * 60)

        for attempt in range(2):  # Try twice
            if attempt > 0:
                logger.info(f"OpenAI retry attempt {attempt + 1}/2")

            results, analysis_method = await analyze_with_openai(post_texts, context, query)

            if results and analysis_method == "openai":
                logger.info("=" * 60)
                logger.info("SUCCESS: OpenAI analysis completed successfully")
                logger.info(f"Cost tracking: Used GPT-4o-mini for {len(posts)} posts")
                logger.info("=" * 60)
                break
            else:
                logger.warning(f"OpenAI attempt {attempt + 1} failed: {analysis_method}")
                if attempt == 0:
                    logger.info("Retrying OpenAI once before fallback...")

        # If OpenAI failed after retries, fall back to Claude
        if not results or analysis_method != "openai":
            logger.info("=" * 60)
            logger.info("FALLBACK: OpenAI failed, switching to Claude (Anthropic)")
            logger.info(f"Reason for fallback: {analysis_method}")
            logger.info("=" * 60)

            results, analysis_method = await analyze_with_claude(post_texts, context, query)

            if results and analysis_method == "claude_ai":
                logger.info("=" * 60)
                logger.info("SUCCESS: Claude analysis completed successfully")
                logger.info(f"Cost tracking: Used Claude 3.5 Sonnet for {len(posts)} posts")
                logger.info("=" * 60)
            else:
                logger.warning(f"Claude also failed: {analysis_method}")

        # If both AI methods failed, use simple rule-based analysis
        if not results:
            logger.info("=" * 60)
            logger.info("FALLBACK: Both AI methods failed, using simple rule-based analysis")
            logger.info(f"Last error: {analysis_method}")
            logger.info("=" * 60)

            analysis_method = "simple"
            results = [SentimentResult(
                idx,  # post_index
                *analyze_sentiment_simple(post_text), 0.5,
                "Rule-based inference",
                ["general discussion"], ["none identified"], ["none identified"]
            ) for idx, post_text in enumerate(post_texts)]

            logger.info("Simple analysis fallback completed")

        # Aggregate results with post indices
        themes_with_indices: List[tuple[str, int]] = []
        pain_points_with_indices: List[tuple[str, int]] = []
        feature_requests_with_indices: List[tuple[str, int]] = []

        for r in results:
            # Collect themes with their post index
            for theme in (r.key_themes or []):
                themes_with_indices.append((theme, r.post_index))
            # Collect pain points with their post index
            for pain_point in (r.pain_points or []):
                pain_points_with_indices.append((pain_point, r.post_index))
            # Collect feature requests with their post index
            for request in (r.feature_requests or []):
                feature_requests_with_indices.append((request, r.post_index))

        total = len(posts)

        # Calculate ranked insights with counts, percentages, and source indices
        insights_like = calculate_ranked_insights(themes_with_indices, total, limit=10)
        insights_dont_like = calculate_ranked_insights(pain_points_with_indices, total, limit=10)
        insights_wish_existed = calculate_ranked_insights(feature_requests_with_indices, total, limit=10)

        # Create high-level summary bullets
        logger.info("Creating key insights summary...")
        key_insights_summary = await create_key_insights_summary(
            insights_like,
            insights_dont_like,
            insights_wish_existed,
            query
        )

        logger.info("=" * 60)
        logger.info(f"FINAL RESULT: Analysis completed using '{analysis_method}' method")
        logger.info(f"Total insights extracted: {len(themes_with_indices)} themes, {len(pain_points_with_indices)} pain points, {len(feature_requests_with_indices)} requests")
        logger.info(f"Summary bullets: {len(key_insights_summary.get('what_users_like', []))} likes, {len(key_insights_summary.get('major_frustrations', []))} frustrations, {len(key_insights_summary.get('what_users_want', []))} wants")
        logger.info("=" * 60)

        result = {
            "overview": {
                "query": query,
                "total_posts": total,
                "subreddits_searched": subreddits,
                "analysis_method": analysis_method
            },
            "key_insights_summary": key_insights_summary,
            "business_insights": {
                "what_users_like": insights_like,
                "what_users_dont_like": insights_dont_like,
                "what_users_wish_existed": insights_wish_existed
            },
            "recommendation": f"Analysis of {total} posts about {query}. Review insights for product strategy."
        }

        # Set the result in the future for any waiting duplicate requests
        future.set_result(result)
        return result

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        error_result = {
            "error": f"Analysis failed: {str(e)}",
            "query": query,
            "suggestion": "Please try again or contact support"
        }
        # Set the error result in the future
        future.set_result(error_result)
        return error_result
    finally:
        # Clean up the in-progress entry
        async with dedup_lock:
            if query_hash in in_progress_analyses:
                del in_progress_analyses[query_hash]
                logger.info(f"Cleaned up analysis hash: {query_hash}")

# ============================================================================
# FIXED SSE TRANSPORT LAYER FOR COPILOT STUDIO
# ============================================================================

# FastAPI app for SSE transport
app = FastAPI(
    title="Reddit Sentiment MCP Server",
    description="Real MCP Server with FIXED SSE transport for Copilot Studio",
    version="1.0.0"
)

# =============================
# Apps SDK UI helper
# =============================

def build_component_iframe_html() -> str:
    """Return an HTML document that mounts the bundled React component.
    ChatGPT Apps will load this via MCP resources/read when using the
    openai/outputTemplate reference.
    The component reads tool outputs via window.openai bridge.
    """
    bundle_path = os.path.join(os.path.dirname(__file__), "web", "dist", "component.js")
    try:
        with open(bundle_path, "r", encoding="utf-8") as f:
            js = f.read()
        inline = f"""
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Product Insights</title>
    <style>
      :root {{ color-scheme: light dark; }}
      html, body, #root {{ height: 100%; margin: 0; }}
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
    </style>
  </head>
  <body>
    <div id=\"root\"></div>
    <script type=\"module\">{js}</script>
  </body>
</html>
"""
    except Exception as e:
        # Fallback: show a helpful message if the bundle is missing
        logger.warning(f"UI bundle not found at {bundle_path}: {e}")
        inline = f"""
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Product Insights (Bundle missing)</title>
  </head>
  <body>
    <div style=\"padding:16px;font-family:system-ui,sans-serif\">\n
      <h3>Custom UX bundle not built</h3>
      <p>Please build the UI bundle on the server:</p>
      <pre>cd web &amp;&amp; npm install &amp;&amp; npm run build</pre>
      <p>Then re-run the tool. The iframe will load <code>web/dist/component.js</code>.</p>
    </div>
  </body>
</html>
"""
    return inline

# CORS for Copilot Studio (wide-open; tighten if used in browsers with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "reddit-sentiment-mcp-server",
        "version": "1.0.0",
        "transport": "sse-fixed",
        "status": "ready",
        "endpoints": {
            "mcp": "/mcp (FIXED SSE endpoint for Copilot Studio)",
            "health": "/health",
            "message": "/message"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "mcp_server": "ready"}

@app.get("/mcp")
async def mcp_sse_endpoint(request: Request):
    """
    FIXED SSE endpoint for Copilot Studio MCP integration
    Creates session and streams all MCP responses through SSE
    """
    session_id = str(uuid.uuid4())

    # Create message queue for this session
    message_queue: asyncio.Queue = asyncio.Queue()
    with session_lock:
        active_sessions[session_id] = message_queue

    logger.info(f"SSE session created: {session_id}")

    async def event_stream():
        try:
            # Send initial endpoint URL
            base_url = str(request.url).replace("/mcp", "")
            message_endpoint = f"{base_url}/message"
            yield f"event: endpoint\ndata: {message_endpoint}\n\n"

            # Heartbeats and message forwarding
            ping_count = 0
            while True:
                try:
                    message = await asyncio.wait_for(message_queue.get(), timeout=30)
                    yield f"event: message\ndata: {json.dumps(message)}\n\n"
                    logger.info(f"SSE: Sent message to session {session_id}")
                except asyncio.TimeoutError:
                    ping_count += 1
                    # Comment heartbeat keeps intermediaries alive, doesn't confuse clients
                    yield f": ping {ping_count}\n\n"
        except Exception as e:
            logger.error(f"SSE stream error for session {session_id}: {e}")
        finally:
            with session_lock:
                active_sessions.pop(session_id, None)
            logger.info(f"SSE session closed: {session_id}")

    sr = StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no"  # helps prevent proxy buffering (nginx etc.)
        }
    )
    # Set session cookie on the actual StreamingResponse
    sr.set_cookie(
        key="mcp_session",
        value=session_id,
        path="/",
        secure=True,
        httponly=True,
        samesite="none"
    )
    return sr

@app.post("/message")
async def mcp_message_endpoint(request: Request, mcp_session: str = Cookie(None)):
    """
    FIXED Message endpoint - routes all responses to SSE stream
    This is the key fix: responses go to SSE, not HTTP body
    """
    try:
        request_data = await request.json()
        method = request_data.get("method", "unknown")
        request_id = request_data.get("id", "unknown")

        # Accept header or query param fallback for session correlation
        header_session = request.headers.get("x-mcp-session")
        query_session = request.query_params.get("session")
        session_key = header_session or query_session or mcp_session

        logger.info(f"MCP request: method={method}, id={request_id}, session={session_key}")

        # Find the SSE session
        with session_lock:
            message_queue = active_sessions.get(session_key)

        if not message_queue:
            # Fallback: try to find any active session (last resort)
            with session_lock:
                message_queue = next(iter(active_sessions.values()), None)
            if not message_queue:
                logger.error(f"No active SSE session found for request {request_id}")
                return JSONResponse(status_code=400, content={"error": "No active SSE session"})

        # Process the MCP request
        params = request_data.get("params", {})

        try:
            if method == "initialize":
                response_data = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {"listChanged": True},
                            "resources": {"listChanged": True}
                        },
                        "serverInfo": {
                            "name": "reddit-sentiment-analyzer",
                            "version": "1.0.0"
                        }
                    }
                }

            elif method == "tools/list":
                tools = await handle_list_tools()
                response_data = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema,
                                "_meta": {
                                    "openai/outputTemplate": "ui://insights/widget.html"
                                }
                            } for tool in tools
                        ]
                    }
                }

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if not tool_name:
                    response_data = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": "Tool name is required"
                        }
                    }
                else:
                    try:
                        # Increased timeout to handle batch processing of 50 posts
                        content_list = await asyncio.wait_for(
                            handle_call_tool(tool_name, arguments),
                            timeout=180.0
                        )
                        # Try to derive structuredContent from the first text item (JSON payload)
                        structured = None
                        try:
                            if content_list and getattr(content_list[0], "text", None):
                                structured = json.loads(content_list[0].text)
                        except Exception:
                            structured = None

                        response_data = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [
                                    {"type": item.type, "text": item.text}
                                    for item in content_list
                                ],
                                # ChatGPT Apps extension: feed our Custom UX via structuredContent
                                **({"structuredContent": structured} if structured is not None else {})
                            }
                        }
                    except Exception as tool_error:
                        logger.error(f"Tool execution error: {tool_error}")
                        response_data = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32603,
                                "message": f"Tool execution failed: {str(tool_error)}"
                            }
                        }

            elif method == "resources/list":
                # Expose our UI template as an MCP resource for ChatGPT Apps to fetch
                response_data = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "resources": [
                            {
                                "uri": "ui://insights/widget.html",
                                "name": "Product Insights UI",
                                "mimeType": "text/html"
                            }
                        ]
                    }
                }

            elif method == "resources/read":
                # Serve the HTML template that inlines the bundled component JS
                uri = params.get("uri") if isinstance(params, dict) else None
                if uri != "ui://insights/widget.html":
                    response_data = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32602, "message": f"Unknown resource: {uri}"}
                    }
                else:
                    html = build_component_iframe_html()
                    response_data = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "contents": [
                                {
                                    "uri": uri,
                                    "mimeType": "text/html",
                                    "text": html
                                }
                            ]
                        }
                    }

            else:
                response_data = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }

            # Route response to SSE stream
            await message_queue.put(response_data)
            logger.info(f"Queued response for SSE stream: method={method}, id={request_id}")

            # Return minimal HTTP acknowledgment
            return JSONResponse(status_code=202, content={"accepted": True, "id": request_id})

        except Exception as method_error:
            logger.error(f"Method handling error for {method}: {method_error}")
            error_response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(method_error)}"
                }
            }
            await message_queue.put(error_response)
            return JSONResponse(status_code=202, content={"accepted": True, "id": request_id})

    except json.JSONDecodeError:
        logger.error("Invalid JSON in request")
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
    except Exception as e:
        logger.error(f"Unexpected error in message endpoint: {e}")
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})

@app.post("/mcp")
async def mcp_streamable_endpoint(request: Request):
    """
    Streamable HTTP transport endpoint for Power Platform / Copilot Studio / Dust.
    Internally reuses the existing /message MCP logic so all tools and methods stay in sync.
    """
    if DUST_MCP_BEARER_TOKEN:
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {DUST_MCP_BEARER_TOKEN}":
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    try:
        # Parse incoming JSON-RPC request
        request_data = await request.json()
        method = request_data.get("method", "unknown")
        request_id = request_data.get("id", "unknown")

        logger.info(f"[Streamable] MCP request: method={method}, id={request_id}")

        # Reuse existing /message handling logic directly
        # We'll call the same code paths without SSE queuing
        params = request_data.get("params", {})

        if method == "initialize":
            response_data = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}, "resources": {"listChanged": True}},
                    "serverInfo": {
                        "name": "reddit-sentiment-analyzer",
                        "version": "1.0.0"
                    }
                }
            }

        elif method == "tools/list":
            tools = await handle_list_tools()
            response_data = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.inputSchema,
                            "_meta": {"openai/outputTemplate": "ui://insights/widget.html"}
                        } for t in tools
                    ]
                }
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            # Increased timeout to handle batch processing of 50 posts (5 batches × ~30s each = ~150s)
            content_list = await asyncio.wait_for(
                handle_call_tool(tool_name, arguments),
                timeout=180.0
            )
            # Derive structuredContent from first text item if JSON
            structured = None
            try:
                if content_list and getattr(content_list[0], "text", None):
                    structured = json.loads(content_list[0].text)
            except Exception:
                structured = None
            response_data = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {"type": item.type, "text": item.text}
                        for item in content_list
                    ],
                    **({"structuredContent": structured} if structured is not None else {})
                }
            }

        elif method == "resources/list":
            response_data = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "resources": [
                        {
                            "uri": "ui://insights/widget.html",
                            "name": "Product Insights UI",
                            "mimeType": "text/html"
                        }
                    ]
                }
            }

        elif method == "resources/read":
            uri = params.get("uri") if isinstance(params, dict) else None
            if uri != "ui://insights/widget.html":
                response_data = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32602, "message": f"Unknown resource: {uri}"}
                }
            else:
                html = build_component_iframe_html()
                response_data = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contents": [
                            {"uri": uri, "mimeType": "text/html", "text": html}
                        ]
                    }
                }

        else:
            response_data = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.exception("Error in /mcp streamable endpoint")
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": request_data.get("id", None),
                "error": {"code": -32603, "message": str(e)}
            }
        )



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    # Check if port is already in use
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    if result == 0:
        print(f"\nERROR: Port {port} is already in use!")
        print("Kill existing processes with: taskkill //F //PID <pid>")
        print("Find PIDs with: netstat -ano | findstr :8000")
        print(f"Or use a different port with: PORT=8001 python reddit_sentiment_server.py")
        sock.close()
        sys.exit(1)
    sock.close()

    print("Starting FIXED MCP Server for Copilot Studio")
    print(f"Server: http://0.0.0.0:{port}")
    print(f"FIXED SSE Endpoint: http://0.0.0.0:{port}/mcp")
    print(f"Message Endpoint: http://0.0.0.0:{port}/message")
    print(f"Health Check: http://0.0.0.0:{port}/health")
    print()
    print("KEY FIX: All MCP responses now route through SSE stream!")
    print("MCP Server Details:")
    print("- Uses official MCP SDK (mcp.server)")
    print("- Implements real MCP protocol")
    print("- FIXED SSE transport for Copilot Studio")
    print("- Routes all responses through SSE (not HTTP body)")
    print()
    print("AI Analysis Configuration:")
    print("- Primary: OpenAI GPT-4o-mini (with 1 retry)")
    print("- Fallback: Anthropic Claude 3.5 Sonnet")
    print("- Last Resort: Simple rule-based analysis")
    print()

    if not REDDIT_CLIENT_ID:
        print("WARNING: Using mock data (set REDDIT_CLIENT_ID for real data)")
    if not OPENAI_API_KEY:
        print("WARNING: OpenAI not configured - will use Claude as primary")
    if not ANTHROPIC_API_KEY:
        print("WARNING: Claude not configured - will use simple analysis if OpenAI fails")
    if OPENAI_API_KEY and ANTHROPIC_API_KEY:
        print("SUCCESS: Both AI providers configured - full fallback chain active")

    # Configure uvicorn logging to show application logs
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(levelprefix)s %(message)s"
    log_config["formatters"]["access"]["fmt"] = '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        log_config=log_config,
        access_log=True
    )
