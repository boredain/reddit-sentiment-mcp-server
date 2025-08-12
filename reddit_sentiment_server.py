#!/usr/bin/env python3
"""
FIXED MCP Server for Copilot Studio
Routes all JSON-RPC responses through SSE stream (not HTTP response body)
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import httpx
import queue
import threading

# Official MCP imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
import mcp.types as types

# FastAPI for SSE transport wrapper
from fastapi import FastAPI, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "RedditSentimentBot/1.0")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Create the REAL MCP server instance
mcp_server = Server("reddit-sentiment-analyzer")

# CRITICAL: Session management for SSE streams
active_sessions: Dict[str, asyncio.Queue] = {}
session_lock = threading.Lock()

@dataclass
class SentimentResult:
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
                        "default": "week"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum posts per subreddit",
                        "default": 5
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
        time_filter = arguments.get("time_filter", "week")
        limit = max(1, min(arguments.get("limit", 5), 10))
        product_context = arguments.get("product_context", "")
        
        # Perform sentiment analysis
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

async def analyze_with_claude(posts: List[str], context: str) -> List[SentimentResult]:
    """Analyze sentiment using Claude AI"""
    if not ANTHROPIC_API_KEY:
        logger.info("Using simple analysis (no Claude API key)")
        return [SentimentResult(
            *analyze_sentiment_simple(post), 0.5,
            ["general discussion"], ["none identified"], ["none identified"]
        ) for post in posts]
    
    prompt = f"""Analyze sentiment for these Reddit posts about: {context or "general topic"}
    
    Return JSON array with this exact structure for each post:
    [{{"score": 0.5, "label": "positive|negative|neutral", "confidence": 0.8, "reasoning": "brief explanation", "themes": ["theme1"], "pain_points": ["issue1"], "requests": ["request1"]}}]
    
    Posts: {json.dumps(posts[:5])}"""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 800,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            if response.status_code == 200:
                text = response.json()["content"][0]["text"]
                try:
                    json_start = text.find("[")
                    json_end = text.rfind("]") + 1
                    if json_start != -1 and json_end > json_start:
                        parsed = json.loads(text[json_start:json_end])
                        return [SentimentResult(
                            p.get("score", 0.0),
                            p.get("label", "neutral"),
                            p.get("confidence", 0.5),
                            p.get("reasoning", ""),
                            p.get("themes", []),
                            p.get("pain_points", []),
                            p.get("requests", [])
                        ) for p in parsed]
                except:
                    pass
    except Exception as e:
        logger.warning(f"Claude API error: {e}")
    
    # Fallback to simple analysis
    return [SentimentResult(
        *analyze_sentiment_simple(post), 0.5,
        ["general feedback"], ["none identified"], ["none identified"]
    ) for post in posts]

async def get_reddit_posts(query: str, subreddits: List[str], time_filter: str, limit: int) -> List[str]:
    """Get Reddit posts - using mock data for demo"""
    try:
        import asyncpraw
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            reddit = asyncpraw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
            reddit.read_only = True
            
            posts = []
            for sr in subreddits:
                try:
                    subreddit = await reddit.subreddit(sr)
                    count = 0
                    async for post in subreddit.search(query, sort="relevance", time_filter=time_filter, limit=limit):
                        if count >= limit or post.selftext in ['[removed]', '[deleted]']:
                            continue
                        text = f"{post.title} {post.selftext}".strip()
                        if len(text) > 10:
                            posts.append(text)
                            count += 1
                except:
                    continue
            
            if posts:
                return posts
    except ImportError:
        pass
    
    # Mock data when Reddit API unavailable
    return [
        f"Great experience with {query}! The new features are amazing and really helpful.",
        f"Having some issues with {query}, hope they can fix the bugs soon.",
        f"Mixed feelings about {query}. Some parts are good, others need work.",
        f"Love the {query} updates! Best improvement in years.",
        f"Disappointed with {query} lately. Performance has been declining."
    ][:limit]

async def perform_sentiment_analysis(query: str, subreddits: List[str], time_filter: str, limit: int, context: str) -> Dict:
    """Main sentiment analysis function"""
    try:
        # Get posts
        posts = await get_reddit_posts(query, subreddits, time_filter, limit)
        
        if not posts:
            return {
                "error": "No posts found",
                "suggestion": "Try different subreddits or search terms",
                "query_info": {"query": query, "subreddits": subreddits}
            }
        
        # Analyze sentiment
        results = await analyze_with_claude(posts, context)
        
        # Aggregate results
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_score = 0
        all_themes = []
        all_pain_points = []
        all_requests = []
        
        for result in results:
            label = result.sentiment_label.lower()
            if label in sentiment_counts:
                sentiment_counts[label] += 1
            total_score += result.sentiment_score
            all_themes.extend(result.key_themes)
            all_pain_points.extend(result.pain_points)
            all_requests.extend(result.feature_requests)
        
        return {
            "overview": {
                "query": query,
                "total_posts": len(posts),
                "subreddits_searched": subreddits,
                "analysis_method": "claude_ai" if ANTHROPIC_API_KEY else "simple"
            },
            "sentiment_summary": {
                "distribution": sentiment_counts,
                "percentages": {k: f"{(v/len(posts)*100):.1f}%" for k, v in sentiment_counts.items()},
                "average_score": round(total_score / len(results), 2),
                "overall_sentiment": max(sentiment_counts, key=sentiment_counts.get)
            },
            "business_insights": {
                "key_themes": list(set(all_themes))[:5],
                "pain_points": list(set(all_pain_points))[:5],
                "feature_requests": list(set(all_requests))[:5]
            },
            "recommendation": f"Analysis of {len(posts)} posts shows {max(sentiment_counts, key=sentiment_counts.get)} sentiment. Use insights for product strategy."
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "query": query,
            "suggestion": "Please try again or contact support"
        }

# ============================================================================
# FIXED SSE TRANSPORT LAYER FOR COPILOT STUDIO
# ============================================================================

# FastAPI app for SSE transport
app = FastAPI(
    title="Reddit Sentiment MCP Server",
    description="Real MCP Server with FIXED SSE transport for Copilot Studio",
    version="1.0.0"
)

# CORS for Copilot Studio
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
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "mcp_server": "ready"}

@app.get("/mcp")
async def mcp_sse_endpoint(request: Request, response: Response):
    """
    FIXED SSE endpoint for Copilot Studio MCP integration
    Creates session and streams all MCP responses through SSE
    """
    session_id = str(uuid.uuid4())
    
    # Set session cookie for POST correlation
    response.set_cookie(
        key="mcp_session",
        value=session_id,
        path="/",
        secure=True,
        httponly=True,
        samesite="none"
    )
    
    # Create message queue for this session
    message_queue = asyncio.Queue()
    with session_lock:
        active_sessions[session_id] = message_queue
    
    logger.info(f"SSE session created: {session_id}")
    
    async def event_stream():
        try:
            # Send initial endpoint URL
            base_url = str(request.url).replace("/mcp", "")
            message_endpoint = f"{base_url}/message"
            yield f"event: endpoint\ndata: {message_endpoint}\n\n"
            
            # Send periodic pings and process messages
            ping_count = 0
            while True:
                try:
                    # Check for queued messages (with timeout)
                    message = await asyncio.wait_for(message_queue.get(), timeout=30)
                    yield f"event: message\ndata: {json.dumps(message)}\n\n"
                    logger.info(f"SSE: Sent message to session {session_id}")
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    ping_count += 1
                    yield f"event: ping\ndata: {json.dumps({'jsonrpc': '2.0', 'id': f'ping-{ping_count}', 'method': 'ping'})}\n\n"
                    
        except Exception as e:
            logger.error(f"SSE stream error for session {session_id}: {e}")
        finally:
            # Clean up session
            with session_lock:
                active_sessions.pop(session_id, None)
            logger.info(f"SSE session closed: {session_id}")
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

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
        
        logger.info(f"MCP request: method={method}, id={request_id}, session={mcp_session}")
        
        # Find the SSE session
        message_queue = None
        if mcp_session:
            with session_lock:
                message_queue = active_sessions.get(mcp_session)
        
        if not message_queue:
            # Fallback: try to find any active session
            with session_lock:
                if active_sessions:
                    message_queue = next(iter(active_sessions.values()))
                    logger.warning(f"Using fallback session for request {request_id}")
        
        if not message_queue:
            logger.error(f"No active SSE session found for request {request_id}")
            return JSONResponse(
                status_code=400,
                content={"error": "No active SSE session"}
            )
        
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
                            "tools": {"listChanged": True}
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
                                "inputSchema": tool.inputSchema
                            } for tool in tools
                        ]
                    }
                }
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if not tool_name:
                    raise ValueError("Tool name is required")
                
                try:
                    content_list = await asyncio.wait_for(
                        handle_call_tool(tool_name, arguments),
                        timeout=25.0
                    )
                    response_data = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {"type": item.type, "text": item.text}
                                for item in content_list
                            ]
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
            
            else:
                response_data = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            
            # CRITICAL FIX: Send response to SSE stream instead of HTTP response
            await message_queue.put(response_data)
            logger.info(f"Queued response for SSE stream: method={method}, id={request_id}")
            
            # Return minimal HTTP acknowledgment
            return JSONResponse(content={"accepted": True, "id": request_id})
            
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
            return JSONResponse(content={"accepted": True, "id": request_id})
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON"}
        )
    except Exception as e:
        logger.error(f"Unexpected error in message endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    print("🚀 Starting FIXED MCP Server for Copilot Studio")
    print(f"📍 Server: http://0.0.0.0:{port}")
    print(f"🔗 FIXED SSE Endpoint: http://0.0.0.0:{port}/mcp")
    print(f"💬 Message Endpoint: http://0.0.0.0:{port}/message")
    print(f"❤️ Health Check: http://0.0.0.0:{port}/health")
    print()
    print("🔧 KEY FIX: All MCP responses now route through SSE stream!")
    print("📋 MCP Server Details:")
    print("• Uses official MCP SDK (mcp.server)")
    print("• Implements real MCP protocol")
    print("• FIXED SSE transport for Copilot Studio")
    print("• Routes all responses through SSE (not HTTP body)")
    
    if not REDDIT_CLIENT_ID:
        print("⚠️ Using mock data (set REDDIT_CLIENT_ID for real data)")
    if not ANTHROPIC_API_KEY:
        print("⚠️ Using simple analysis (set ANTHROPIC_API_KEY for Claude)")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
