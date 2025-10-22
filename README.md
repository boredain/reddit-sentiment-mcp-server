# Reddit Sentiment Analysis MCP Server

An enterprise-grade Reddit sentiment analysis server with dual AI providers (OpenAI & Claude) and Model Context Protocol (MCP) integration for ChatGPT Apps and other AI platforms.

## 🚀 Key Features

- **Dual AI Analysis**: OpenAI GPT-4o-mini (primary) with Claude 3.5 Sonnet fallback
- **ChatGPT Apps Integration**: Custom UI widget with interactive visualizations
- **Smart Categorization**: AI-powered insight extraction with validation logic
- **MCP Protocol**: Full support for ChatGPT and Copilot Studio integration
- **SSE Streaming**: Real-time responses via Server-Sent Events
- **Deduplication**: Intelligent request caching to prevent duplicate API calls
- **50 Posts Analysis**: Fixed batch processing across multiple subreddits
- **Custom UI Components**: React-based visualization with source tracking

## 🆕 Recent Improvements (Latest)

- ✅ **Optimized AI Prompts**: 55% reduction in prompt length while maintaining accuracy
- ✅ **Fixed Categorization**: Negative keywords filter prevents misclassification
- ✅ **Timeout Fix**: Reduced processing time from ~175s to ~100-125s
- ✅ **Time Filter Default**: Changed from "week" to "all" for comprehensive analysis
- ✅ **Query Capitalization**: Auto-capitalize first letter in UI headings
- ✅ **Validation Logic**: Built-in sentiment misclassification detection (available but disabled)

## 📋 Core Capabilities

### Analysis Features
- **Multi-subreddit Search**: Analyze up to 50 posts across multiple subreddits
- **Sentiment Categorization**:
  - **Themes**: What users like (positive sentiment only)
  - **Pain Points**: What users complain about (negative sentiment)
  - **Feature Requests**: What users want (actionable improvements)
- **Source Tracking**: Every insight links back to original Reddit posts
- **Bubble Chart Visualization**: Count and percentage for each insight

### Technical Features
- **Dual AI Providers**: OpenAI primary, Claude fallback, simple rule-based last resort
- **Request Deduplication**: Prevents duplicate analysis via hash-based caching
- **Batch Processing**: 5 batches of 10 posts each with alignment validation
- **SSE Transport**: Server-Sent Events for ChatGPT Apps compatibility
- **Streamable HTTP**: Alternative transport for Copilot Studio

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Reddit API credentials
- OpenAI API key (primary)
- Anthropic API key (optional fallback)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/reddit-sentiment-mcp-server.git
cd reddit-sentiment-mcp-server
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
Create a `.env` file in the project root:
```env
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=RedditSentimentBot/1.0
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
```

5. **Build the UI component** (for ChatGPT Apps)
```bash
cd web
npm install
npm run build
cd ..
```

6. **Run the server**
```bash
python reddit_sentiment_server.py
```

Server will start on `http://0.0.0.0:8000` (or custom PORT environment variable)

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `REDDIT_CLIENT_ID` | Yes | Reddit application client ID |
| `REDDIT_CLIENT_SECRET` | Yes | Reddit application secret |
| `REDDIT_USER_AGENT` | Yes | User agent string (format: AppName/Version) |
| `OPENAI_API_KEY` | Recommended | OpenAI API key (primary AI provider) |
| `ANTHROPIC_API_KEY` | Optional | Anthropic API key (fallback provider) |
| `PORT` | Optional | Server port (default: 8000) |

### Getting API Credentials

**Reddit API:**
1. Visit [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Click "Create App" → Select "script"
3. Copy the client ID and secret

**OpenAI API:**
1. Visit [OpenAI Platform](https://platform.openai.com)
2. Go to API Keys section
3. Create new secret key

**Anthropic API:**
1. Visit [Anthropic Console](https://console.anthropic.com)
2. Generate API key
3. Add credits for usage

## 📖 Usage

### MCP Tool: `analyze_reddit_sentiment`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | Required | Search term (e.g., "Spotify", "Instacart") |
| `subreddits` | array | ["all"] | List of subreddit names |
| `time_filter` | string | "all" | Time period: hour, day, week, month, year, all |
| `limit` | integer | 50 | Fixed to 50 posts across all subreddits |
| `product_context` | string | "" | Additional business context |

**Example ChatGPT Prompt:**
```
Analyze Reddit sentiment for "Uber Eats" in subreddits InstacartShoppers, gigwork, and frugal
```

**Example Response Structure:**
```json
{
  "overview": {
    "query": "Uber Eats",
    "total_posts": 50,
    "subreddits_searched": ["InstacartShoppers", "gigwork", "frugal"],
    "analysis_method": "openai"
  },
  "posts": ["post1 text", "post2 text", ...],
  "posts_with_urls": [
    {"title": "First delivery", "url": "https://reddit.com/..."},
    ...
  ],
  "business_insights": {
    "what_users_like": [
      {
        "theme": "Promotional offers – Users appreciate discounts...",
        "count": 5,
        "percentage": 10.0,
        "source_indices": [12, 23, 34]
      }
    ],
    "what_users_dont_like": [...],
    "what_users_wish_existed": [...]
  },
  "key_insights_summary": {
    "what_users_like": [
      {
        "text": "Promotional offers: Discounts for new accounts",
        "source_indices": [12, 23, 34]
      }
    ],
    "major_frustrations": [...],
    "what_users_want": [...]
  }
}
```

## 🏗️ Architecture

### System Components

1. **MCP Server** (`reddit_sentiment_server.py`)
   - FastAPI application with MCP protocol support
   - SSE and HTTP transport layers
   - Tool handler: `analyze_reddit_sentiment`

2. **AI Analysis Pipeline**
   - **Primary**: OpenAI GPT-4o-mini (5 batches × 10 posts)
   - **Fallback**: Claude 3.5 Sonnet
   - **Last Resort**: Simple rule-based analysis
   - **Validation**: Sentiment misclassification detection (optional)

3. **Prompt System** (`prompts/`)
   - `analysis_prompt.txt`: Extract themes, pain points, requests
   - `summary_prompt.txt`: Generate concise key insights

4. **Web UI Component** (`web/`)
   - React-based visualization
   - Bubble charts with post counts
   - Source drawer with Reddit links
   - ChatGPT Apps SDK integration

### Data Flow

```
ChatGPT Request → MCP Server → Reddit API (fetch 50 posts)
                              ↓
                     OpenAI GPT-4o-mini (batch analysis)
                              ↓
                     Validation Logic (optional)
                              ↓
                     Summary Generation (OpenAI)
                              ↓
                     Custom UI Rendering (ChatGPT Apps)
```

### AI Provider Fallback Chain

```
OpenAI (2 retries) → Claude → Simple Rule-Based
```

## 🎨 Custom UI (ChatGPT Apps)

The server provides a custom React-based UI when used with ChatGPT Apps:

**Features:**
- **Three-column layout**: What Users Like, Don't Like, Want
- **Interactive chips**: Click to view source posts
- **Source drawer**: Direct Reddit links for verification
- **Responsive design**: Works on mobile and desktop
- **Post count indicators**: Visual feedback on data volume

**Build UI:**
```bash
cd web
npm run build
```

Output: `web/dist/component.js` (auto-served by MCP server)

## 🔒 Security & Best Practices

- **API Keys**: Store in `.env` file, never commit to git
- **Port Checking**: Server validates port availability before starting
- **Timeout Protection**: 180s timeout for tool execution
- **Error Handling**: Graceful degradation with fallback providers
- **Request Deduplication**: Prevents duplicate API costs

## 🐛 Troubleshooting

### Server won't start (port in use)
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <pid>

# Mac/Linux
lsof -i :8000
kill -9 <pid>
```

### Analysis timeout errors
- **Cause**: ChatGPT client timeout < server processing time
- **Fix**: Optimized prompts reduce time from ~175s to ~100-125s
- **Status**: Fixed in latest version

### Misclassified insights
- **Example**: Negative sentiment in "What Users Like"
- **Fix**: Improved prompts with negative keyword filter
- **Status**: Fixed in latest version

## 📊 Performance Metrics

- **Posts per analysis**: 50 (fixed)
- **Processing time**: ~100-125 seconds (optimized)
- **Batch size**: 10 posts per batch
- **Total batches**: 5
- **API calls per analysis**: 6 (5 batches + 1 summary)
- **Success rate**: 98%+ with dual fallback

## 🚀 Deployment

### Local Development
```bash
python reddit_sentiment_server.py
```

### Production (Render/Railway/Heroku)
```bash
# Set environment variables in platform dashboard
# Deploy via git push or platform CLI
```

### Docker (Optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "reddit_sentiment_server.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

MIT License - See LICENSE file for details

## 📞 Support

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/reddit-sentiment-mcp-server/issues)
- **Email**: abhishek.singh@vanderbilt.edu
- **LinkedIn**: [Abhishek Singh](https://www.linkedin.com/in/abhishek-singh-nitjsr/)

## 🙏 Acknowledgments

- **OpenAI**: GPT-4o-mini for primary analysis
- **Anthropic**: Claude 3.5 Sonnet for fallback
- **Reddit**: API access for sentiment data
- **MCP Protocol**: Standard for AI tool integration
- **ChatGPT Apps SDK**: Custom UI framework

---

**Built for enterprise AI applications with production-grade reliability**
