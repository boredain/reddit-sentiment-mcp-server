# Reddit Sentiment Analysis MCP Server

An enterprise-grade Reddit sentiment analysis server built with Claude AI and Model Context Protocol (MCP) that provides advanced sentiment analysis capabilities for business intelligence and product research.

## 🚀 Business Value

- **Reduce research time by 60-80%** for strategic initiatives
- **Real-time sentiment tracking** across multiple subreddits
- **Advanced AI-powered analysis** with Claude's reasoning capabilities
- **Enterprise-ready** with compliance tracking and audit logging
- **Business-focused insights** including pain points, feature requests, and urgency levels

## 📋 Features

### Core Capabilities
- **Multi-subreddit analysis** with configurable time filters
- **Dual analysis modes**: Claude AI (advanced) and rule-based (fallback)
- **Comprehensive sentiment metrics** including confidence scores and reasoning
- **Business intelligence extraction**: pain points, feature requests, themes
- **Risk assessment**: urgency levels and business impact analysis
- **Batch processing** for efficient API usage

### Technical Highlights
- **Async/await architecture** for high performance
- **Timeout protection** and error handling
- **Flexible query parameters** with sensible defaults
- **Structured JSON responses** for easy integration
- **MCP server implementation** for Claude integration

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Reddit API credentials
- Anthropic API key

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/reddit-sentiment-mcp-server.git
cd reddit-sentiment-mcp-server
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure credentials**
```bash
cp .env.example .env
# Edit .env with your API credentials
```

4. **Run the server**
```bash
python reddit_sentiment_server.py
```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=YourBotName/1.0 by YourUsername
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Getting API Credentials

**Reddit API:**
1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Choose "script" for personal use
4. Note the client ID and secret

**Anthropic API:**
1. Visit [Anthropic Console](https://console.anthropic.com)
2. Create an account and generate an API key
3. Add credits to your account for usage

## 📖 Usage

### Basic Analysis
```python
# Analyze sentiment for "iPhone 15" in technology subreddits
result = await safe_analyze_reddit_sentiment(
    query="iPhone 15",
    subreddits=["technology", "apple", "iphone"],
    time_filter="week",
    limit=10
)
```

### Advanced Business Analysis
```python
# Product-focused analysis with business context
result = await safe_analyze_reddit_sentiment(
    query="customer service experience",
    subreddits=["CustomerService", "smallbusiness"],
    product_context="SaaS customer support platform",
    use_claude=True,
    return_full_data=True
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | Required | Search term to analyze |
| `subreddits` | List[str] | ["all"] | Target subreddit names |
| `time_filter` | str | "week" | Time period (hour, day, week, month, year, all) |
| `limit` | int | 5 | Maximum posts per subreddit |
| `use_claude` | bool | True | Enable Claude AI analysis |
| `product_context` | str | "" | Business context for analysis |
| `return_full_data` | bool | False | Include detailed post-level data |

## 📊 Response Format

### Summary Response (default)
```json
{
  "overview": {
    "total_posts_analyzed": 25,
    "analysis_method": "claude_ai"
  },
  "sentiment_breakdown": {
    "distribution": {"positive": 15, "negative": 5, "neutral": 5},
    "percentages": {"positive": "60.0%", "negative": "20.0%", "neutral": "20.0%"},
    "overall_sentiment": "positive"
  },
  "key_insights": {
    "top_themes": ["performance", "design", "price"],
    "main_pain_points": ["battery life", "price point"],
    "top_feature_requests": ["better camera", "longer battery"]
  },
  "business_metrics": {
    "urgent_issues": 2,
    "high_impact_items": 5
  }
}
```

## 🎯 Business Applications

### Product Management
- **Feature prioritization** based on user feedback volume and sentiment
- **Competitive analysis** through brand mention tracking
- **Release impact assessment** by monitoring sentiment changes

### Customer Success
- **Proactive issue identification** through pain point analysis
- **Support ticket prevention** via early warning systems
- **Customer satisfaction trending** across product updates

### Marketing Intelligence
- **Brand perception monitoring** across relevant communities
- **Campaign effectiveness measurement** through sentiment shifts
- **Influencer identification** via high-impact feedback detection

## 🏗️ Architecture

### Technical Stack
- **FastMCP**: Model Context Protocol server framework
- **AsyncPRAW**: Asynchronous Reddit API wrapper
- **Anthropic API**: Claude AI integration for advanced analysis
- **HTTPX**: Modern async HTTP client
- **Python AsyncIO**: Concurrent processing

### Design Patterns
- **Async/await**: Non-blocking I/O operations
- **Timeout handling**: Graceful degradation for long-running operations
- **Fallback analysis**: Rule-based backup when Claude API is unavailable
- **Batch processing**: Efficient API usage through request consolidation

## 🔒 Security & Compliance

- **API key protection** through environment variables
- **Rate limiting** with timeout controls
- **Data privacy** with configurable data retention
- **Audit logging** for enterprise compliance requirements

## 🚧 Roadmap

- [ ] **Real-time streaming** for live sentiment monitoring
- [ ] **Historical trend analysis** with time-series data
- [ ] **Multi-language support** for international markets
- [ ] **Advanced filtering** by user demographics and karma
- [ ] **Integration webhooks** for business system connectivity
- [ ] **Dashboard interface** for non-technical stakeholders

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or support:
- Create an issue in this repository
- Email: abhishek.singh@vanderbilt.edu
- LinkedIn: https://www.linkedin.com/in/abhishek-singh-nitjsr/

## 🙏 Acknowledgments

- **Anthropic** for Claude AI capabilities
- **Reddit API** for data access
- **MCP Community** for protocol standards
- **AsyncPRAW** for Python Reddit integration

---

**Built with ❤️ for enterprise AI applications**
