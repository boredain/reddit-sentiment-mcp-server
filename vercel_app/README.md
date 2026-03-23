# Customer Intelligence Engine

A conversational AI agent that answers natural language questions about products and companies using real Reddit sentiment data.

Built with the Vercel AI SDK and a Reddit sentiment MCP server.

## What it does

Ask any product question in natural language:

- "What do people think about Cursor?"
- "How does it compare to Codex?"
- "Prepare an executive summary comparing all three tools"

The agent autonomously decides when to fetch Reddit data, calls the MCP server as a tool, and streams a structured response back in real time. It maintains conversation context across turns — if you ask for a comparison after already fetching one product, it reuses the existing data without redundant calls.

## Tech stack

- **Next.js 15** — App Router, Route Handlers
- **Vercel AI SDK** — `streamText` with tool calling, `useChat` for streaming chat UI
- **Reddit Sentiment MCP Server** — called as an AI tool via JSON-RPC over HTTP

## How it works

```
User message
      ↓
useChat → POST /api/chat
      ↓
streamText (gpt-4o-mini) + tool: getRedditInsights
      ↓
LLM autonomously calls MCP server when it needs Reddit data
      ↓
MCP server fetches and analyzes 50 Reddit posts
      ↓
LLM synthesizes results and streams response
      ↓
useChat renders each chunk as it arrives
```

The key technical detail: the LLM decides when and what to query — not the user. `maxSteps: 5` allows multiple tool calls in a single conversation turn, enabling multi-product comparisons in one response.

## Setup

**Prerequisites:**
- Node.js 18+
- A running Reddit Sentiment MCP server exposed via ngrok
- OpenAI API key

**Install dependencies:**
```bash
cd vercel_app
npm install
```

**Configure environment variables:**

Create a `.env.local` file in the `vercel_app/` directory:
```
OPENAI_API_KEY=your_openai_api_key
MCP_SERVER_URL=https://your-ngrok-url.ngrok.io
```

**Run locally:**
```bash
npm run dev
```

Open `http://localhost:3000`

## MCP Server

This app calls the Reddit Sentiment MCP server via its Streamable HTTP transport (`POST /mcp`). The server must be running and accessible via the `MCP_SERVER_URL` environment variable before starting the Next.js app.

See the root `README.md` for MCP server setup instructions.
