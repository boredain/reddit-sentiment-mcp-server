import { openai } from "@ai-sdk/openai";
import { streamText, tool } from "ai";
import { z } from "zod";

export const runtime = "nodejs";

async function callMCPServer(query: string) {
  const mcpUrl = process.env.MCP_SERVER_URL?.replace(/\/$/, "");

  if (!mcpUrl) throw new Error("MCP_SERVER_URL is not set in .env.local");

  const res = await fetch(`${mcpUrl}/mcp`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "tools/call",
      params: {
        name: "analyze_reddit_sentiment",
        arguments: { query },
      },
    }),
  });

  if (!res.ok) throw new Error(`MCP server returned ${res.status}`);

  const data = await res.json();

  if (data.error) throw new Error(data.error.message ?? "MCP error");

  return (
    data.result?.structuredContent ??
    JSON.parse(data.result?.content?.[0]?.text ?? "{}")
  );
}

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: openai("gpt-4o-mini"),
    system: `You are a Customer Intelligence Engine. When asked about a product or company,
use the getRedditInsights tool to fetch real Reddit sentiment data.
Always format your response using exactly these section headers:
## What Customers Love
## Key Pain Points
## Top Feature Requests
Use "- " bullet points under each section. Be concise and professional.`,
    messages,
    tools: {
      getRedditInsights: tool({
        description:
          "Fetch real Reddit sentiment data for a product or company. Use this whenever the user asks about a product, brand, or tool.",
        parameters: z.object({
          query: z
            .string()
            .describe("The product or company name to analyze on Reddit"),
        }),
        execute: async ({ query }) => {
          const insights = await callMCPServer(query);
          return insights.key_insights_summary ?? insights;
        },
      }),
    },
    maxSteps: 5,
  });

  return result.toDataStreamResponse();
}
