import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";

export const runtime = "nodejs";

function buildPrompt(product: string, insights: Record<string, unknown>): string {
  const summary = (insights.key_insights_summary ?? insights) as Record<
    string,
    Array<{ text: string }>
  >;

  const fmt = (items: Array<{ text: string }> = []) =>
    items
      .slice(0, 5)
      .map((i) => `- ${i.text}`)
      .join("\n") || "None identified";

  return `You are a Customer Engineer preparing a customer insights card for ${product}.

Here is Reddit sentiment data:

What users like:
${fmt(summary.what_users_like)}

Pain points:
${fmt(summary.major_frustrations)}

Feature requests:
${fmt(summary.what_users_want)}

Write a concise customer insights card with exactly these four sections using markdown:

## What Customers Love
3-4 bullet points

## Key Pain Points
3-4 bullet points

## Top Feature Requests
3-4 bullet points

## CE Takeaway
One actionable sentence a Customer Engineer would say before a call with this customer.`;
}

export async function POST(req: Request) {
  const { product, rawInsights } = await req.json();

  const result = streamText({
    model: openai("gpt-4o-mini"),
    prompt: buildPrompt(product, rawInsights),
  });

  return result.toDataStreamResponse();
}
