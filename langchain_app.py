import asyncio
import json
from typing import TypedDict, Optional

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

nest_asyncio.apply()
load_dotenv()


# ── State ──────────────────────────────────────────────────────────────────────

class InsightsState(TypedDict):
    ngrok_url: str
    product: str
    raw_insights: Optional[dict]
    summary: Optional[str]


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_tool_result(result) -> dict:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw": result}
    if isinstance(result, list) and result:
        first = result[0]
        if hasattr(first, "text"):
            return json.loads(first.text)
        if isinstance(first, str):
            return json.loads(first)
    return {}


def extract_bullets(insights: Optional[dict], key: str) -> str:
    if not insights:
        return "None identified"
    items = insights.get("key_insights_summary", {}).get(key, [])
    return "\n".join(f"- {i['text']}" for i in items[:5]) or "None identified"


# ── LangGraph ──────────────────────────────────────────────────────────────────

async def run_analysis(ngrok_url: str, product: str) -> InsightsState:
    client = MultiServerMCPClient({
        "reddit-sentiment": {
            "url": f"{ngrok_url}/mcp",
            "transport": "sse",
        }
    })
    tools = await client.get_tools()
    sentiment_tool = next(t for t in tools if t.name == "analyze_reddit_sentiment")

    async def fetch_insights(state: InsightsState) -> dict:
        result = await sentiment_tool.ainvoke({"query": state["product"]})
        return {"raw_insights": parse_tool_result(result)}

    async def format_summary(state: InsightsState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a Customer Engineer preparing a customer insights card "
                "from Reddit data. Be concise and actionable."
            )),
            ("user", """Reddit sentiment data for {product}:

What users like:
{likes}

Pain points:
{pains}

Feature requests:
{requests}

Write a customer insights card with these three sections:

### What Customers Love
3-4 bullets on what users genuinely value about {product}.

### Key Pain Points
3-4 bullets on the most common frustrations — prioritised by frequency.

### Top Feature Requests
3-4 bullets on what users most want next from {product}.

End with one sentence: a CE Takeaway summarising the overall sentiment.""")
        ])

        chain = prompt | llm
        response = await chain.ainvoke({
            "product": state["product"],
            "likes": extract_bullets(state["raw_insights"], "what_users_like"),
            "pains": extract_bullets(state["raw_insights"], "major_frustrations"),
            "requests": extract_bullets(state["raw_insights"], "what_users_want"),
        })
        return {"summary": response.content}

    graph = StateGraph(InsightsState)
    graph.add_node("fetch_insights", fetch_insights)
    graph.add_node("format_summary", format_summary)
    graph.set_entry_point("fetch_insights")
    graph.add_edge("fetch_insights", "format_summary")
    graph.add_edge("format_summary", END)

    app = graph.compile()
    return await app.ainvoke({
        "ngrok_url": ngrok_url,
        "product": product,
        "raw_insights": None,
        "summary": None,
    })


# ── Streamlit UI ───────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Reddit Customer Insights", layout="wide")
    st.title("Reddit Customer Insights")
    st.caption("LangGraph + LangSmith + Reddit Sentiment MCP")

    with st.form("insights_form"):
        ngrok_url = st.text_input(
            "MCP Server URL",
            placeholder="https://xyz.ngrok.io",
            help="Your ngrok URL — the app appends /mcp automatically"
        )
        product = st.text_input("Product name", placeholder="e.g. LangChain")
        submitted = st.form_submit_button("Get Insights")

    if submitted:
        if not all([ngrok_url, product]):
            st.warning("Please fill in both fields.")
            return

        with st.spinner(f"Fetching Reddit insights for {product}..."):
            try:
                result = asyncio.run(run_analysis(ngrok_url.rstrip("/"), product))
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        st.success("Done. Full trace available in your LangSmith dashboard.")

        col_summary, col_raw = st.columns([3, 2])

        with col_summary:
            st.markdown("## Customer Insights")
            st.markdown(result["summary"])

        with col_raw:
            with st.expander("Raw Reddit data"):
                st.json(result.get("raw_insights", {}))


if __name__ == "__main__":
    main()
