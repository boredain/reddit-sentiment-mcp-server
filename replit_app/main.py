import os
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "").rstrip("/")

app = FastAPI(title="Reddit Sentiment Analyzer")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/api/analyze")
async def analyze(body: dict):
    if not MCP_SERVER_URL:
        return JSONResponse(
            status_code=500,
            content={"error": "MCP_SERVER_URL environment variable is not set"}
        )
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{MCP_SERVER_URL}/analyze", json=body)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
