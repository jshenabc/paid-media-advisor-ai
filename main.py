# main.py
import os
import json
from typing import List, Optional, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from graph.flow import graph_app

load_dotenv()
app = FastAPI(title="Paid Media Advisor API", version="1.0.0")


# ---------- Pydantic schemas (for OpenAPI + Orchestrate) ----------

class SegmentRecord(BaseModel):
    business_unit: str
    channel: str
    region: str
    roi: float
    clicks: int
    cost_per_click: float
    cpm: float

class PerformanceSummary(BaseModel):
    high_performers: List[str]
    low_performers: List[str]
    average_roi: Optional[float] = None
    top_features: str
    projected_roi: Optional[float] = None

class RecommendRequest(BaseModel):
    query: str
    format: Literal["json", "html", "markdown"] = "json"

class RecommendResponse(BaseModel):
    segment: List[SegmentRecord]
    performance: PerformanceSummary
    strategy: str
    html: Optional[str] = None
    markdown: Optional[str] = None


# ---------- Helpers ----------

def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=1, classes="carbon-table")

def _table_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


# ---------- Health ----------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- The one tool Orchestrate will call ----------

@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is empty")

    # 1) LangGraph：segment -> performance -> strategy
    result = graph_app.invoke({"query": query})

    # 2) segment
    seg_json_str = result.get("segment_insight", "[]")
    try:
        seg_list = json.loads(seg_json_str) if isinstance(seg_json_str, str) else seg_json_str
    except Exception:
        seg_list = []
    seg_df = pd.DataFrame(seg_list)
 
    for col in ["business_unit","channel","region","roi","clicks","cost_per_click","cpm"]:
        if col not in seg_df.columns:
            seg_df[col] = [] if col in ["business_unit","channel","region"] else 0

    # 3) performance
    perf_raw = result.get("performance_analysis", {})
    if isinstance(perf_raw, dict) and "roi_summary" in perf_raw:
        perf = perf_raw["roi_summary"]
    else:
        perf = perf_raw  # already a dict summary

    perf_obj = PerformanceSummary(
        high_performers = perf.get("high_performers", []) or [],
        low_performers  = perf.get("low_performers", []) or [],
        average_roi     = perf.get("average_roi"),
        top_features    = perf.get("top_features", "") or "",
        projected_roi   = perf.get("projected_roi"),
    )

    # 4) strategy
    strategy_text = result.get("strategy_generator", "")
    if "token_quota_reached" in str(strategy_text).lower() or "status code: 403" in str(strategy_text):
        strategy_text = (
            "⚠️ LLM token quota exhausted for this project. "
            "Returning segment & performance analysis. "
            "Please add quota or switch to a smaller model / lower max_new_tokens to enable strategy generation."
        )
    elif not strategy_text:
        strategy_text = "No strategy generated. Please refine your query."

    # 5) HTML/Markdown
    html_view = None
    md_view = None
    if not seg_df.empty:
        seg_html = _table_html(seg_df)
        seg_md = _table_markdown(seg_df)
    else:
        seg_html = "<i>No matched campaigns.</i>"
        seg_md = "_No matched campaigns._"

    perf_html = f"""
    <table class="carbon-table">
        <tr><th>Avg ROI</th><td>{perf_obj.average_roi if perf_obj.average_roi is not None else "N/A"}</td></tr>
        <tr><th>Projected ROI</th><td>{perf_obj.projected_roi if perf_obj.projected_roi is not None else "N/A"}</td></tr>
        <tr><th>Top Features</th><td>{perf_obj.top_features or "N/A"}</td></tr>
        <tr><th>High Performers</th><td>{'<br>'.join(perf_obj.high_performers) or "None"}</td></tr>
        <tr><th>Low Performers</th><td>{'<br>'.join(perf_obj.low_performers) or "None"}</td></tr>
    </table>
    """.strip()

    perf_md = (
        f"| Metric | Value |\n|---|---|\n"
        f"| Avg ROI | {perf_obj.average_roi if perf_obj.average_roi is not None else 'N/A'} |\n"
        f"| Projected ROI | {perf_obj.projected_roi if perf_obj.projected_roi is not None else 'N/A'} |\n"
        f"| Top Features | {perf_obj.top_features or 'N/A'} |\n"
        f"| High Performers | {( '; '.join(perf_obj.high_performers) or 'None')} |\n"
        f"| Low Performers | {( '; '.join(perf_obj.low_performers) or 'None')} |\n"
    )

    if body.format in ("html", "markdown"):
        html_view = f"""
        <div class="recommendation-box">
            <h3>Matched Campaign Segment</h3>
            {seg_html}
            <h3>Performance Analysis</h3>
            {perf_html}
            <h3>AI Marketing Strategy Recommendation</h3>
            <p>{strategy_text}</p>
        </div>
        """.strip()

        md_view = (
            "### Matched Campaign Segment\n"
            f"{seg_md}\n\n"
            "### Performance Analysis\n"
            f"{perf_md}\n\n"
            "### AI Marketing Strategy Recommendation\n"
            f"{strategy_text}\n"
        )

    # 6) response
    resp = RecommendResponse(
        segment=[SegmentRecord(**r) for r in seg_df.to_dict(orient="records")],
        performance=perf_obj,
        strategy=strategy_text,
        html=html_view if body.format == "html" else None,
        markdown=md_view if body.format == "markdown" else None,
    )
    return JSONResponse(content=json.loads(resp.model_dump_json()))
