import pandas as pd
from langchain_core.tools import tool

df = pd.read_csv("data/campaign_data.csv")

@tool
def segment_insight(query: str) -> dict:
    """
    Analyze campaign data based on query. Automatically matches region, business_unit, channel, and performance terms.
    """
    query = query.lower()
    filtered = df.copy()

    keyword_map = {
        "region": ["north america", "europe", "asia"],
        "business_unit": ["gbs", "watson", "cloud", "security", "analytics"],
        "channel": ["display", "paid search", "paid social"]
    }

    # filter
    for col, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in query:
                filtered = filtered[filtered[col].str.lower() == keyword]

    # specific performance terms
    if "low roi" in query:
        filtered = filtered[filtered["roi"] < 0.2]
    if "high cpc" in query:
        filtered = filtered[filtered["cost_per_click"] > 2.0]

    if filtered.empty:
        return {"segment_insight": "[]"}

    segment_json = filtered[[
        "business_unit", "channel", "region", "roi", "clicks",
        "cost_per_click", "cpm"
    ]].to_json(orient="records")

    return {"segment_insight": segment_json}
