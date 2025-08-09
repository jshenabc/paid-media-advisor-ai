import pandas as pd
import xgboost as xgb
from pydantic import BaseModel
from langchain_core.tools import StructuredTool
import re
from io import StringIO

booster = xgb.Booster()
booster.load_model("tools/xgb_model.json")

class ModelInput(BaseModel):
    query: str
    segment_json: str

def _predict_and_explain(query: str, segment_json: str) -> dict:
    """
    Predict ROI for each campaign in the segment and return explainable summary.
    If query includes an investment (e.g., $20,000), simulate a campaign and return projected ROI.
    """
    segment_df = pd.read_json(StringIO(segment_json))

    if segment_df.empty:
        return {"roi_summary": {
            "high_performers": [],
            "low_performers": [],
            "average_roi": None,
            "top_features": "",
            "projected_roi": None
        }}

    model_input_df = segment_df[["clicks", "cost_per_click", "cpm"]]
    dmatrix = xgb.DMatrix(model_input_df)
    predictions = booster.predict(dmatrix)

    segment_df = segment_df.copy()
    segment_df["predicted_roi"] = predictions

    avg_roi = round(segment_df["predicted_roi"].mean(), 4)

    high = segment_df[segment_df["roi"] > 2.0]
    low = segment_df[segment_df["roi"] < 1.0]

    def format_campaign(row):
        return f"{row['business_unit']} | {row['channel']} | {row['region']} → ROI: {round(row['roi'], 2)}"

    high_list = [format_campaign(r) for _, r in high.iterrows()]
    low_list = [format_campaign(r) for _, r in low.iterrows()]

    importance = booster.get_score(importance_type="gain")
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    explanation = ", ".join([f"{k} ({round(v, 2)})" for k, v in top_features])

    # Optional projected ROI from investment
    projected_roi = None
    match = re.search(r"\$?([\d,]+)", query)
    if match:
        amount_str = match.group(1).replace(",", "")
        try:
            budget = float(amount_str)

            # Use avg CPC and CPM to estimate clicks
            avg_cpc = segment_df["cost_per_click"].mean()
            avg_cpm = segment_df["cpm"].mean()
            estimated_clicks = budget / avg_cpc if avg_cpc > 0 else 1000

            # Construct fake campaign
            synthetic = pd.DataFrame([{
                "clicks": estimated_clicks,
                "cost_per_click": avg_cpc,
                "cpm": avg_cpm
            }])
            synthetic_dmatrix = xgb.DMatrix(synthetic)
            projected = booster.predict(synthetic_dmatrix)[0]
            projected_roi = round(projected, 4)
        except Exception:
            projected_roi = None

    return {
        "roi_summary": {
            "high_performers": high_list,
            "low_performers": low_list,
            "average_roi": avg_roi,
            "top_features": explanation,
            "projected_roi": projected_roi
        }
    }

performance_analysis = StructuredTool.from_function(
    _predict_and_explain,
    name="performance_analysis",
    description="Estimate campaign ROI from segment data and explain key drivers. If budget is specified in the query, return projected ROI.",
    args_schema=ModelInput
)
