# flow.py
from langgraph.graph import StateGraph
from tools.segment_insight_tool import segment_insight
from tools.performance_analysis_tool import performance_analysis
from agents.strategy_generator_agent import generate_strategy
from typing import TypedDict

class FlowState(TypedDict):
    query: str
    segment_insight: str           # JSON string of records
    performance_analysis: dict     # {"roi_summary": {...}}
    strategy_generator: str

workflow = StateGraph(FlowState)

def segment_node_func(state):
    result = segment_insight.invoke({"query": state["query"]})
    # tool return {"segment_insight": "<json str>"}
    seg_str = result.get("segment_insight", "[]") if isinstance(result, dict) else (result or "[]")
    return {"segment_insight": seg_str}

def performance_node_func(state):
    result = performance_analysis.invoke({
        "query": state["query"],
        "segment_json": state["segment_insight"]
    })
    # tool return {"roi_summary": {...}}
    return {"performance_analysis": result if isinstance(result, dict) else {"roi_summary": {}}}

def strategy_node_func(state):
    res = generate_strategy(
        query=state["query"],
        performance_analysis=state["performance_analysis"]
    )
    return {"strategy_generator": res}

workflow.add_node("segment_insight", segment_node_func)
workflow.add_node("performance_analysis", performance_node_func)
workflow.add_node("strategy_generator", strategy_node_func)

workflow.set_entry_point("segment_insight")
workflow.add_edge("segment_insight", "performance_analysis")
workflow.add_edge("performance_analysis", "strategy_generator")
workflow.set_finish_point("strategy_generator")

graph_app = workflow.compile()
