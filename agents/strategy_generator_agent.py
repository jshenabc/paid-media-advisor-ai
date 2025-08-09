from langchain_ibm import WatsonxLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

watsonx_llm = WatsonxLLM(
    model_id="meta-llama/llama-2-13b-chat",
    url="https://us-south.ml.cloud.ibm.com",
    apikey=os.getenv("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 200,
        "temperature": 0.3,
        "repetition_penalty": 1.0,
    }
)

prompt = PromptTemplate.from_template("""
You are a marketing advisor AI. Use the analysis to produce a concise, actionable and EXPLAINABLE strategy.
- If the user asks ROI with a budget, refer to `projected_roi`.
- Justify each recommendation with a driver (top_features/high_performers/low_performers).
- Keep it under 8 bullet points.

User question:
{query}

Analysis (JSON-like):
{performance_analysis}

Now output:
- 1–2 sentence summary with projected ROI if available
- 3–6 bullets of actions with reasons
""")

chain = LLMChain(llm=watsonx_llm, prompt=prompt)

def generate_strategy(query: str, performance_analysis) -> str:
    try:
        return chain.run({"query": query, "performance_analysis": performance_analysis})
    except Exception as e:
        return f"Strategy generation error: {e}"
