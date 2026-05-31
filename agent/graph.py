"""
agent/graph.py
Builds and compiles the Tadarruj LangGraph agent.
Also handles loading the fine-tuned model (called once at app startup).
"""

import os
from langgraph.graph import StateGraph, END

from .state import TadarrujState
from . import nodes

MODEL_PATH     = os.path.join("models", "tadarruj_qwen2_5_7b_lora")
MAX_SEQ_LENGTH = 2048

_agent = None   # compiled graph, cached after first load


def load_model():
    """Load the fine-tuned LoRA model. Called once via @st.cache_resource."""
    from unsloth import FastLanguageModel  # GPU-only; imported lazily so Streamlit Cloud can start without it
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    nodes.set_model(model, tokenizer)
    return model, tokenizer


def build_agent():
    """Compile and return the LangGraph agent (cached after first call)."""
    global _agent
    if _agent is not None:
        return _agent

    g = StateGraph(TadarrujState)
    g.add_node("plan_generator",  nodes.plan_generator_node)
    g.add_node("recalibration",   nodes.recalibration_node)

    g.set_entry_point("plan_generator")
    g.add_edge("plan_generator", END)
    # recalibration is invoked separately (not in the default path)

    _agent = g.compile()
    return _agent


def generate_plan(request) -> str:
    """Convenience function: generate a fresh study plan from a StudyPlanRequest."""
    agent = build_agent()
    state = TadarrujState(request=request)
    result = agent.invoke(state)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result["plan"]


def chat_reply(state: TadarrujState) -> TadarrujState:
    """Convenience function: run the recalibration node for a chat message."""
    return nodes.recalibration_node(state)
