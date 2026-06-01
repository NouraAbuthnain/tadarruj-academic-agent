"""
agent/graph.py
Builds and compiles the Tadarruj multi-agent LangGraph workflow (Phase 2).

Graph flow:
  intent_node → rag_node → [plan_generator | roadmap_generator | recalibration] → END
                                      ↓ (on error)
                              error_handler_node → retry or END
"""

from langgraph.graph import StateGraph, END

from .state import TadarrujState
from . import nodes

_agent = None   # compiled graph, cached after first build


# ── Routing functions (conditional edges) ────────────────────

def _route_after_intent(state: TadarrujState) -> str:
    """Route to RAG after intent classification, or error handler on failure."""
    if state.error:
        return "error_handler"
    return "rag"


def _route_after_rag(state: TadarrujState) -> str:
    """Route to the correct agent based on intent."""
    if state.error:
        return "error_handler"
    intent = state.agent_intent or "study_plan"
    if intent == "roadmap":
        return "roadmap_generator"
    if intent in ("recalibration", "question"):
        return "recalibration"
    return "plan_generator"   # default: study_plan


def _route_after_agent(state: TadarrujState) -> str:
    """After any agent: go to error handler if failed, otherwise END."""
    if state.error:
        return "error_handler"
    return END


def _route_after_error(state: TadarrujState) -> str:
    """After error handler: retry the failed node or END if max retries reached."""
    if state.retry_count > 0 and state.failed_node:
        node_map = {
            "intent_node":         "intent",
            "plan_generator_node": "plan_generator",
            "roadmap_node":        "roadmap_generator",
            "recalibration_node":  "recalibration",
        }
        return node_map.get(state.failed_node, END)
    return END


# ── Graph builder ─────────────────────────────────────────────

def build_agent():
    """Compile and return the LangGraph multi-agent graph (cached)."""
    global _agent
    if _agent is not None:
        return _agent

    g = StateGraph(TadarrujState)

    # Register all nodes
    g.add_node("intent",         nodes.intent_node)
    g.add_node("rag",            nodes.rag_node)
    g.add_node("plan_generator", nodes.plan_generator_node)
    g.add_node("roadmap_generator", nodes.roadmap_node)
    g.add_node("recalibration",  nodes.recalibration_node)
    g.add_node("error_handler",  nodes.error_handler_node)

    # Entry point
    g.set_entry_point("intent")

    # Conditional edges
    g.add_conditional_edges("intent", _route_after_intent, {
        "rag":           "rag",
        "error_handler": "error_handler",
    })

    g.add_conditional_edges("rag", _route_after_rag, {
        "plan_generator": "plan_generator",
        "roadmap_generator":  "roadmap_generator",
        "recalibration":  "recalibration",
        "error_handler":  "error_handler",
    })

    g.add_conditional_edges("plan_generator", _route_after_agent, {
        "error_handler": "error_handler",
        END:              END,
    })

    g.add_conditional_edges("roadmap_generator", _route_after_agent, {
        "error_handler": "error_handler",
        END:              END,
    })

    g.add_conditional_edges("recalibration", _route_after_agent, {
        "error_handler": "error_handler",
        END:              END,
    })

    g.add_conditional_edges("error_handler", _route_after_error, {
        "intent":         "intent",
        "plan_generator": "plan_generator",
        "roadmap_generator": "roadmap_generator",
        "recalibration":  "recalibration",
        END:               END,
    })

    _agent = g.compile()
    return _agent


# ── Convenience functions (used by app.py) ────────────────────

def generate_plan(request) -> str:
    """Generate a fresh study plan from a StudyPlanRequest."""
    agent  = build_agent()
    state  = TadarrujState(request=request)
    result = agent.invoke(state)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result["plan"]


def generate_roadmap(roadmap_request) -> str:
    """Generate a 3-year academic roadmap from a RoadmapRequest."""
    agent  = build_agent()
    state  = TadarrujState(roadmap_request=roadmap_request)
    result = agent.invoke(state)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result["roadmap"]


def chat_reply(state: TadarrujState) -> TadarrujState:
    """Run the full agent graph for a chat message (recalibration or question)."""
    agent  = build_agent()
    result = agent.invoke(state)
    return TadarrujState(**result)


def load_model():
    """Load the fine-tuned LoRA model (GPU only — not used on Streamlit Cloud)."""
    from unsloth import FastLanguageModel  # lazy import — GPU only
    import os
    MODEL_PATH = os.path.join("models", "tadarruj_qwen2_5_7b_lora")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    nodes.set_model(model, tokenizer)
    return model, tokenizer