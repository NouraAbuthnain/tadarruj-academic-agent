"""
agent/state.py
Pydantic models for the Tadarruj agent (Phase 1 + Phase 2).
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class StudyPlanRequest(BaseModel):
    """Validated inputs from the Streamlit structured form."""
    subject:       str
    days_left:     int
    chapters:      Optional[int]   = None   # either chapters OR total_hours must be set
    total_hours:   Optional[float] = None
    hours_per_day: float
    difficulty:    str                      # "سهل" | "متوسط" | "صعب"
    current_score: Optional[int]   = None   # for retake scenarios
    target_score:  int

    # ── Roadmap context (optional, from Phase 2 fields) ───────
    grade:         Optional[str]   = None   # e.g. "الصف الثالث الثانوي"
    major:         Optional[str]   = None   # e.g. "هندسة"

    def content_description(self) -> str:
        """Returns a single Arabic phrase describing the content volume."""
        if self.chapters:
            return f"{self.chapters} فصلاً"
        if self.total_hours:
            return f"إجمالي ساعات المحتوى: {self.total_hours} ساعة"
        return "محتوى غير محدد"


class RoadmapRequest(BaseModel):
    """Inputs for the 3-year academic roadmap feature."""
    grade:             Optional[str] = None   # e.g. "الصف الثالث الثانوي"
    major_interest:    Optional[str] = None   # e.g. "طب", "هندسة", "حاسب"
    target_university: Optional[str] = None   # e.g. "جامعة الملك سعود"
    subject:           Optional[str] = None   # exam subject for context
    target_score:      Optional[int] = None   # target exam score for context


class ChatMessage(BaseModel):
    role:    str   # "user" | "assistant"
    content: str


class TadarrujState(BaseModel):
    """Full agent state, carried through the LangGraph graph."""

    # ── Core inputs ───────────────────────────────────────────
    request:         Optional[StudyPlanRequest] = None
    roadmap_request: Optional[RoadmapRequest]   = None

    # ── Agent classification ──────────────────────────────────
    # NOTE: never store this field directly in st.session_state —
    # it is internal to the agent graph only.
    agent_intent: Optional[Literal[
        "study_plan",       # generate a new study plan
        "roadmap",          # generate a 3-year academic roadmap
        "recalibration",    # update existing plan via chat
        "question",         # answer a general academic question
    ]] = None

    # ── Agent outputs ─────────────────────────────────────────
    plan:        Optional[str] = None   # generated study plan text
    roadmap:     Optional[str] = None   # generated 3-year roadmap text
    rag_context: Optional[str] = None   # retrieved knowledge from RAG

    # ── Multi-turn memory ─────────────────────────────────────
    chat_history: List[ChatMessage] = Field(default_factory=list)

    # ── Error handling ────────────────────────────────────────
    error:       Optional[str] = None
    retry_count: int           = 0
    failed_node: Optional[str] = None