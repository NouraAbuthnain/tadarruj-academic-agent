"""
agent/state.py
Pydantic models for the Tadarruj Phase 1 agent.
"""

from pydantic import BaseModel, Field
from typing import Optional, List

class StudyPlanRequest(BaseModel):
    """Validated inputs from the Streamlit structured form."""
    subject:        str
    days_left:      int
    chapters:       Optional[int]   = None   # either chapters OR total_hours must be set
    total_hours:    Optional[float] = None
    hours_per_day:  float
    difficulty:     str                      # "سهل" | "متوسط" | "صعب"
    current_score:  Optional[int]   = None   # for retake scenarios
    target_score:   int

    def content_description(self) -> str:
        """Returns a single Arabic phrase describing the content volume."""
        if self.chapters:
            return f"{self.chapters} فصلاً"
        if self.total_hours:
            return f"إجمالي ساعات المحتوى: {self.total_hours} ساعة"
        return "محتوى غير محدد"


class ChatMessage(BaseModel):
    role:    str   # "user" | "assistant"
    content: str


class TadarrujState(BaseModel):
    """Full agent state, carried through the LangGraph graph."""
    request:      StudyPlanRequest
    plan:         Optional[str]              = None   # generated study plan text
    chat_history: List[ChatMessage]          = Field(default_factory=list)
    error:        Optional[str]              = None
