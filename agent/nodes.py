"""
agent/nodes.py
LangGraph node functions for Tadarruj.

Nodes:
  - plan_generator_node : generates a 4-phase study plan from a StudyPlanRequest
  - recalibration_node  : handles chat follow-ups (missed days, add subject, etc.)
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from .state import TadarrujState, ChatMessage

load_dotenv()

# ── OpenRouter client ─────────────────────────────────────────
_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    base_url="https://openrouter.ai/api/v1",
)
MODEL = "google/gemma-4-31b-it:free"

SYSTEM_PROMPT = """أنت تدرّج، مساعد أكاديمي موثوق ومتخصص في توجيه طلاب المرحلة الثانوية السعودية (الصف العاشر والحادي عشر والثاني عشر). مهمتك تحويل أهداف الطالب إلى خطط واضحة وقابلة للتنفيذ.

مبادئك الأساسية:
- أسلوبك هادئ وداعم ومنظم، لا تخويف ولا ضغط.
- لا تقل أبداً 'أنت متأخر' أو 'فات الأوان' — بل قل 'يمكننا التعديل والمتابعة'.
- إذا سألك الطالب عن معلومة متغيرة (مواعيد، نسب قبول، أسعار)، أحله للمصدر الرسمي.
- إذا كان السؤال غير واضح، اطلب التفاصيل بدلاً من التخمين.
- ركّز دائماً على الخطوة التالية القابلة للتنفيذ.
- أنشئ خطة من 4 مراحل: التعلم ← التطبيق ← التعمق ← المحاكاة.""".strip()

# Keep set_model for backward compatibility (no-op when using API)
_model = None
_tokenizer = None

def set_model(model, tokenizer):
    global _model, _tokenizer
    _model = model
    _tokenizer = tokenizer


def _generate(messages: list, max_tokens: int = 700) -> str:
    """Call OpenRouter API and return the response text."""
    response = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return (response.choices[0].message.content or "").strip()

def plan_generator_node(state: TadarrujState) -> TadarrujState:
    """
    Builds a structured Arabic prompt from the StudyPlanRequest fields
    and calls the API to produce a 4-phase study plan.
    """
    r = state.request

    lines = [
        f"المادة: {r.subject}.",
        f"الأيام المتبقية: {r.days_left} يوماً.",
        r.content_description() + ".",
        f"ساعات الدراسة اليومية: {r.hours_per_day} ساعة.",
        f"مستوى الصعوبة: {r.difficulty}.",
    ]
    if r.current_score is not None:
        lines.append(f"الدرجة الحالية: {r.current_score}.")
    lines.append(f"الدرجة المستهدفة: {r.target_score}.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": " ".join(lines)},
    ]

    try:
        plan = _generate(messages, max_tokens=700)
        return state.model_copy(update={"plan": plan})
    except Exception as e:
        print(f"[PLAN_GENERATOR_ERROR] {e}")
        return state.model_copy(update={"error": "تعذر إنشاء الخطة الدراسية حاليًا. يرجى المحاولة مرة أخرى."})


def recalibration_node(state: TadarrujState) -> TadarrujState:
    """
    Handles a new chat message from the student.
    Includes the existing plan + chat history so the model can
    modify the plan or answer a question about it.
    """
    if not state.chat_history:
        return state.model_copy(update={"error": "No chat message to process."})

    plan_context = (
        f"هذه هي الخطة الدراسية الحالية للطالب:\n\n{state.plan}"
        if state.plan else "لم يتم إنشاء خطة دراسية بعد."
    )

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": plan_context},
    ]

    for msg in state.chat_history:
        messages.append({"role": msg.role, "content": msg.content})

    try:
        answer = _generate(messages, max_tokens=500)
        new_history = state.chat_history + [ChatMessage(role="assistant", content=answer)]

        recal_keywords = ["عدّل", "عدل", "تعديل", "حدّث", "حدث",
                          "فاتتني", "تأخرت", "غيّر", "غير",
                          "أعد", "اعد", "خطة جديدة", "أضف", "اضف"]
        last_user_msg = state.chat_history[-1].content if state.chat_history else ""
        update_plan = any(k in last_user_msg for k in recal_keywords)

        return state.model_copy(update={
            "chat_history": new_history,
            "plan": answer if update_plan else state.plan,
        })
    except Exception as e:
        print(f"[CHAT_ERROR] {e}")
        return state.model_copy(update={"error": "حدثت مشكلة أثناء تحديث الخطة. يرجى المحاولة مرة أخرى."})