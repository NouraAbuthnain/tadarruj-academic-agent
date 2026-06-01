"""
agent/nodes.py
LangGraph node functions for Tadarruj Phase 2.

Agents:
  - intent_node         : classifies the user request type
  - rag_node            : retrieves relevant knowledge context
  - plan_generator_node : generates a 4-phase study plan (with RAG context)
  - roadmap_node        : generates a 3-year academic roadmap (with RAG context)
  - recalibration_node  : handles chat follow-ups with memory + RAG
  - error_handler_node  : catches failures and retries or falls back gracefully
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from .state import TadarrujState, ChatMessage
from .rag import get_exam_context, get_roadmap_context, search, study_context_tool, roadmap_context_tool

load_dotenv()

# ── OpenRouter client ─────────────────────────────────────────
_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    base_url="https://openrouter.ai/api/v1",
)
MODEL = "mistralai/mistral-7b-instruct:free"
# MODEL = "google/gemma-4-31b-it:free"

SYSTEM_PROMPT = """أنت تدرّج، مساعد أكاديمي موثوق ومتخصص في توجيه طلاب المرحلة الثانوية السعودية. مهمتك تحويل أهداف الطالب إلى خطط واضحة وقابلة للتنفيذ.

مبادئك الأساسية:
- أسلوبك هادئ وداعم ومنظم، لا تخويف ولا ضغط.
- لا تقل أبداً 'أنت متأخر' أو 'فات الأوان' — بل قل 'يمكننا التعديل والمتابعة'.
- إذا سألك الطالب عن معلومة متغيرة، أحله للمصدر الرسمي.
- ركّز دائماً على الخطوة التالية القابلة للتنفيذ.
- أنشئ خطة من 4 مراحل: التعلم ← التطبيق ← التعمق ← المحاكاة.""".strip()

# Keep set_model for backward compatibility with Phase 1 (no-op when using API)
_model     = None
_tokenizer = None

def set_model(model, tokenizer):
    global _model, _tokenizer
    _model     = model
    _tokenizer = tokenizer


def _call_llm(messages: list, max_tokens: int = 700) -> str:
    """Call OpenRouter API and return the response text."""
    response = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return (response.choices[0].message.content or "").strip()


# ══════════════════════════════════════════════════════════════
# AGENT 1 — Intent Classifier
# ══════════════════════════════════════════════════════════════

def intent_node(state: TadarrujState) -> TadarrujState:
    """
    Classifies the user's intent based on available state:
    - study_plan:    request has a StudyPlanRequest and no existing plan
    - roadmap:       request has a RoadmapRequest
    - recalibration: existing study plan + update chat message
    - question:      chat message about study plan or roadmap
    """
    try:
        recal_keywords = [
            "عدّل", "عدل", "تعديل", "حدّث", "حدث",
            "فاتتني", "تأخرت", "غيّر", "غير",
            "أعد", "اعد", "خطة جديدة", "أضف", "اضف"
        ]

        # 1) Fresh roadmap generation
        if state.roadmap_request is not None and state.roadmap is None:
            return state.model_copy(update={"agent_intent": "roadmap"})

        # 2) Fresh study-plan generation
        if state.request is not None and state.plan is None:
            return state.model_copy(update={"agent_intent": "study_plan"})

        # 3) Chat mode: study plan or roadmap already exists
        if state.chat_history:
            last_msg = state.chat_history[-1].content

            if state.plan is not None:
                if any(k in last_msg for k in recal_keywords):
                    return state.model_copy(update={"agent_intent": "recalibration"})
                return state.model_copy(update={"agent_intent": "question"})

            if state.roadmap is not None:
                return state.model_copy(update={"agent_intent": "question"})

        # 4) Safe fallback
        if state.roadmap_request is not None:
            return state.model_copy(update={"agent_intent": "roadmap"})

        return state.model_copy(update={"agent_intent": "study_plan"})

    except Exception as e:
        return state.model_copy(update={
            "error": str(e),
            "failed_node": "intent_node"
        })


# ══════════════════════════════════════════════════════════════
# AGENT 2 — RAG Retrieval
# ══════════════════════════════════════════════════════════════

def rag_node(state: TadarrujState) -> TadarrujState:
    """
    Retrieves relevant knowledge context based on intent.
    Calls tools via .invoke() so tool usage is tracked by LangGraph.
    """
    try:
        context = ""

        if state.agent_intent == "study_plan" and state.request:
            context = get_exam_context(
                subject=state.request.subject,
                days_left=state.request.days_left,
            )

        elif state.agent_intent == "roadmap" and state.roadmap_request:
            grade  = state.roadmap_request.grade or "غير محدد"
            major  = state.roadmap_request.major_interest or "غير محدد"
            context = get_roadmap_context(grade=grade, major_interest=major)

        elif state.agent_intent in ("recalibration", "question") and state.chat_history:
            last_msg = state.chat_history[-1].content
            context  = search(last_msg)

        return state.model_copy(update={"rag_context": context})

    except Exception as e:
        # RAG failure is non-critical — continue without context
        print(f"[RAG NODE ERROR] {e}")
        return state.model_copy(update={"rag_context": ""})


# ══════════════════════════════════════════════════════════════
# AGENT 3 — Study Plan Generator
# ══════════════════════════════════════════════════════════════

def plan_generator_node(state: TadarrujState) -> TadarrujState:
    """
    Generates a 4-phase Arabic study plan from StudyPlanRequest.
    Injects RAG context if available to enrich the plan.
    """
    if not state.request:
        return state.model_copy(update={"error": "لا توجد بيانات طلب للخطة الدراسية."})

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

    user_prompt = " ".join(lines)

    if state.rag_context:
        user_prompt += f"\n\nمعلومات إضافية مفيدة من قاعدة المعرفة:\n{state.rag_context}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    try:
        plan = _call_llm(messages, max_tokens=800)
        return state.model_copy(update={"plan": plan, "error": None})
    except Exception as e:
        print(f"[PLAN_GENERATOR_ERROR] {e}")
        return state.model_copy(update={
            "error": "تعذر إنشاء الخطة الدراسية حاليًا. يرجى المحاولة مرة أخرى.",
            "failed_node": "plan_generator_node",
        })


# ══════════════════════════════════════════════════════════════
# AGENT 4 — Roadmap Generator
# ══════════════════════════════════════════════════════════════

def roadmap_node(state: TadarrujState) -> TadarrujState:
    """
    Generates a 3-year academic roadmap based on the student's
    current grade, major interest, and target university.
    Uses RAG context for exam requirements and university info.
    """
    if not state.roadmap_request:
        return state.model_copy(update={"error": "لا توجد بيانات لإنشاء خارطة الطريق."})

    r = state.roadmap_request

    grade = r.grade or "غير محدد"
    major = r.major_interest or "غير محدد"

    user_prompt = (
        f"أنشئ خارطة طريق أكاديمية مفصّلة لطالب في الصف {grade} "
        f"يرغب في دراسة {major}."
    )

    if r.target_university:
        user_prompt += f" جامعته المستهدفة: {r.target_university}."

    user_prompt += (
        "\n\nيجب أن تشمل خارطة الطريق:\n"
        "1. أهداف كل صف دراسي (العاشر، الحادي عشر، الثاني عشر)\n"
        "2. متطلبات القبول في التخصص المستهدف\n"
        "3. متى يبدأ التحضير لاختبار القدرات والتحصيلي\n"
        "4. ساعات التطوع\n"
        "5. نصائح للحفاظ على المعدل التراكمي"
    )

    if state.rag_context:
        user_prompt += f"\n\nمعلومات من قاعدة المعرفة:\n{state.rag_context}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    try:
        roadmap = _call_llm(messages, max_tokens=900)
        return state.model_copy(update={"roadmap": roadmap, "error": None})
    except Exception as e:
        print(f"[ROADMAP_ERROR] {e}")
        return state.model_copy(update={
            "error": "تعذر إنشاء خارطة الطريق حاليًا. يرجى المحاولة مرة أخرى.",
            "failed_node": "roadmap_node",
        })


# ══════════════════════════════════════════════════════════════
# AGENT 5 — Recalibration (Chat)
# ══════════════════════════════════════════════════════════════

def recalibration_node(state: TadarrujState) -> TadarrujState:
    """
    Handles chat follow-ups from the student.
    Supports both study plan recalibration and roadmap Q&A.
    """
    if not state.chat_history:
        return state.model_copy(update={"error": "No chat message to process."})

    recal_keywords = [
        "عدّل", "عدل", "تعديل", "حدّث", "حدث",
        "فاتتني", "تأخرت", "غيّر", "غير",
        "أعد", "اعد", "خطة جديدة", "أضف", "اضف"
    ]

    last_user_msg = state.chat_history[-1].content
    update_plan   = state.plan is not None and any(k in last_user_msg for k in recal_keywords)

    if state.plan:
        context_text = f"هذه هي الخطة الدراسية الحالية للطالب:\n\n{state.plan}"
    elif state.roadmap:
        context_text = f"هذه هي خارطة الطريق الأكاديمية الحالية للطالب:\n\n{state.roadmap}"
    else:
        context_text = "لم يتم إنشاء خطة دراسية أو خارطة طريق بعد."

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": context_text},
    ]

    if state.rag_context:
        messages.append({
            "role":    "system",
            "content": f"معلومات مفيدة من قاعدة المعرفة:\n{state.rag_context}"
        })

    for msg in state.chat_history:
        messages.append({"role": msg.role, "content": msg.content})

    try:
        answer      = _call_llm(messages, max_tokens=600)
        new_history = state.chat_history + [ChatMessage(role="assistant", content=answer)]

        return state.model_copy(update={
            "chat_history": new_history,
            "plan":         answer if update_plan else state.plan,
            "roadmap":      state.roadmap,
            "error":        None,
        })

    except Exception as e:
        print(f"[CHAT_ERROR] {e}")
        return state.model_copy(update={
            "error": "حدثت مشكلة أثناء تحديث الخطة. يرجى المحاولة مرة أخرى.",
            "failed_node": "recalibration_node",
        })


# ══════════════════════════════════════════════════════════════
# ERROR HANDLER NODE
# ══════════════════════════════════════════════════════════════

def error_handler_node(state: TadarrujState) -> TadarrujState:
    """
    Handles node failures with retry logic.
    - retry_count < 2 : increment and retry the failed node
    - retry_count >= 2: return a user-friendly Arabic fallback message
    """
    if state.retry_count < 2:
        print(f"[ERROR HANDLER] Retrying... attempt {state.retry_count + 1}")
        return state.model_copy(update={
            "retry_count": state.retry_count + 1,
            "error":       None,
            "failed_node": state.failed_node,
        })

    # Max retries reached — return friendly fallback
    fallback = (
        "عذراً، حدث خطأ غير متوقع ولم نتمكن من إكمال طلبك. "
        "يرجى المحاولة مرة أخرى لاحقاً، أو تحديث الصفحة."
    )
    return state.model_copy(update={
        "plan":        fallback if not state.plan    else state.plan,
        "roadmap":     fallback if not state.roadmap else state.roadmap,
        "error":       None,
        "retry_count": 0,
    })