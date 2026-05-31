"""
agent/nodes.py
LangGraph node functions for Tadarruj.

Nodes:
  - plan_generator_node : generates a 4-phase study plan from a StudyPlanRequest
  - recalibration_node  : handles chat follow-ups (missed days, add subject, etc.)
"""

from typing import Any

import torch
from .state import TadarrujState, ChatMessage

SYSTEM_PROMPT = """أنت تدرّج، مساعد أكاديمي موثوق ومتخصص في توجيه طلاب المرحلة الثانوية السعودية (الصف العاشر والحادي عشر والثاني عشر). مهمتك تحويل أهداف الطالب إلى خطط واضحة وقابلة للتنفيذ.

مبادئك الأساسية:
- أسلوبك هادئ وداعم ومنظم، لا تخويف ولا ضغط.
- لا تقل أبداً 'أنت متأخر' أو 'فات الأوان' — بل قل 'يمكننا التعديل والمتابعة'.
- إذا سألك الطالب عن معلومة متغيرة (مواعيد، نسب قبول، أسعار)، أحله للمصدر الرسمي.
- إذا كان السؤال غير واضح، اطلب التفاصيل بدلاً من التخمين.
- ركّز دائماً على الخطوة التالية القابلة للتنفيذ.""".strip()

# These are set once by graph.py when the model is loaded
_model: Any = None
_tokenizer: Any = None
MAX_SEQ_LENGTH = 2048


def set_model(model: Any, tokenizer: Any) -> None:
    global _model, _tokenizer
    _model = model
    _tokenizer = tokenizer


def _generate(messages: list, max_new_tokens: int = 600) -> str:
    """Core generation helper with attention_mask fix for Qwen2.5."""
    if _model is None or _tokenizer is None:
        raise RuntimeError(
            "Model and tokenizer are not loaded. Call set_model(model, tokenizer) first."
        )

    input_ids = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_model.device)

    # Fix for Qwen2.5 pad_token == eos_token warning
    attention_mask = (input_ids != _tokenizer.eos_token_id).long()

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    generated = output_ids[0][input_ids.shape[-1]:]
    return _tokenizer.decode(generated, skip_special_tokens=True).strip()


def plan_generator_node(state: TadarrujState) -> TadarrujState:
    """
    Builds a structured Arabic prompt from the StudyPlanRequest fields
    and calls the fine-tuned model to produce a 4-phase study plan.
    """
    r = state.request

    # Build the structured user prompt (matches training data format)
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    try:
        plan = _generate(messages, max_new_tokens=700)
        return state.model_copy(update={"plan": plan})
    except Exception as e:
        print(f"[PLAN_GENERATOR_ERROR] {e}")  
        
        return state.model_copy(update={
        "error": "تعذر إنشاء الخطة الدراسية حاليًا. يرجى المحاولة مرة أخرى."
        })


def recalibration_node(state: TadarrujState) -> TadarrujState:
    """
    Handles a new chat message from the student.
    Includes the existing plan + chat history so the model can
    modify the plan or answer a question about it.
    """
    if not state.chat_history:
        return state.model_copy(update={"error": "No chat message to process."})

    # Build conversation: system → plan summary → chat history
    plan_context = (
    f"هذه هي الخطة الدراسية الحالية للطالب:\n\n{state.plan}"
    if state.plan else "لم يتم إنشاء خطة دراسية بعد."
    )
    
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": plan_context},
    ]

    # Add full chat history
    for msg in state.chat_history:
        messages.append({"role": msg.role, "content": msg.content})

    try:
        answer = _generate(messages, max_new_tokens=500)

        new_history = state.chat_history + [
            ChatMessage(role="assistant", content=answer)
        ]

        # If the student asked to recalibrate, replace the plan
        recal_keywords = [
            "عدّل", "عدل", "تعديل", "حدّث", "حدث",
            "فاتتني", "تأخرت", "غيّر", "غير",
            "أعد", "اعد", "خطة جديدة", "أضف", "اضف"
            ]
            
        last_user_msg  = state.chat_history[-1].content if state.chat_history else ""
        update_plan    = any(k in last_user_msg for k in recal_keywords)

        return state.model_copy(update={
            "chat_history": new_history,
            "plan": answer if update_plan else state.plan,
        })
    except Exception as e:
        print(f"[CHAT_ERROR] {e}")
        
        return state.model_copy(update={
        "error": "حدثت مشكلة أثناء تحديث الخطة. يرجى المحاولة مرة أخرى."
        })