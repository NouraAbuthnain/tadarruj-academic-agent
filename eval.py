"""
eval.py
Step 6 — Evaluation script for Tadarruj Phase 2.

Tests:
  A) RAG vs no-RAG plan quality comparison
  B) Error handling with bad / edge-case inputs

Run from project root:
    python eval.py
"""

import os
import sys
import time

# ── Fix import path ───────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from agent.graph import build_agent
from agent.state import (
    TadarrujState,
    StudyPlanRequest,
    RoadmapRequest,
    ChatMessage,
)
from agent import nodes
from agent import rag as rag_module

# ── Helpers ───────────────────────────────────────────────────

SEP  = "─" * 60
SEP2 = "═" * 60


def _run(state: TadarrujState) -> TadarrujState:
    agent = build_agent()
    result = agent.invoke(state)
    return TadarrujState(**result)


def _score_plan(text: str) -> dict:
    """
    Heuristic quality metrics for a generated plan.
    Returns a dict of scores (all 0-1 or counts).
    """
    if not text:
        return {"length": 0, "has_phases": 0, "has_days": 0, "has_tips": 0, "total": 0}

    phase_keywords  = ["التعلم", "التطبيق", "التعمق", "المحاكاة", "المرحلة", "مرحلة"]
    day_keywords    = ["يوم", "أسبوع", "يومياً", "كل يوم"]
    tip_keywords    = ["نصيحة", "تذكر", "احرص", "ركّز", "ركز", "تأكد"]

    has_phases = int(any(k in text for k in phase_keywords))
    has_days   = int(any(k in text for k in day_keywords))
    has_tips   = int(any(k in text for k in tip_keywords))
    length     = min(len(text) / 800, 1.0)   # normalised, cap at 1

    total = round((has_phases + has_days + has_tips + length) / 4, 2)

    return {
        "length":     round(length, 2),
        "has_phases": has_phases,
        "has_days":   has_days,
        "has_tips":   has_tips,
        "total":      total,
    }


def _print_result(label: str, result: TadarrujState, scores: dict, elapsed: float):
    print(f"\n  [{label}]")
    print(f"  الوقت      : {elapsed:.1f}s")
    print(f"  طول الإخراج: {len(result.plan or result.roadmap or '')} حرف")
    print(f"  مراحل      : {'✓' if scores.get('has_phases') else '✗'}")
    print(f"  أيام/أسابيع: {'✓' if scores.get('has_days')   else '✗'}")
    print(f"  نصائح      : {'✓' if scores.get('has_tips')   else '✗'}")
    print(f"  جودة كلية  : {scores.get('total', 0) * 100:.0f}%")
    print(f"\n  أول 300 حرف من الناتج:")
    output = (result.plan or result.roadmap or result.error or "— لا ناتج —")
    print(f"  {output[:300].strip()}")


# ══════════════════════════════════════════════════════════════
# SECTION A — RAG vs no-RAG comparison
# ══════════════════════════════════════════════════════════════

def eval_rag_vs_no_rag():
    print(f"\n{SEP2}")
    print("  القسم أ — مقارنة الخطة: مع RAG مقابل بدون RAG")
    print(SEP2)

    base_request = StudyPlanRequest(
        subject="قدرات (كمي + لفظي)",
        days_left=30,
        chapters=10,
        hours_per_day=2.0,
        difficulty="متوسط",
        target_score=90,
    )

    # ── WITH RAG (normal flow) ────────────────────────────────
    print(f"\n{SEP}")
    print("  1) مع RAG (التدفق الكامل)")
    print(SEP)
    t0 = time.time()
    result_rag = _run(TadarrujState(request=base_request))
    elapsed_rag = time.time() - t0
    scores_rag  = _score_plan(result_rag.plan or "")
    _print_result("RAG مفعّل", result_rag, scores_rag, elapsed_rag)

    # ── WITHOUT RAG (bypass rag_node by injecting empty context) ─
    print(f"\n{SEP}")
    print("  2) بدون RAG (حقن سياق فارغ)")
    print(SEP)
    t0 = time.time()
    # Pre-fill rag_context="" so rag_node is effectively skipped
    state_no_rag = TadarrujState(request=base_request, rag_context="")
    # Jump straight to plan_generator
    result_no_rag = TadarrujState(**nodes.plan_generator_node(
        state_no_rag.model_copy(update={"agent_intent": "study_plan"})
    ).model_dump())
    elapsed_no_rag = time.time() - t0
    scores_no_rag  = _score_plan(result_no_rag.plan or "")
    _print_result("بدون RAG", result_no_rag, scores_no_rag, elapsed_no_rag)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ملخص المقارنة")
    print(SEP)
    diff = scores_rag["total"] - scores_no_rag["total"]
    print(f"  جودة مع RAG   : {scores_rag['total']*100:.0f}%")
    print(f"  جودة بدون RAG : {scores_no_rag['total']*100:.0f}%")
    if diff > 0:
        print(f"  ✓ RAG حسّن الجودة بمقدار {diff*100:.0f} نقطة")
    elif diff == 0:
        print("  = جودة متساوية (قد تكون قاعدة المعرفة فارغة)")
    else:
        print(f"  ✗ RAG لم يحسّن الجودة في هذه الحالة ({diff*100:.0f} نقطة)")


# ══════════════════════════════════════════════════════════════
# SECTION B — Error handling tests
# ══════════════════════════════════════════════════════════════

def eval_error_handling():
    print(f"\n{SEP2}")
    print("  القسم ب — اختبار معالجة الأخطاء")
    print(SEP2)

    test_cases = [
        # (label, state, expected_behaviour)
        (
            "طلب بدون request أو roadmap_request",
            TadarrujState(),
            "يجب أن يُكمل بدون crash — fallback لـ study_plan",
        ),
        (
            "رسالة chat بدون خطة موجودة",
            TadarrujState(
                chat_history=[ChatMessage(role="user", content="عدّل خطتي")]
            ),
            "يجب أن يرد بدون crash",
        ),
        (
            "roadmap_request بحقول فارغة",
            TadarrujState(
                roadmap_request=RoadmapRequest(
                    grade=None,
                    major_interest=None,
                    target_university=None,
                )
            ),
            "يجب أن يُنشئ خارطة مع قيم افتراضية 'غير محدد'",
        ),
        (
            "أيام متبقية = 1 (حالة حدية)",
            TadarrujState(
                request=StudyPlanRequest(
                    subject="رياضيات",
                    days_left=1,
                    hours_per_day=1.0,
                    difficulty="صعب",
                    target_score=80,
                )
            ),
            "يجب أن يُنشئ خطة رغم ضيق الوقت",
        ),
        (
            "درجة مستهدفة = 1 (أدنى قيمة)",
            TadarrujState(
                request=StudyPlanRequest(
                    subject="IELTS",
                    days_left=14,
                    hours_per_day=3.0,
                    difficulty="سهل",
                    target_score=1,
                )
            ),
            "يجب أن يُنشئ خطة بدون خطأ",
        ),
    ]

    passed = 0
    failed = 0

    for i, (label, state, expectation) in enumerate(test_cases, 1):
        print(f"\n{SEP}")
        print(f"  اختبار {i}: {label}")
        print(f"  المتوقع: {expectation}")
        print(SEP)

        try:
            t0     = time.time()
            result = _run(state)
            elapsed = time.time() - t0

            crashed = False
            has_output = bool(result.plan or result.roadmap or result.chat_history)
            has_error  = bool(result.error)

            status = "✓ نجح" if has_output and not has_error else (
                     "⚠ اكتمل مع خطأ" if has_error else
                     "✗ لا ناتج")

            if "نجح" in status:
                passed += 1
            else:
                failed += 1

            print(f"  النتيجة  : {status}")
            print(f"  الوقت    : {elapsed:.1f}s")
            print(f"  خطأ      : {result.error or '—'}")
            output = result.plan or result.roadmap or (
                result.chat_history[-1].content if result.chat_history else "—"
            )
            print(f"  أول 200 حرف: {str(output)[:200].strip()}")

        except Exception as e:
            failed += 1
            print(f"  ✗ CRASH: {e}")

    print(f"\n{SEP2}")
    print(f"  نتائج اختبارات الأخطاء: {passed} نجح / {failed} فشل من {len(test_cases)}")
    print(SEP2)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(SEP2)
    print("  Tadarruj — تقييم الجودة ومعالجة الأخطاء")
    print(SEP2)

    eval_rag_vs_no_rag()
    eval_error_handling()

    print(f"\n{SEP2}")
    print("  اكتمل التقييم")
    print(SEP2)