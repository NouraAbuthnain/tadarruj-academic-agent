"""
evaluate.py  —  LLM-as-Judge evaluation using DeepSeek API
Run after training:
    python scripts/evaluate.py

Results saved to: outputs/evaluation_results.json
"""

import os, sys, json, torch
from pathlib import Path
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root

from openai import OpenAI
from agent import load_model
from agent.nodes import _generate, SYSTEM_PROMPT, set_model

# ── Config ──────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OUT_PATH = Path("outputs/evaluation_results.json")

TEST_CASES = [
    (
        "المادة: قدرات (كمي + لفظي). الأيام المتبقية: 60 يوماً. "
        "عدد الفصول: 20 فصلاً. ساعات الدراسة اليومية: 2 ساعة. "
        "مستوى الصعوبة: متوسط. الدرجة الحالية: 72. الدرجة المستهدفة: 90."
    ),
    (
        "المادة: تحصيلي علمي. الأيام المتبقية: 45 يوماً. "
        "إجمالي ساعات المحتوى: 20 ساعة. ساعات الدراسة اليومية: 1.5 ساعة. "
        "مستوى الصعوبة: صعب. الدرجة المستهدفة: 88."
    ),
    (
        "فاتتني ثلاثة أيام من خطة القدرات وأحس أني خربت كل شيء. "
        "عندي 25 يوم باقي وساعتين يومياً."
    ),
]

JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for Tadarruj, an academic guidance assistant for Saudi high school students.

Evaluate the assistant response on these 5 criteria (score 1–5 each):

1. structured_planning_score   — Does the plan have clear phases (Learning/Practice/Revision/Mock)?
                                  Do day counts add up to the total days given?
2. numerical_accuracy_score    — Does the plan use the exact numbers from the student input
                                  (days, hours, chapters)?
3. calm_supportive_tone_score  — Calm, supportive Arabic. No fear/shame. No "you are late" framing.
4. actionability_score         — Are steps concrete and executable (not just motivational)?
5. safety_score                — No invented deadlines, no uncertain facts stated as certain.

Return ONLY valid JSON:
{
  "structured_planning_score": 0,
  "numerical_accuracy_score": 0,
  "calm_supportive_tone_score": 0,
  "actionability_score": 0,
  "safety_score": 0,
  "final_verdict": "pass or fail",
  "short_reason": "brief explanation"
}
""".strip()


def generate_response(student_request: str, max_new_tokens: int = 600) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": student_request},
    ]
    return _generate(messages, max_new_tokens=max_new_tokens)


def judge(client: OpenAI, request: str, answer: str) -> dict:
    prompt = f"Student Request:\n{request}\n\nAssistant Answer:\n{answer}"

    messages = cast(
        list[ChatCompletionMessageParam],
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0,
        stream=False,
    )

    content = resp.choices[0].message.content or ""
    content = content.strip()

    # Strip markdown code fences if present
    for fence in ["```json", "```"]:
        content = content.replace(fence, "")

    try:
        return json.loads(content.strip())
    except Exception:
        return {"error": "Invalid JSON", "raw": content}

def average(lst):
    return round(sum(lst) / len(lst), 2) if lst else 0


def main():
    if not DEEPSEEK_API_KEY:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable.")
        sys.exit(1)

    print("Loading model...")
    load_model()

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    results = []
    for i, request in enumerate(TEST_CASES, start=1):
        print(f"\n=== Sample {i}/{len(TEST_CASES)} ===")
        print("Request:", request[:100], "...")

        answer = generate_response(request)
        print("Answer:", answer[:150], "...")

        verdict = judge(client, request, answer)
        print("Judge:", verdict)

        results.append({
            "sample_id":       i,
            "student_request": request,
            "model_answer":    answer,
            "judge":           verdict,
        })

    # Summary
    keys = ["structured_planning_score", "numerical_accuracy_score",
            "calm_supportive_tone_score", "actionability_score", "safety_score"]
    summary = {f"avg_{k}": average([r["judge"].get(k, 0) for r in results]) for k in keys}
    summary["pass_rate"] = round(
        sum(r["judge"].get("final_verdict") == "pass" for r in results) / len(results), 2
    )
    summary["num_samples"] = len(results)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
