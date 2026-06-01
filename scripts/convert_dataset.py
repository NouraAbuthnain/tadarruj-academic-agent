"""
convert_dataset.py
Converts tadarruj_dataset_200.jsonl (Alpaca format) -> tadarruj_chat_format.jsonl (Qwen chat template).
Run once before training:
    python scripts/convert_dataset.py
"""

import json
from pathlib import Path

SYSTEM_PROMPT = """أنت تدرّج، مساعد أكاديمي موثوق ومتخصص في توجيه طلاب المرحلة الثانوية السعودية (الصف العاشر والحادي عشر والثاني عشر). مهمتك تحويل أهداف الطالب إلى خطط واضحة وقابلة للتنفيذ.

مبادئك الأساسية:
- أسلوبك هادئ وداعم ومنظم، لا تخويف ولا ضغط.
- لا تقل أبداً 'أنت متأخر' أو 'فات الأوان' — بل قل 'يمكننا التعديل والمتابعة'.
- إذا سألك الطالب عن معلومة متغيرة (مواعيد، نسب قبول، أسعار)، أحله للمصدر الرسمي.
- إذا كان السؤال غير واضح، اطلب التفاصيل بدلاً من التخمين.
- ركّز دائماً على الخطوة التالية القابلة للتنفيذ.""".strip()

ROOT = Path("data/datasets/tadarruj_dataset_200.jsonl")
OUT = Path("data/datasets/tadarruj_chat_format.jsonl")

def convert():
    if not ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {ROOT}\nRun from the tadarruj/ project root.")

    chat_examples, seen = [], set()

    with ROOT.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                print(f"  Skipping row {i}: invalid JSON")
                continue
            if "instruction" not in item or "output" not in item:
                print(f"  Skipping row {i}: missing fields")
                continue
            instruction = str(item["instruction"]).strip()
            output      = str(item["output"]).strip()
            if not instruction or not output:
                print(f"  Skipping row {i}: empty fields")
                continue
            key = (instruction, output)
            if key in seen:
                print(f"  Skipping row {i}: duplicate")
                continue
            seen.add(key)
            chat_examples.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": instruction},
                    {"role": "assistant", "content": output},
                ]
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for ex in chat_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Done. Valid examples: {len(chat_examples)}")
    print(f"Saved -> {OUT}")
    print("\nPreview:")
    print(json.dumps(chat_examples[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    convert()
