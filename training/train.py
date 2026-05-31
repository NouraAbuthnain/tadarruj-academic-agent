"""
train.py  —  Tadarruj Phase 1 Fine-Tuning
Trains Qwen2.5-7B-Instruct with QLoRA (Unsloth) on tadarruj_chat_format.jsonl.

Run from the tadarruj/ project root:
    python training/train.py

Prerequisites:
    1. pip install -r requirements.txt
    2. python scripts/convert_dataset.py   (generates tadarruj_chat_format.jsonl)
"""

import os, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH    = os.path.join("data", "datasets", "tadarruj_chat_format.jsonl")
SAVE_PATH       = os.path.join("models", "tadarruj_qwen2_5_7b_lora")
MAX_SEQ_LENGTH  = 2048
LOAD_IN_4BIT    = True

# LoRA
LORA_R          = 16
LORA_ALPHA      = 16
LORA_DROPOUT    = 0

# Training
BATCH_SIZE      = 2
GRAD_ACCUM      = 4          # effective batch = 8
NUM_EPOCHS      = 3
LEARNING_RATE   = 2e-4
WARMUP_STEPS    = 10
SEED            = 3407

SYSTEM_PROMPT = """أنت تدرّج، مساعد أكاديمي موثوق ومتخصص في توجيه طلاب المرحلة الثانوية السعودية (الصف العاشر والحادي عشر والثاني عشر). مهمتك تحويل أهداف الطالب إلى خطط واضحة وقابلة للتنفيذ.

مبادئك الأساسية:
- أسلوبك هادئ وداعم ومنظم، لا تخويف ولا ضغط.
- لا تقل أبداً 'أنت متأخر' أو 'فات الأوان' — بل قل 'يمكننا التعديل والمتابعة'.
- إذا سألك الطالب عن معلومة متغيرة (مواعيد، نسب قبول، أسعار)، أحله للمصدر الرسمي.
- إذا كان السؤال غير واضح، اطلب التفاصيل بدلاً من التخمين.
- ركّز دائماً على الخطوة التالية القابلة للتنفيذ.""".strip()


# ── Helpers ─────────────────────────────────────────────────────
def find_subsequence(seq, sub):
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i:i+m] == sub:
            return i
    return -1


def formatting_prompts_func(examples, tokenizer, max_seq_length):
    all_input_ids, all_attention_masks, all_labels = [], [], []
    assistant_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    for messages_list in examples["messages"]:
        input_ids = tokenizer.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        labels = list(input_ids)
        asst_idx = find_subsequence(input_ids, assistant_tokens)
        if asst_idx != -1:
            mask_end = asst_idx + len(assistant_tokens)
            for i in range(mask_end):
                labels[i] = -100
        else:
            labels = [-100] * len(input_ids)

        all_input_ids.append(input_ids)
        all_attention_masks.append([1] * len(input_ids))
        all_labels.append(labels)

    return {
        "input_ids":       all_input_ids,
        "attention_mask":  all_attention_masks,
        "labels":          all_labels,
    }

def run_inference(model, tokenizer, user_message, max_new_tokens=400, label=""):
    """Helper to run a quick inference check."""
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=False,
        truncation=True, max_length=MAX_SEQ_LENGTH
    ).to("cuda")

    # DELETE the attention_mask line — remove this:
    # attention_mask = (inputs["input_ids"] != tokenizer.eos_token_id).long()

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3, top_p=0.8, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[-1]:]
    response  = tokenizer.decode(generated, skip_special_tokens=True)
    if label:
        print(f"\n{'='*20} {label} {'='*20}")
    print(response)
    return response


# ── 1. Load Model ───────────────────────────────────────────────
print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# ── 2. Baseline Inference ───────────────────────────────────────
print("\n[Baseline — before fine-tuning]")
FastLanguageModel.for_inference(model)
BASELINE_PROMPT = (
    "المادة: قدرات (كمي + لفظي). الأيام المتبقية: 60 يوماً. "
    "عدد الفصول: 20 فصلاً. ساعات الدراسة اليومية: 2 ساعة. "
    "مستوى الصعوبة: متوسط. الدرجة المستهدفة: 90."
)
run_inference(model, tokenizer, BASELINE_PROMPT, label="BEFORE TRAINING")

# ── 3. Add QLoRA Adapters ───────────────────────────────────────
print("\nAdding QLoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
)

# ── 4. Load & Format Dataset ────────────────────────────────────
print(f"\nLoading dataset from {DATASET_PATH}...")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Run: python scripts/convert_dataset.py"
    )

raw = load_dataset("json", data_files=DATASET_PATH, split="train")
raw = raw.shuffle(seed=SEED)
split = raw.train_test_split(test_size=0.1, seed=SEED)

map_fn = lambda x: formatting_prompts_func(x, tokenizer, MAX_SEQ_LENGTH)

train_dataset = split["train"].map(map_fn, batched=True, remove_columns=split["train"].column_names)
eval_dataset  = split["test"].map( map_fn, batched=True, remove_columns=split["test"].column_names)

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

# Quick debug — verify labels are masked correctly
sample = train_dataset[0]
visible = tokenizer.decode([t for t in sample["labels"] if t != -100], skip_special_tokens=False)
print("\nVisible training target (first example):")
print(visible[:300])

# ── 5. Trainer ──────────────────────────────────────────────────
os.makedirs(SAVE_PATH, exist_ok=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        output_dir=os.path.join("outputs", "tadarruj-checkpoints"),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=SEED,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    ),
)

# ── 6. Train ────────────────────────────────────────────────────
print("\nStarting training...")
trainer_stats = trainer.train()
print(f"\nTraining complete. Loss: {trainer_stats.training_loss:.4f}")

# ── 7. Save ─────────────────────────────────────────────────────
print(f"\nSaving model to {SAVE_PATH}...")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("Saved.")

# ── 8. Post-Training Inference ──────────────────────────────────
print("\n[Post-training check]")
FastLanguageModel.for_inference(model)
run_inference(model, tokenizer, BASELINE_PROMPT, label="AFTER TRAINING")

print("\nDone. Model saved to:", SAVE_PATH)
