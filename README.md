# تدرّج — Tadarruj Academic Agent

> **"طريقك الهادئ نحو النجاح"** — Your calm path to success.

Tadarruj is an AI-powered academic guidance system for Saudi high school students. It takes a student's exam date, daily study hours, and target score, then generates a structured, personalized Arabic study plan — and lets the student refine it through a live chat interface.

---

## Project Overview

| Item | Details |
|---|---|
| **Phase** | Phase 1 — Study Planning Engine |
| **LLM** | Google Gemma 4 31B via OpenRouter API |
| **Fine-tuning experiment** | Qwen2.5-7B-Instruct with QLoRA via Unsloth (see Fine-Tuning section) |
| **Agent** | LangGraph StateGraph + Pydantic v2 state management |
| **Interface** | Streamlit (Arabic RTL, Cairo font) |
| **Evaluation** | LLM-as-judge with DeepSeek API |
| **Target users** | Saudi high school students preparing for Qudurat, Tahsili, SAT, STEP, IELTS |

**Live demo:** [tadarruj-academic-agent.streamlit.app](https://tadarruj-academic-agent.streamlit.app)
---

## Features

- **Structured form** — collects exam type, date, content volume, daily hours, difficulty, and target score
- **AI plan generation** — LLM produces a 4-phase Arabic study plan (Learn → Practice → Apply → Simulate)
- **Progress tracking** — log study hours daily; see total, completed, and remaining hours with a progress bar
- **Chat recalibration** — ask the agent to adjust the plan if days were missed or circumstances changed
- **Custom subject** — free-text input when the exam is not in the predefined list

---

## Dataset

The training dataset (`data/datasets/tadarruj_dataset_200.jsonl`) contains **200 Arabic instruction-response pairs** in chat format. Each sample includes:

- A structured student request (subject, days left, hours/day, difficulty, target score)
- A 4-phase Arabic study plan response covering: التعلم (Learn) → التطبيق (Practice) → التعمق (Apply) → المحاكاة (Simulate)

The dataset was built specifically for this project to reflect real Saudi student exam contexts (Qudurat, Tahsili, SAT, STEP, IELTS).

---

## Fine-Tuning Experiment

> **Note:** This is an ongoing experiment. The fine-tuned model is not used in the current deployment — the app uses the OpenRouter API instead.

**Model:** `Qwen/Qwen2.5-7B-Instruct`  
**Method:** QLoRA (4-bit quantization, r=16, alpha=16)  
**Library:** [Unsloth](https://github.com/unslothai/unsloth) — requires a CUDA GPU  
**Hardware:** Tesla T4 (Lightning AI)

**Training results:**
- Initial loss: `2.038` → Final train loss: `1.368`
- Best step loss: `0.990` (step 60, epoch 2.62)
- Eval loss: `1.476`
- Steps: 69 | Epochs: 3 | Duration: ~9 minutes

---

## Evaluation

Evaluation uses **DeepSeek** as an LLM-as-judge, scoring each generated plan across 5 criteria:

| Criterion | Description |
|---|---|
| Structured Planning | Clear 4-phase structure with day counts |
| Numerical Accuracy | Hours and days add up correctly |
| Calm Supportive Tone | Encouraging and motivating Arabic |
| Actionability | Concrete, executable daily steps |
| Safety | No harmful or misleading advice |

**Results summary** (3 test cases):

| Criterion | Avg Score (out of 5) |
|---|---|
| Structured Planning | 2.67 |
| Numerical Accuracy | 2.00 |
| Calm Supportive Tone | 4.67 |
| Actionability | 3.33 |
| Safety | 4.67 |
| **Pass rate** | **0.67 (2/3)** |

Tone and safety were the strongest dimensions — the two core Tadarruj values. Numerical accuracy was the weakest (2.0/5), seen in Sample 2 where `1.5h` was mislabeled as `ساعتان ونصف`. Sample 3 failed because the student sent an emotional free-text message with no structured inputs — out-of-scope in production since the Streamlit form enforces all required fields before generation.

---

## Agent Architecture

```
User Input (StudyPlanRequest)
        │
        ▼
  plan_generator_node          ← Gemma 4 31B via OpenRouter API
        │
        ▼
  TadarrujState (plan + history)
        │
    [chat input]
        │
        ▼
  chat_reply()                 ← recalibration_node called directly
        │
        ▼
  Updated TadarrujState
```

State is managed with **Pydantic v2** models (`StudyPlanRequest`, `TadarrujState`, `ChatMessage`), ensuring type safety across the agent graph.

---

## Running the App

```bash
# 1. Clone the repo and switch to phase-1
git clone https://github.com/NouraAbuthnain/tadarruj-academic-agent.git
cd tadarruj-academic-agent
git checkout phase-1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys
cp .env.example .env
# edit .env and add your OPENROUTER_API_KEY and DEEPSEEK_API_KEY

# 4. Run the app
streamlit run app/app.py --server.port 8080
```

**To run fine-tuning on a GPU machine:**
```bash
pip install unsloth
python training/train.py
```

> **Note:** The fine-tuned model weights are not included in the repo. Run `training/train.py` on a GPU machine to generate them, or load any Qwen2.5-7B-Instruct compatible adapter.

---

## Requirements

See `requirements.txt` for the full list.

---

## Roadmap

- [x] **Phase 1** — Study planning engine with API-powered LLM + Streamlit app
- [ ] **Phase 1 (ongoing)** — Fine-tuning experiments (Qwen3-8B, more parameters, QLoRA vs PEFT)
- [ ] **Phase 2** — Multi-agentic workflow (LangGraph) with RAG, multi-turn memory, Pydantic state, error handling, and agent tools