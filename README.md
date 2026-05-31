# تدرّج — Tadarruj Academic Agent

> **"طريقك الهادئ نحو النجاح"** — Your calm path to success.

Tadarruj is an AI-powered academic guidance system for Saudi high school students. It takes a student's exam date, daily study hours, and target score, then generates a structured, personalized Arabic study plan — and lets the student refine it through a live chat interface.

---

## Project Overview

| Item | Details |
|---|---|
| **Phase** | Phase 1 — Study Planning Engine |
| **Model** | Qwen2.5-7B-Instruct fine-tuned with QLoRA via Unsloth |
| **Agent** | LangGraph StateGraph + Pydantic v2 state management |
| **Interface** | Streamlit (Arabic RTL, Cairo font) |
| **Evaluation** | LLM-as-judge with DeepSeek API |
| **Target users** | Saudi high school students preparing for Qudurat, Tahsili, SAT, STEP, IELTS |

---

## Features

- **Structured form** — collects exam type, date, content volume, daily hours, difficulty, and target score
- **AI plan generation** — fine-tuned SLM produces a 4-phase Arabic study plan (Learn → Practice → Apply → Simulate)
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

## Fine-Tuning

**Model:** `Qwen/Qwen2.5-7B-Instruct`  
**Method:** QLoRA (4-bit quantization, r=16, alpha=16)  
**Library:** [Unsloth](https://github.com/unslothai/unsloth) — requires a CUDA GPU  
**Hardware:** Tesla T4 (Lightning AI)


**Training results:**
- Initial loss: `2.038`
- Final loss: `0.990`
- Steps: 69 | Epochs: 3 | Duration: ~9 minutes

---

## Evaluation

Evaluation uses **DeepSeek** as an LLM-as-judge, scoring each generated plan across 5 criteria:

| Criterion | Description |
|---|---|
| Relevance | Plan matches the student's subject and inputs |
| Structure | Clear 4-phase structure |
| Tone | Encouraging and motivating Arabic |
| Numerical Accuracy | Hours and days add up correctly |
| Safety | No harmful or misleading advice |

**Results summary** (3 test cases):

| Criterion | Avg Score (out of 5) |
|---|---|
| Relevance | 4.00 |
| Structure | 3.67 |
| Tone | 4.67 |
| Numerical Accuracy | 2.00 |
| Safety | 4.67 |
| **Overall** | **3.80** |

Tone and safety were the strongest dimensions. Numerical accuracy was the weakest — a known limitation of small language models on arithmetic, mitigated in production by the structured form (hours × days is computed in Python, not by the model).

---

## Agent Architecture

```
User Input (StudyPlanRequest)
        │
        ▼
  plan_generator_node          ← fine-tuned Qwen2.5 via Unsloth
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

# 2. Install dependencies (CPU)
pip install -r requirements.txt

# 3. Install Unsloth on a GPU machine
pip install unsloth

# 4. Add your API key
cp .env.example .env
# edit .env and add your DEEPSEEK_API_KEY

# 5. Run the app
streamlit run app/app.py --server.port 8080
```

> **Note:** The fine-tuned model weights are not included in the repo. Run `training/train.py` on a GPU machine to generate them, or load any Qwen2.5-7B-Instruct compatible adapter.

---

## Requirements

See `requirements.txt` for the full list. 

---

## Roadmap

- [x] **Phase 1** — Study planning engine with fine-tuned SLM + Streamlit app
- [ ] **Phase 2** — Multi-agentic workflow (LangGraph) with RAG, multi-turn memory, Pydantic state, error handling, and agent tools