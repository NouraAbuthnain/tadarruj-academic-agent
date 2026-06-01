# تدرّج — Tadarruj Academic Agent

> **"طريقك الهادئ نحو النجاح"** — Your calm path to success.

Tadarruj is an AI-powered academic guidance system for Saudi high school students. It takes a student's exam date, daily study hours, and target score, then generates a structured, personalized Arabic study plan — and lets the student refine it through a live chat interface. Phase 2 adds a full multi-agent LangGraph workflow with RAG, a 3-year academic roadmap generator, and robust error handling.

---

## Project Overview

| Item | Details |
|---|---|
| **Phase** | Phase 2 — Multi-Agentic Workflow with RAG |
| **LLM** | Google Gemma 4 31B |
| **Fine-tuning experiment** | Qwen2.5-7B-Instruct with QLoRA via Unsloth (see Fine-Tuning section) |
| **Agent** | LangGraph StateGraph (6 nodes) + Pydantic v2 state management |
| **RAG** | Contextual Compression — ChromaDB + multilingual sentence-transformers |
| **Interface** | Streamlit (Arabic RTL, Cairo font) |
| **Evaluation** | RAG vs no-RAG quality comparison + error handling tests |
| **Target users** | Saudi high school students preparing for Qudurat, Tahsili, SAT, STEP, IELTS |

**Live demo:** [tadarruj-academic-agent.streamlit.app](https://tadarruj-academic-agent.streamlit.app)

---

## Features

- **Mode selection** — Study Plan or 3-Year Academic Roadmap
- **Structured form** — collects exam type, date, content volume, daily hours, difficulty, and target score
- **AI plan generation** — LLM produces a 4-phase Arabic study plan (Learn → Practice → Apply → Simulate)
- **Academic roadmap** — personalised 3-year plan with exam prep timelines, university admission requirements, volunteering hours, and GPA advice
- **Contextual Compression RAG** — retrieves and compresses relevant knowledge from a curated Arabic knowledge base
- **Progress tracking** — log study hours daily; see total, completed, and remaining hours with a progress bar
- **Chat recalibration** — ask the agent to adjust the plan if days were missed or circumstances changed
- **Error handling with retry** — automatic retries up to 2× before graceful Arabic fallback message
- **Custom subject** — free-text input when the exam is not in the predefined list

---

## Project Structure

```
tadarruj/
├── agent/
│   ├── __init__.py       # Public API: generate_plan, generate_roadmap, chat_reply
│   ├── graph.py          # LangGraph graph definition and routing
│   ├── nodes.py          # Node functions (intent, RAG, plan, roadmap, chat, error)
│   ├── rag.py            # Contextual Compression RAG + LangChain tool wrappers
│   └── state.py          # Pydantic models: TadarrujState, StudyPlanRequest, RoadmapRequest
├── app/
│   └── app.py            # Streamlit UI (mode selection, forms, chat, progress tracking)
├── data/
│   ├── datasets/
│   │   └── tadarruj_dataset_200.jsonl
│   └── knowledge/        # .txt knowledge files indexed by RAG at startup
├── assets/
│   └── page_icon.svg
├── training/
│   └── train.py          # QLoRA fine-tuning script (GPU only)
├── eval.py               # Evaluation script: RAG vs no-RAG + error handling tests
├── .env                  # API keys (not committed)
└── requirements.txt
```

---

## Agent Architecture

### Phase 2 — Multi-Agent Workflow

```
User Input
    │
    ▼
intent_node          ← classifies: study_plan | roadmap | recalibration | question
    │
    ▼
rag_node             ← Contextual Compression RAG from ChromaDB knowledge base
    │
    ├──► plan_generator_node    ← 4-phase Arabic study plan
    ├──► roadmap_node           ← 3-year academic roadmap
    └──► recalibration_node     ← chat follow-up with multi-turn memory
              │
              ▼ (on any error)
        error_handler_node      ← retry up to 2×, then Arabic fallback message
```

**RAG technique:** Contextual Compression — retrieved chunks are scored by query-term overlap and trimmed to the most relevant sentences before being injected into the LLM prompt.

State is managed with **Pydantic v2** models (`StudyPlanRequest`, `RoadmapRequest`, `TadarrujState`, `ChatMessage`), ensuring type safety across the entire graph.

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

### Phase 1 — LLM-as-Judge (DeepSeek)

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

### Phase 2 — RAG vs no-RAG + Error Handling

Run the evaluation script from the project root:

```bash
python eval.py | tee eval_results.txt
```

**Section A — RAG vs no-RAG:** generates a study plan twice (with full RAG context, and with empty context), scores each on 4 heuristics (phase structure, time references, study tips, output length), and reports a quality percentage difference.

**Section B — Error handling tests:** runs 5 edge cases through the full graph — empty state, chat without a plan, null roadmap fields, 1-day deadline, and minimum target score — verifying crash-free completion and graceful error messages.

---

## Running the App

```bash
# 1. Clone the repo
git clone https://github.com/NouraAbuthnain/tadarruj-academic-agent.git
cd tadarruj-academic-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# edit .env and add your OPENROUTER_API_KEY

# 4. Add knowledge files to data/knowledge/ (plain .txt files)

# 5. Run the app
streamlit run app/app.py --server.port 8080
```

**To run fine-tuning on a GPU machine:**
```bash
pip install unsloth
python training/train.py
```

> **Note:** The fine-tuned model weights are not included in the repo. Run `training/train.py` on a GPU machine to generate them, or load any Qwen2.5-7B-Instruct compatible adapter.

---

## Configuration

| Setting | File | Default |
|---|---|---|
| LLM model | `agent/nodes.py` → `MODEL` | `google/gemma-4-31b-it:free` |
| Embedding model | `agent/rag.py` → `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` |
| Chunk size | `agent/rag.py` → `CHUNK_SIZE` | `500` characters |
| Max retries | `agent/nodes.py` → `error_handler_node` | `2` |
| RAG results | `agent/rag.py` → `search()` | `n_results=4` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent orchestration | LangGraph |
| State management | Pydantic v2 |
| RAG vector store | ChromaDB (in-memory) |
| Embeddings | sentence-transformers |
| LLM API | OpenRouter (OpenAI-compatible) |
| Tool interface | LangChain Core (`@tool`) |

---

## Roadmap

- [x] **Phase 1** — Study planning engine with API-powered LLM + Streamlit app
- [x] **Phase 1 (ongoing)** — Fine-tuning experiments (Qwen2.5-7B, QLoRA via Unsloth)
- [x] **Phase 2** — Multi-agentic workflow (LangGraph) with RAG, multi-turn memory, Pydantic state, error handling, agent tools, and 3-year academic roadmap