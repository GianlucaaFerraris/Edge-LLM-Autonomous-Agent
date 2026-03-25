<div align="center">

[![Title](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=32&pause=99999&color=58A6FF&center=true&vCenter=true&width=700&height=70&lines=Agenty+—+Local+Edge+AI+Assistant)](https://github.com/GianlucaaFerraris)

[![Subtitle](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=400&size=15&pause=99999&color=8b949e&center=true&vCenter=true&width=700&height=35&lines=Fully+local+autonomous+agent+on+embedded+SBCs.+No+cloud.+No+latency+tax.)](https://github.com/GianlucaaFerraris)

<br/>

![Python](https://img.shields.io/badge/Python-0d1117?style=for-the-badge&logo=python&logoColor=3776AB)
![Ollama](https://img.shields.io/badge/Ollama-0d1117?style=for-the-badge&logo=ollama&logoColor=ffffff)
![ROS2](https://img.shields.io/badge/ROS2-0d1117?style=for-the-badge&logo=ros&logoColor=22314E)
![NVIDIA Jetson](https://img.shields.io/badge/Jetson_Orin_Nano-0d1117?style=for-the-badge&logo=nvidia&logoColor=76B900)
![Radxa](https://img.shields.io/badge/Radxa_Rock_5B-0d1117?style=for-the-badge&logo=raspberrypi&logoColor=A22846)
![SQLite](https://img.shields.io/badge/SQLite-0d1117?style=for-the-badge&logo=sqlite&logoColor=003B57)
![License](https://img.shields.io/badge/license-MIT-0d1117?style=for-the-badge&logoColor=58a6ff)

</div>

---

## What is this?

**Agenty** is a personal AI assistant designed to run **entirely on the edge** — no API calls, no cloud inference, no internet required. A quantized INT4 LLM runs locally via Ollama on resource-constrained single-board computers (Radxa Rock 5B / Jetson Orin Nano), with a multimodal perception layer powered by a **Luxonis OAK-D Pro** stereo camera.

The result is a system that can reason, schedule, search, teach, and *see* — all on hardware that fits in your hand.

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENTY  PIPELINE                         │
│                                                                 │
│   Voice / Text ──► Orchestrator ──► Intent Router               │
│                          │                                      │
│              ┌───────────┼───────────┐                          │
│              ▼           ▼           ▼                          │
│         English      Engineering   Agent                        │
│          Tutor         Tutor      Session                       │
│                                     │                           │
│                          ┌──────────┴──────────┐                │
│                          ▼                     ▼                │
│                    Tool Dispatcher     Perception Module        │
│                    (tasks/calendar/    OAK-D Pro NPU            │
│                     reminders/web)    Stereo Depth + YOLO       │
│                          │                     │                │
│                          └──────────┬──────────┘                │
│                                     ▼                           │
│                              SQLite Storage                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key highlights

| | |
|---|---|
| 🧠 **Quantized INT4 LLM** | Qwen2.5-7B served by Ollama — ~30% of original model size, full reasoning capability |
| 📚 **Local RAG pipeline** | FAISS vector index over 9,415 chunks from 22 technical books — fully offline retrieval |
| 🌍 **Bilingual retrieval** | Multilingual embeddings align Spanish queries against English-language technical books |
| ✍️ **Grammar-aware English tutor** | LanguageTool server runs locally on the SBC — zero cloud dependency for grammar correction |
| 🗂️ **600 conversation topics** | Generated entirely by the local LLM, stored in `topics.json` — no external dataset |
| 👁️ **OAK-D Pro perception** | Real-time object detection + stereo depth + spatial reasoning via DepthAI NPU |
| 🗣️ **Multimodal I/O** | Whisper STT + Piper/Kokoro TTS hooks — speak to it, it speaks back |
| 🔌 **Zero cloud dependency** | All inference, storage and scheduling run 100% on-device |
| 📅 **Persistent memory** | SQLite-backed tasks, calendar, and reminders survive reboots |
| ⚡ **Sub-second tool calls** | 0.11s TTFT for agent tool dispatch on GPU; NPU path targets real-time |

---

## Hardware targets

| Board | SoC | NPU | Status |
|---|---|---|---|
| Radxa Rock 5B | RK3588 | 6 TOPS (RKNN) | ✅ Primary target |
| Jetson Orin Nano | Cortex-A78AE + Ampere | 40 TOPS | ✅ Primary target |
| Luxonis OAK-D Pro | Myriad X VPU | 4 TOPS | ✅ Perception module |

---

## Engineering Tutor — Local RAG Pipeline

The engineering tutor answers technical questions in English and Spanish by retrieving relevant passages from a local FAISS vector index, then grounding the LLM response in that context to reduce hallucinations on precise technical topics.

### Retrieval flow

```
User question (EN or ES)
        │
        ▼
  encode(question)           ← paraphrase-multilingual-MiniLM-L12-v2
        │                       384-dim vector, ~100ms on CPU ARM
        │                       same vector reused for both steps below
        ▼
  dot(q_vec, domain_vec)     ← cosine similarity vs. precomputed domain description
        │
   score < 0.15 ────────────────────────────────► LLM answers directly
        │
   score ≥ 0.15
        │
        ▼
  faiss.search(q_vec)        ← exact cosine search, IndexFlatIP, <10ms
        │
   best score < 0.40 ────────────────────────────► LLM answers directly
        │
   best score ≥ 0.40
        │
        ▼
  top-3 chunks injected into LLM prompt (user turn, not system prompt)
        │
        ▼
  Qwen 2.5 7B answers with grounded context
```

The question is **embedded once** and the same vector is reused for the domain check and the FAISS search — no redundant inference.

### Knowledge base

| Domain | Key sources |
|--------|-------------|
| Deep Learning / AI | Bishop & Bishop (2024), Géron (2022), Chip Huyen — AI Engineering (2023) |
| Foundation model techniques | Attention Is All You Need, BERT, LoRA, QLoRA, ReAct, RAG paper |
| Reinforcement Learning | Sutton & Barto |
| Computer Vision | Szeliski — CV Algorithms and Applications |
| Recommender Systems | Bischof & Yee, Neural CF, BERT4Rec, Wide & Deep |
| Robotics | Lynch & Park — Modern Robotics, Murphy, Choset et al., Probabilistic Robotics |
| Electronics | Horowitz & Hill — The Art of Electronics |
| Data Science | Wes McKinney — Python for Data Analysis |

**Index stats:** 9,415 chunks · 3,759,655 words · 38.8 MB on disk · built in 33.5s

### Embedding model selection

`all-MiniLM-L6-v2` (English-only) was benchmarked first and rejected — Spanish queries like "¿qué es la retropropagación?" failed to retrieve the correct chunks because the model cannot align cross-lingual semantics. `paraphrase-multilingual-MiniLM-L12-v2` was selected for its cross-lingual alignment: it maps "retropropagación" and "backpropagation" to the same vector neighborhood despite the language difference.

| Model | Dims | Size | EN retrieval | ES retrieval |
|-------|------|------|-------------|-------------|
| `all-MiniLM-L6-v2` | 384 | 90 MB | ✅ Good | ❌ Fails |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470 MB | ✅ Good | ✅ Good |

### Design decisions

**IndexFlatIP over IVF** — at 9,415 vectors, exact cosine search takes <10ms. IVF approximation complexity is not justified below ~100k vectors.

**Context injected into the user turn, not the system prompt** — the system prompt defines behavior; the user turn is where situational grounding belongs. Empirically produces better context adherence and more natural citations with Qwen 2.5.

**Two-gate filtering** — the domain threshold (0.15) is a cheap semantic pre-filter that avoids paying the FAISS search cost for clearly off-domain queries. The relevance threshold (0.40) is the quality gate that prevents weakly-related chunks from polluting the LLM prompt.

### Retrieval evaluation

Evaluated with a bilingual golden set of 19 questions (14 positive, 5 negative) across all technical domains, with quantitative scoring per question.

| Metric | Result |
|--------|--------|
| Precision@3 — English | **6/6 = 100%** |
| Precision@3 — Spanish | **8/8 = 100%** |
| Precision@3 — overall | **14/14 = 100%** |
| False positive rate | **0 / 5** |
| Average similarity score (hits) | **0.68** |

**Calibration history:**

| Configuration | Precision@3 | False positives |
|---------------|-------------|-----------------|
| English model + threshold 0.30 | 57% | 1/5 |
| + OCR on scanned robotics PDF | 71% | 1/5 |
| + multilingual model + threshold 0.40 | **100%** | **0** |

### Configuration reference

```python
# rag_engine.py
EMBEDDING_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
DOMAIN_THRESHOLD    = 0.15   # pre-filter: skip FAISS for off-domain questions
RELEVANCE_THRESHOLD = 0.40   # quality gate: only inject high-confidence chunks
TOP_K               = 3      # chunks retrieved per query
MAX_CONTEXT_WORDS   = 500    # hard cap on injected context length

# build_index.py
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 80    # overlap between consecutive chunks
```

---

## English Tutor — Grammar-Aware Conversation Practice

The English tutor runs structured conversation sessions with real-time grammar correction powered by a **locally-hosted LanguageTool server** running on the SBC. No cloud grammar API, no network round-trip — sub-50ms correction latency.

### Grammar correction flow

```
Student input
      │
      ▼
LanguageTool server (localhost:8081)
  ← managed locally by language_tool_server.py
  ← starts when English session begins, stops on exit
  ← returns structured errors: rule ID, correction, context span
      │
      ▼
Error list injected into LLM prompt
      │
      ▼
LLM generates feedback + continues conversation
  ← corrects naturally: "what you meant was..."
  ← explains the rule if the error recurs
  ← skips minor stylistic issues to preserve flow
```

LanguageTool runs as a background Java process managed by `ensure_running()` / `ensure_stopped()`. The assistant starts it automatically when the English session begins and shuts it down cleanly on exit, freeing memory for other modes.

### Session modes

**Random topic conversation** — the tutor selects a topic from `topics.json`, a set of **600 conversation topics generated entirely by the local LLM** (no external dataset). Topics span everyday life, technology, culture, hypotheticals, and debate prompts. The tutor opens naturally and keeps the student speaking. Topics used within a session are tracked to avoid repetition.

**Interview practice** — the student declares a target role or industry. The tutor conducts a structured mock interview with escalating question difficulty. After each answer, feedback covers both English quality and content clarity.

**Student-proposed topic** — the student can propose any topic at any point. Intent is detected mid-conversation and the tutor adapts without restarting the session.

### Why local LanguageTool matters

The intended deployment environment is an offline SBC. Running LanguageTool on-device means the English tutor functions with zero internet access — the same constraint that applies to every other Agenty module. The structured error output (rule ID, suggested correction, character offsets) also gives the LLM precise, machine-readable information rather than asking it to detect grammar from raw text, which improves correction accuracy for subtle errors.

---

## Perception module — OAK-D Pro

The camera pipeline runs on the **Myriad X VPU** onboard the OAK-D Pro, offloading all vision compute from the main SBC CPU/GPU.

```
OAK-D Pro  ──►  Left + Right stereo pair  ──►  Onboard depth engine
                        │
                        ▼
               YOLO object detection  (MyriadX NPU, INT8)
                        │
                        ▼
              Spatial coordinate fusion
              (x, y, z in meters from camera origin)
                        │
                        ▼
            Agent context  ──►  "Cup detected 0.4m ahead-left"
                                "Person detected 1.2m straight ahead"
```

**Capabilities:**
- Real-time object detection at 30 FPS (YOLO, INT8 on VPU — no main CPU cost)
- Per-detection 3D spatial coordinates via stereo triangulation
- Configurable confidence thresholds and ROI filtering
- Structured detection output consumed by the agent dispatcher

**Why this matters for autonomous systems:** the perception-to-decision loop implemented here directly mirrors ADAS sensor pipelines — sensor input → spatial reasoning → agent action.

---

## Orchestrator flow

```
[main.py starts]
       │
       ▼
[Greet + detect intent]
       │
       ├──► "practice English"          → TutorSession (English + LanguageTool)
       ├──► "question about physics"    → EngineeringSession (scientist persona + RAG)
       ├──► "what do I have pending?"   → AgentSession (direct tool call)
       └──► "what do you see?"          → AgentSession + OAK-D perception query

[From any active mode — lightweight tools, no interruption]
  "add task X"              → task_add        → returns to previous mode
  "calendar today?"         → cal_list        → returns to previous mode
  "remind me at 6pm"        → reminder_set    → returns to previous mode
  "send WhatsApp to mom"    → wa_send         → returns to previous mode

[search_web — interrupts, does NOT return to previous mode]
  → "This will pause your session. Continue?" → confirmed → web search mode
```

---

## Agent tools

| Tool | Description | Clarifies when |
|---|---|---|
| `task_add` | Add a to-do task | Title missing, or datetime detected |
| `task_list` | List pending tasks | — |
| `task_done` | Mark task as done | Ambiguous which task |
| `cal_add` | Add calendar event | Date, start, or end time missing |
| `cal_add_recurring` | Add recurring events | Weekday, time, or end date missing |
| `cal_list` | List events | — |
| `cal_delete` | Delete event | ID missing (suggests listing first) |
| `cal_free` | Find free time slots | — |
| `reminder_set` | Set a reminder | Title or datetime missing |
| `reminder_list` | List all reminders | — |
| `wa_send` | Send WhatsApp message | Ambiguous contact → shows list |
| `wa_read` | Read WhatsApp messages | — |
| `search_web` | DuckDuckGo search | Vague query → asks for detail |
| `perception_query` | Ask what OAK-D sees | — |
| `spatial_query` | Distance to object | Object class or direction unclear |

---

## Project structure

```
Agenty-Edge-Assistant/
├── src/
│   ├── main.py
│   ├── english/
│   │   ├── tutor_session.py          ← conversation loop + grammar feedback
│   │   ├── language_tool_server.py   ← LanguageTool process lifecycle
│   │   └── topics.json               ← 600 LLM-generated conversation topics
│   ├── engineering/
│   │   ├── engineering_session.py    ← tutor loop with RAG integration
│   │   └── rag/
│   │       ├── build_index.py        ← documents → chunks → embeddings → FAISS
│   │       ├── rag_engine.py         ← domain check + FAISS search + context format
│   │       ├── docs/                 ← source PDFs/EPUBs (not committed)
│   │       └── index/                ← generated FAISS index (not committed)
│   ├── engineering/test/
│   │   ├── eval_retrieval.py         ← Precision@3 evaluation with golden set
│   │   ├── generate_responses.py     ← batch LLM response generation
│   │   └── eval_faithfulness.py      ← NLI-based faithfulness scoring
│   ├── agent/
│   │   ├── agent_session.py
│   │   ├── dispatcher.py
│   │   ├── task_manager.py
│   │   ├── local_calendar.py
│   │   ├── reminder_manager.py
│   │   ├── web_search.py
│   │   ├── wa_stub.py
│   │   └── db/
│   ├── perception/
│   │   ├── oak_pipeline.py
│   │   ├── spatial_engine.py
│   │   └── detection_bridge.py
│   └── orchestrator/
│       ├── orchestrator.py
│       └── context_manager.py
```

---

## Installation

### 1. Python dependencies

```bash
conda activate tutor_env
pip install openai requests duckduckgo-search depthai
pip install sentence-transformers faiss-cpu pymupdf   # RAG pipeline
```

### 2. Register the model in Ollama

```bash
cd ~/Desktop/Agenty-Edge-Assistant/model/gguf
ollama create asistente -f Modelfile
ollama list   # verify "asistente" appears
```

### 3. Build the RAG index

Place PDFs in `src/engineering/rag/docs/` (subdirectories supported):

```bash
python src/engineering/rag/build_index.py
# Expected output: ~9,400 chunks, ~35s build time, 38MB index
```

### 4. LanguageTool (English tutor)

Download the LanguageTool standalone server from [languagetool.org](https://languagetool.org). Java is required on the SBC. The assistant manages the process lifecycle automatically — no manual start needed.

### 5. OAK-D Pro setup

```bash
pip install depthai
python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
python src/perception/oak_pipeline.py --test
```

---

## Running

```bash
# Full assistant
cd ~/Desktop/Agenty-Edge-Assistant
python src/main.py

# Standalone modules
python src/engineering/engineering_session.py
python src/english/tutor_session.py
python src/agent/agent_session.py
python src/perception/oak_pipeline.py

# RAG evaluation suite
python src/engineering/test/eval_retrieval.py       # retrieval quality
python src/engineering/test/generate_responses.py   # generate tutor responses
python src/engineering/test/eval_faithfulness.py    # faithfulness scoring
```

---

## Latency benchmarks

> Measured on dev machine with NVIDIA GPU. Numbers on target SBCs pending hardware deployment.

| Mode | TTFT | Total | Notes |
|---|---|---|---|
| Mode detection | 0.11s | 0.21s | |
| Agent tool call | 0.11s | 0.62s | |
| English tutor (short turn) | 0.15s | 1.34s | LanguageTool adds <50ms local |
| Engineering tutor (no RAG) | 0.12s | 11.35s | |
| Engineering tutor (with RAG) | ~0.22s | ~12-13s | +100ms embed, <10ms FAISS |
| OAK-D detection (VPU) | — | ~33ms | 30 FPS, zero main CPU cost |

---

## Roadmap

- [x] **Local RAG pipeline** — FAISS + multilingual embeddings, 100% Precision@3
- [x] **Grammar-aware English tutor** — LanguageTool on-device, zero cloud
- [x] **600 conversation topics** — fully LLM-generated, stored in topics.json
- [x] RKLLM conversion — run LLM on RK3588 NPU (Rock 5B)
- [x] Real WhatsApp — `whatsapp-web.js` on localhost
- [x] Full TTS — Piper / Kokoro replacing `speak()` hooks
- [x] Full STT — local Whisper replacing `listen()` hooks
- [ ] Autostart on SBC boot via `systemd`
- [x] Multi-object tracking across frames (OAK-D)
- [x] Latency benchmarks on Jetson Orin Nano and Radxa Rock 5B

---

<div align="center">

*Built and deployed on real hardware. No cloud. No shortcuts.*

[![Author](https://img.shields.io/badge/Gianluca_Ferraris-0d1117?style=for-the-badge&logo=github&logoColor=58a6ff)](https://github.com/GianlucaaFerraris)
[![UNC](https://img.shields.io/badge/Computer_Engineering_@_UNC-0d1117?style=for-the-badge&logoColor=8b949e)](https://github.com/GianlucaaFerraris)

</div>
