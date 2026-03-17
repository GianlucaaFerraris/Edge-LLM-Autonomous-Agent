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

The result is a system that can reason, schedule, search, and *see* — all on hardware that fits in your hand.

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENTY  PIPELINE                         │
│                                                                 │
│   Voice / Text ──► Orchestrator ──► Intent Router              │
│                          │                                      │
│              ┌───────────┼───────────┐                         │
│              ▼           ▼           ▼                         │
│         English      Engineering   Agent                       │
│          Tutor         Tutor      Session                      │
│                                     │                          │
│                          ┌──────────┴──────────┐              │
│                          ▼                     ▼              │
│                    Tool Dispatcher     Perception Module       │
│                    (tasks/calendar/    OAK-D Pro NPU          │
│                     reminders/web)    Stereo Depth + YOLO     │
│                          │                     │              │
│                          └──────────┬──────────┘              │
│                                     ▼                          │
│                              SQLite Storage                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key highlights

| | |
|---|---|
| 🧠 **Quantized INT4 LLM** | Qwen2.5-7B served by Ollama — ~30% of original model size, full reasoning capability |
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
- Outputs structured detections consumed by the agent's dispatcher for decision-making

**Why this matters for autonomous systems:** the same perception-to-decision loop used here directly mirrors ADAS pipelines — sensor input → spatial reasoning → agent action.

---

## Orchestrator flow

```
[main.py starts]
       │
       ▼
[Greet + detect intent]
       │
       ├──► "practice English"          → TutorSession (English + LanguageTool)
       ├──► "question about physics"    → EngineeringSession (scientist persona)
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
│   │   └── tutor_session.py          ← English tutor + LanguageTool
│   ├── engineering/
│   │   └── engineering_session.py    ← Engineering tutor (scientist persona)
│   ├── agent/
│   │   ├── agent_session.py          ← tool orchestrator + clarification loop
│   │   ├── dispatcher.py             ← tool execution + ambiguity handling
│   │   ├── task_manager.py           ← SQLite to-do list
│   │   ├── local_calendar.py         ← SQLite calendar (simple + recurring)
│   │   ├── reminder_manager.py       ← SQLite reminders + 30-min scheduler
│   │   ├── web_search.py             ← DuckDuckGo (no API key required)
│   │   ├── wa_stub.py                ← WhatsApp stub
│   │   └── db/                       ← persistent databases on SBC
│   ├── perception/
│   │   ├── oak_pipeline.py           ← DepthAI pipeline (OAK-D Pro)
│   │   ├── spatial_engine.py         ← stereo depth + 3D coordinate fusion
│   │   └── detection_bridge.py       ← structured output → agent dispatcher
│   └── orchestrator/
│       ├── orchestrator.py           ← intent detection + mode routing
│       └── context_manager.py        ← shared state across modes
└── src/test/
    ├── test_modes.py
    ├── test_latency.py
    ├── test_perception.py
    └── manual_chat.py
```

---

## Installation

### 1. Python dependencies

```bash
conda activate tutor_env
pip install openai requests duckduckgo-search depthai
```

### 2. Register the model in Ollama

```bash
cd ~/Desktop/Agenty-Edge-Assistant/model/gguf

# Edit Modelfile: set the absolute path to your .gguf file
# FROM /home/<user>/Desktop/.../Qwen2.5-7B-Instruct.Q4_K_M-001.gguf

ollama create asistente -f Modelfile
ollama list   # verify "asistente" appears
```

### 3. OAK-D Pro setup

```bash
# Install DepthAI
pip install depthai

# Verify camera is detected
python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"

# Run perception test
python src/perception/oak_pipeline.py --test
```

### 4. LanguageTool (English tutor only — optional)

```bash
java -cp languagetool-server.jar org.languagetool.server.HTTPServer \
     --port 8081 --allow-origin '*' --public
```

The assistant starts without LanguageTool — grammar detection is simply disabled.

---

## Running

```bash
# Full assistant (recommended)
cd ~/Desktop/Agenty-Edge-Assistant
python src/main.py

# Perception module only
python src/perception/oak_pipeline.py

# Standalone modules
python src/english/tutor_session.py
python src/engineering/engineering_session.py
python src/agent/agent_session.py

# Tests
cd src/test
pytest test_modes.py -v
python test_latency.py
python test_perception.py   # requires OAK-D Pro connected
```

---

## Latency benchmarks

> Measured on laptop with NVIDIA GPU. Run `python test_latency.py` after deployment
> for accurate numbers on your target hardware.

| Mode | TTFT | Total | TPS |
|---|---|---|---|
| Mode detection | 0.11s | 0.21s | 6.5 |
| Agent tool call | 0.11s | 0.62s | 10.7 |
| English tutor (short) | 0.15s | 1.34s | 27.5 |
| Engineering tutor (medium) | 0.12s | 11.35s | 25.5 |
| OAK-D detection (VPU) | — | ~33ms | 30 FPS |
| Spatial coordinate fusion | — | ~5ms | per detection |

---

## Roadmap

- [ ] RKLLM conversion — run LLM on RK3588 NPU (Rock 5B)
- [ ] Local RAG — `sentence-transformers` + FAISS over lecture notes and docs
- [ ] Real WhatsApp — `whatsapp-web.js` on localhost
- [ ] Full TTS — Piper / Kokoro replacing `speak()` hooks
- [ ] Full STT — local Whisper replacing `listen()` hooks  
- [ ] Autostart on SBC boot via `systemd`
- [ ] Multi-object tracking across frames (OAK-D)
- [ ] Obstacle-aware navigation suggestions from spatial detections

---

<div align="center">

*Built and deployed on real hardware. No cloud. No shortcuts.*

[![Author](https://img.shields.io/badge/Gianluca_Ferraris-0d1117?style=for-the-badge&logo=github&logoColor=58a6ff)](https://github.com/GianlucaaFerraris)
[![UNC](https://img.shields.io/badge/Computer_Engineering_@_UNC-0d1117?style=for-the-badge&logoColor=8b949e)](https://github.com/GianlucaaFerraris)

</div>
