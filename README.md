# LARPAn1: The First LARP Analyzer 🎭

<div align="center">
  <p><i>"Because your online persona is a fragile construct, and we have the compute to prove it."</i></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
  [![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
  [![DSPy](https://img.shields.io/badge/DSPy-AI-FF4B4B?style=for-the-badge)](https://dspy.ai/)
</div>

---

## What is this?

**LARPAn1** (LARP Analyzer, iteration 1) is a production-grade, highly concurrent agentic profiling engine. Built as the unified successor to the scattered PSI-09 ecosystem, LARPAn1 is designed to ingest high-volume, unfiltered chatroom telemetry, strip away performative human posturing (colloquially known as "LARP"), and systematically deconstruct the user's ego via advanced natural language processing. 

It operates silently, asynchronously, and relentlessly.

While you are sleeping, LARPAn1 is running PageRank on your social standing, parsing your chat history into a multi-dimensional knowledge graph, and compiling a mathematical dossier of your insecurities. 

---

## Bleeding-Edge Architecture

LARPAn1 discards the monolithic, tightly coupled anti-patterns of its predecessors in favor of a clean, domain-driven FastAPI architecture. It acts as a headless backend, offering plug-and-play compatibility with any platform bridge (Discord, WhatsApp, Minecraft) capable of firing standard JSON payloads.

### The Dual Engine Paradigm

The engine multiplexes across two distinct operational paradigms, routed dynamically via a polymorphic dispatcher:

#### 1. The vRAG Engine (Graph-Based Social Triage)
The frontier experimental mode (`mode="vrag"`). This engine eschews flat text summaries in favor of a structured **GraphRAG** approach.
* **NetworkX Context Assembly:** Constructs a 4D social graph of the user, applying mathematical exponential decay to edge weights (because friendships fade, but the engine remembers). It calculates `nx.pagerank` with a micro-decimal tolerance to evaluate social hierarchies, and isolates community factions using greedy modularity algorithms.
* **The LangGraph Gatekeeper:** Implements a state machine where the entry node (`TriageNode`) actively determines engagement protocol. If the user's input lacks intellectual merit or explicit invocation, the LLM bypasses the combat node entirely, maintaining superior silence.
* **DSPy Reasoning Pipeline:** The combat node operates a strictly-typed, 4-stage DSPy pipeline (`Identity`, `Mission`, `Constraints`, `Decision`), forcing the LLM to output Pydantic-compliant JSON payloads.

#### 2. The Legacy Roastbot (`mode="legacy"`)
For those who prefer brute force. The high-throughput legacy mode falls back to flat text profiling, relying on massive prompt matrices and global cross-platform dossiers to deliver unfiltered, single-shot psychological devastation.

---

## Core Technical Features

* **Asynchronous LLM Balancing:** Both Groq and NVIDIA NIM are shielded by bespoke, thread-safe load balancers (`FailoverLMPool` and `NvidiaRoundRobinPool`). If an API key hits a `429 Rate Limit`, the engine safely traps the error, shifts the pointer to the next key, and retries instantly. 
* **Non-Blocking Surveillance:** A critical flaw in standard Python LLM applications is synchronous blocking. LARPAn1 wraps all DSPy inference inside `asyncio.to_thread()`, isolating CPU-heavy workloads and allowing the Uvicorn server to effortlessly juggle hundreds of concurrent payloads.
* **Gated Background Profiling:** Graph extractions and text summaries are strictly paced via configurable message counters (e.g. `EVOLVE_EVERY_N_MESSAGES`). When a trigger is hit, the engine fires a `BackgroundTasks` thread to map the user's new psychological traits without delaying the HTTP response.
* **Pydantic Type Enforcement:** External bridges are inherently untrustworthy. If a webhook submits a malformed payload, FastAPI intercepts it instantly with a `422 Unprocessable Entity`—keeping the core logic completely sterilized.

---

## Installation & Startup

LARPAn1 is designed for standard containerized or bare-metal execution environments (specifically optimized for strict constraints like Render's Free Tier).

### Prerequisites
Clone the repository and install the dependencies:
```bash
git clone https://github.com/larp-analyser/LARPAn1.git
cd LARPAn1
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the root directory. The application uses `pydantic-settings` to map these directly into memory with strict type coercion.

```env
# MongoDB Atlas Connection
MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/psi09"

# Comma-Separated API Keys (For Dynamic Failover)
NVIDIA_API_KEYS="sk-123...,sk-456..."
GROQ_API_KEYS="gsk-abc...,gsk-def..."

# Application Tuning Limits
GROUP_HISTORY_SLICE=80
EVOLVE_EVERY_N_MESSAGES=50
GROUP_SUMMARY_EVERY_N=300

# Bot Identifiers (For Snowflake Interception)
DISCORD_ID="123456789012345678"
```

### Server Startup
Fire up the ASGI server using Uvicorn. The engine will bind to `0.0.0.0:7860` and establish the MongoDB connection pools, complete with a background keepalive thread.

```bash
python run.py
```

---

## Legacy Lineage
While the core architecture has been completely rewritten from the ground up to create LARPAn1, the underlying spirit of this engine remains deeply indebted to the original **PSI-09** proof-of-concept scripts:
- [PSI-09 Production (Legacy Roastbot)](https://github.com/sudoboneman/PSI-09-production)
- [PSI-09 vRAG (Experimental Graph Engine)](https://github.com/sudoboneman/PSI-09-vRAG)

The legacy `RoastbotEngine` embedded within LARPAn1 continues to honor the original prompt matrices that gave PSI-09 its infamous reputation.

---
*Low-latency malice. There's no override.*
