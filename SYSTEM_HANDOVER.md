# LARPAn1: System Handover & Context Document

This document is generated to provide a fresh instance of the AI assistant with a comprehensive understanding of the **LARPAn1** (LARP Analyzer, iteration 1) project. Read this to understand the architecture, the services, the recent optimizations, and the complete history of our development journey.

## 1. Project Overview
LARPAn1 is the unified successor to the PSI-09 ecosystem. It is a production-grade, highly concurrent agentic profiling and conversational engagement engine. Its goal is to ingest chatroom telemetry, strip away performative human behavior ("LARP"), and process conversational dynamics via advanced NLP. 

It acts as a headless FastAPI backend that connects to any platform bridge (Discord, WhatsApp, Minecraft) using standard JSON payloads.

## 2. Core Architecture & Services
The application follows a clean, domain-driven FastAPI architecture designed for non-blocking, high-volume throughput. 

### A. Routing & Ingestion
- Bridges send JSON payloads to `POST /psi09`.
- Payloads are strictly typed and validated using `Pydantic` (`IncomingPayload`). Invalid payloads are instantly rejected with `422 Unprocessable Entity` to protect the backend.
- The `EngineDispatcher` reads the `mode` flag (`"vrag"`, `"legacy"`, `"auto"`) and routes the payload to the appropriate engine.

### B. Dual Engine Paradigms
1. **VRAGEngine (Graph-Based Social Triage):**
   - The modern, experimental engine. Uses **GraphRAG**.
   - Maintains a `NetworkX` knowledge graph of users and groups. Calculates social standing using **PageRank**, evaluates community factions (greedy modularity), and tracks temporal decay of relationships.
   - Uses a **LangGraph** state machine. A `TriageNode` decides if the bot should engage. If yes, it passes to the combat node.
   - The combat node runs a 4-stage **DSPy pipeline** (Identity, Mission, Constraints, Decision) to dynamically formulate text or emoji reactions.

2. **RoastbotEngine (Legacy Text-Based Profiling):**
   - The high-throughput legacy engine.
   - Relies on brute-force text concatenation. It injects recent history and global text behavioral summaries into fixed prompt templates.
   - Bypasses LangGraph triage; reacts on hardcoded triggers (e.g., explicit pings) with a single-shot LLM completion.

### C. Background Profiling & Non-Blocking Concurrency
- To prevent LLM inference (DSPy/Groq/NVIDIA NIM) and sync database I/O from freezing the `asyncio` event loop, LARPAn1 wraps these calls in `asyncio.to_thread()`.
- **Background Tasks:** Once the API responds to the bridge, FastAPI spins up background tasks (`evolve_profile_task`) to update longitudinal psychological profiles or social graphs asynchronously without delaying the user HTTP response.

### D. Repository Layer & MongoDB
- Data logic is abstracted into singletons (`ChatRepository`, `MemoryRepository`, `GraphRepository`, etc.).
- Caching is handled locally via a custom `TTLCache` in memory, drastically reducing the number of read operations on the MongoDB cluster.

### E. Dynamic LLM Load Balancing
- Uses custom balancers (`FailoverLMPool`, `NvidiaRoundRobinPool`) mapped to lists of API keys via `.env`.
- Automatically traps `429 Rate Limit` and `500 Server Error` exceptions, advancing to the next API key in the pool seamlessly.

---

## 3. The Development Journey (How We Got Here)

This project evolved through intense refactoring and problem-solving. Here is everything we've done since the beginning, mapped chronologically:

### Phase 1: The Hugging Face Deployment Nightmare
We initially tried to deploy LARPAn1 on Hugging Face Spaces but were met with a gauntlet of deployment and dependency issues. 
- We wrestled with Docker paid tier constraints versus the Gradio SDK. We continuously bounced between deploying via raw Docker and deploying via the Gradio Python wrapper.
- We hit "dependency hell" with Hugging Face's internal environment (Gradio 6 transpilation crashes) forcing us to pin Gradio to `4.44.1`, then `5.9.1`.
- We encountered Python shadowing errors (`app.py` conflicting with our internal `app/` directory), which we fixed by renaming the entry point to `main.py`.
- We even went as far as injecting a dummy `@spaces.GPU` decorator to trick Hugging Face's ZeroGPU strict eviction policy into not killing our tasks.

### Phase 2: The Render.com Migration
Recognizing Hugging Face was not built for headless, high-throughput backend APIs, **we completely abandoned Hugging Face** and migrated the project to Render.com. We stripped out the Gradio wrappers, removed the HF sync workflows, and optimized purely for a standard ASGI containerized deployment.

### Phase 3: Core Model Fixes & GraphRAG (vRAG) Implementation
Once stable on Render, we focused on the AI logic:
- We debugged `404 Not Found` errors by correcting hallucinated LLM model strings. We strictly mapped to valid **Groq partner models** and **NVIDIA Mistral-Large**.
- We implemented the **vRAG graph extraction pipeline**, plugging in `NetworkX` to calculate **PageRank** and **community detection (greedy modularity)** to map social cliques dynamically.
- We upgraded to **DSPy 3** compatibility, swapping out the deprecated `TypedPredictor` for the modern `Predictor`.
- We fixed a critical bug where payloads were dispatched before being saved in MongoDB, which was starving the background graph evolution task of the current message data.

### Phase 4: Closing Parity Gaps with PSI-09 Legacy
To fully replace the original PSI-09 ecosystem, we executed a deep audit and closed 6 major parity gaps:
1. Added `fetch_tagged_profiles()` to inject `<tagged_member_profiles>`, giving the AI bystander awareness.
2. Ported the `EVOLVE_EVERY_N_MESSAGES` counter-gating logic (bypassing for First Contact) to save API tokens during high-volume chat spikes.
3. Implemented a token budget trimmer (`trim_messages_to_token_budget`) using word-count estimations to prevent context window overflow.
4. Ported a dense **5-regex think-tag sanitization chain** to ensure `<think>` reasoning blocks don't leak into the persistent MongoDB behavioral profiles.
5. Restored global memory profiling.
6. Enriched global history entries with platform and channel tags (`[Sent via platform - group #channel]`).

### Phase 5: The Micro-VM Optimization Audit (Final Polish)
Render's free tier provides a micro-VM (0.1 vCPU / 512MB RAM). Our sophisticated pipeline was causing resource starvation. We executed a massive optimization pass:
- **Threaded I/O:** Wrapped every single synchronous `PyMongo` call in `asyncio.to_thread()` to prevent the ASGI event loop from freezing.
- **TTLCache:** Created a custom thread-safe TTL cache in `repositories.py` to cache profile and graph reads.
- **Memory Leak Protection:** Bound MongoDB array growth using `$slice` during `$push` to prevent documents from ballooning past 16MB. Fixed a memory leak in `TTLCache` by sweeping expired keys and lowering the `max_size` from 10000 -> 500 -> and finally tuning to **1500**.
- **Connection Pooling:** Dropped PyMongo `maxPoolSize` to 10 to reduce idle RAM consumption.
- **CPU Guardians:** Bounded `nx.pagerank` with `max_iter=50` and added a bypass check to completely skip $O(n^2 \log n)$ community detection if a graph has more than 150 nodes. 

## 4. Next Steps for New AI Instances
When jumping into this context window, remember that this is a highly constrained micro-VM environment. Any new features involving heavy computation (like graph traversal or heavy NLP) must include CPU and Memory guardian checks. Database writes must respect token limits and BSON sizing, and all I/O must remain non-blocking via threads.

### Phase 6: Deep Legacy Parity Audit & Final Hardening
A line-by-line comparison of both `psi-09-roastbot` and `PSI-09-vRAG` legacy codebases was performed against LARPAn1, and 9 issues were identified and fixed:

**Critical Fixes:**
1. **GROUP_SUMMARY_PROMPT restored** — `background.py` was using the generic individual `EVOLUTION_PROMPT` for group profiling. Now uses the dedicated `GROUP_SUMMARY_PROMPT` which focuses on social hierarchy, inside jokes, and collective dynamics (matching legacy surveillance engine behavior).
2. **Roastbot structured message payloads** — `roastbot.py` was concatenating everything into a flat string. Now builds proper `messages=[{role: "system", ...}, {role: "user", ...}]` chat-completion payloads, giving the NVIDIA combat LLM proper role separation for instruction-following (matching legacy's `llm_feed` structure).

**High-Severity Fixes:**
3. **vRAG think-tag sanitization** — `vrag.py` combat node was not stripping `<think>` blocks from replies. Now applies `sanitize_think_tags()` to combat output (matching roastbot).
4. **Discord snowflake mention cleaning** — Raw `<@123456>` strings from Discord bridges are now replaced with `@PSI-09` at the route level (`routes.py`) and in history formatting (`roastbot.py`), matching legacy's `bot_mentioned_in()` and inline substitution logic.
5. **Graph extraction input sanitization** — `background.py` now strips Discord snowflakes and bare 17-19 digit IDs from chat text before sending to the graph extractor, preventing junk entity creation (matching legacy vRAG's `re.sub` sanitization).

**Medium Fixes:**
6. **vRAG message/location formatting** — Messages now include sender attribution (`[username]: message`) and locations are descriptive (`Server: X | Channel: #Y`) matching legacy vRAG.
7. **Package structure** — Added `__init__.py` to all 7 package directories for proper Python packaging.
8. **Dead dependencies removed** — Stripped `huggingface_hub` and `transformers` from `requirements.txt` (abandoned HF-era baggage).
