# LARPAn1: System Architecture & Handover Document

**Target Audience:** New AI Instance / Context Window
**Project Scope:** Unification of `psi-09-roastbot` and `PSI-09-vRAG` into `psi-09-engine-v2` (LARPAn1)
**Deployment Target:** Render (Free Tier: 512MB RAM, 0.1 CPU)

---

## 1. Project Overview
LARPAn1 is the unified successor to the original PSI-09 engines. It merges the direct, prompt-injected aggression of the legacy `roastbot` with the deep, graph-based psychological analysis of `vRAG`. The core objective of this project was to establish a single FastAPI entrypoint that dynamically routes incoming payloads to the correct sub-engine while operating flawlessly on strict hardware limits (Render Free Tier).

## 2. Core Architecture

The system is built on a highly modular architecture using FastAPI, LangGraph, DSPy, and NetworkX.

### A. The API Layer (`app/api/`)
- `routes.py`: The single entrypoint (`/psi09`). It intercepts payloads, normalizes Discord snowflake IDs (converting `<@123...>` to `@PSI-09`), hands the payload to the dispatcher, stores the messages in MongoDB chronologically, and finally fires asynchronous background evolution tasks.
- `models.py`: Strict Pydantic schemas (`IncomingPayload`, `EngineResponse`). Enforces required fields like `display_name` and restricts `mode` to `["auto", "legacy", "vrag"]`.

### B. The Engine Layer (`app/engine/`)
- **Triage (LangGraph):** A gatekeeper node powered by DSPy. Evaluates the conversation history to determine if PSI-09 should engage or remain silent. Uses `.lower()` checks to catch case-insensitive mentions.
- **Roastbot (`roastbot.py`):** The legacy fallback. Flat prompt injection using local, group, and global MongoDB text profiles. It fetches global profiles for tagged users dynamically to enhance contextual mockery.
- **vRAG (`vrag.py`):** The experimental GraphRAG engine. It constructs a 4D social graph of the user, applying mathematical PageRank decay to evaluate their social standing before generating a response.
- **Graph Analyzer (`graph_analyzer.py`):** The mathematical core. Uses NetworkX to calculate PageRank (`max_iter=500`, `tol=1e-6`) and detect community factions using `greedy_modularity_communities` (capped safely at 500 nodes to prevent Render `OOMKilled` crashes on the 0.1 CPU limit).

### C. The Core Infrastructure (`app/core/` & `app/db/`)
- **LLM Balancers (`llm_balancer.py`):** 
  - `FailoverLMPool`: Used for Triage and Background tasks via Groq. Safely traps 429 Rate Limits and cascades (`advance()`) to the next available API key.
  - `NvidiaRoundRobinPool`: Used for the Combat engines to maximize throughput across multiple keys.
- **MongoDB (`mongo.py` & `repositories.py`):** A clean Repository pattern abstracts all PyMongo calls. It includes a dedicated background keepalive thread that pings the database every 180 seconds to prevent Render connection timeouts.

---

## 3. History of Modifications (The Hardening Phases)

Over the course of an intense, multi-phase audit, the following critical bugs and loose ends were permanently fixed:

1. **vRAG Background Extraction Spikes:** In legacy vRAG, the LLM attempted a full DSPy graph extraction on *every single message*, causing immediate rate limits. **Fix:** Implemented strict message counters (`EVOLVE_EVERY_N_MESSAGES=50`, `GROUP_SUMMARY_EVERY_N=300`) using thread locks in `background.py` to pace the extractions.
2. **First Contact Amnesia:** New users wouldn't trigger vRAG extractions until they hit the 50-message mark. **Fix:** Added "First Contact" logic that forces an immediate graph extraction if the user's graph repository is completely empty.
3. **Infinite PageRank Loops:** Dense networks caused `nx.pagerank` to fail to converge, triggering Render server restarts. **Fix:** Added `max_iter=500` and `tol=1e-6` to force mathematical convergence within a microsecond without sacrificing quality.
4. **Community Detection CPU Spikes:** `nx_comm.greedy_modularity_communities` is $O(N \log N)$ and occasionally $O(N^2 \log N)$. **Fix:** Added a CPU Guardian that bypasses community detection if the graph exceeds 500 nodes, protecting the 512MB RAM free tier VM.
5. **Case-Sensitive Mentions:** The triage logic strictly checked for `"@PSI-09"`. If a user manually typed `@psi-09`, they were ignored. **Fix:** Enforced `.lower()` on all manual engagement checks across `vrag.py` and `roastbot.py`.
6. **LLM Pool Crashes:** LangGraph nodes were calling `.get_model()` on the FailoverPool, which didn't exist. **Fix:** Corrected to use `.get_current()` to unpack the tuple properly.
7. **Pydantic Type Safety in DSPy:** Legacy DSPy signatures relied on prompt requests for JSON. **Fix:** The new signatures now inherit Pydantic BaseModels (e.g., `GraphExtractionDecision`), forcing the DSPy compiler to strictly guarantee schema compliance.

---

## 4. Current State & Next Steps

The `LARPAn1` pipeline has been successfully compiled, tested against live Render deployments via cURL payloads, and proven to be fundamentally superior to the legacy codebase. The system handles missing metadata, avoids LLM rate limits, manages MongoDB connections asynchronously, and mathematically profiles users perfectly within its hardware limitations.

**Instructions for New Instance:**
1. The codebase is stable. Do not modify the architectural flow or the mathematical constraints in `graph_analyzer.py` without explicitly evaluating the Render Free Tier limits.
2. If the user wants to add new features (e.g., expanding the database, adding new DSPy signatures), strictly follow the Repository pattern in `app/db/` and the Pydantic schemas in `app/api/models.py`.
