# LARPAn1: The Apex LARP Analyzer

> *"Because your online persona is a fragile construct, and we have the compute to prove it."*

**LARPAn1** (LARP Analyzer, iteration 1) is a production-grade, highly concurrent agentic profiling and conversational engagement engine. Developed as the unified successor to the scattered PSI-09 ecosystem, LARPAn1 is engineered to ingest high-volume, unfiltered chatroom telemetry, strip away performative human posturing (colloquially known as "LARP"), and systematically deconstruct the user's ego via advanced natural language processing.

It operates silently, asynchronously, and relentlessly.

---

## 🎭 System Architecture

LARPAn1 discards the monolithic, tightly coupled anti-patterns of its predecessors in favor of a clean, domain-driven FastAPI architecture. It is designed to act as a headless backend, offering plug-and-play compatibility with any platform bridge (Discord, WhatsApp, Minecraft) capable of authenticating and formatting standard JSON payloads.

The engine multiplexes across two distinct operational paradigms, routed seamlessly via a polymorphic dispatcher.

### Internal Processing Flow

1. **Ingestion & Validation**: Bridges send payloads to `POST /psi09`. The payload is strictly validated against Pydantic schemas.
2. **Dispatching**: The `EngineDispatcher` reads the `mode` parameter and routes the payload to either the `RoastbotEngine` or `VRAGEngine`.
3. **Execution**: The selected engine retrieves required context from MongoDB via the Repository Layer, processes the LLM inference synchronously within a background thread (to preserve the asyncio event loop), and formats the output.
4. **Background Profiling**: Regardless of the engine selected, asynchronous background tasks are queued via FastAPI's `BackgroundTasks` to update the user's longitudinal psychological profile or social graph without delaying the HTTP response.

---

## Technical Specifications

### 1. Payload Contracts and Validation (Pydantic)
External bridges are inherently untrustworthy. LARPAn1 enforces strict typing via `pydantic.BaseModel`.

```python
class IncomingPayload(BaseModel):
    message: str 
    sender_id: str 
    username: str 
    display_name: str 
    group_name: str 
    channel: str 
    tagged_users: List[TaggedUser]
    platform: Literal["discord", "whatsapp", "minecraft"]
    force_reply: bool
    mode: Literal["auto", "legacy", "vrag"]
```
If a bridge submits a malformed payload, FastAPI automatically intercepts the request and issues a `422 Unprocessable Entity` response, entirely isolating the backend from data-formatting faults.

### 2. Dual Engine Paradigms

The system relies on a polymorphic `BaseEngine` interface. Implementations must return an `EngineResponse` containing `reply`, `reaction`, and `engine_used`.

#### A. VRAGEngine (Graph-Based Social Triage)
The frontier experimental mode (`mode="vrag"`). This engine eschews flat text summaries in favor of a structured GraphRAG approach.
- **Context Assembly**: Utilizes a `NetworkX` knowledge graph to evaluate social standing via PageRank, detect isolated communities, and track temporally decaying alliances.
- **The Gatekeeper (LangGraph)**: Implements a state machine where the entry node (`TriageNode`) actively determines engagement protocol. If the user's input lacks intellectual merit or explicit invocation, the LLM sets `should_engage = False`, bypassing the combat node entirely to maintain superior silence.
- **DSPy Reasoning**: The combat node operates a 4-stage DSPy pipeline:
  1. `IdentitySignature`: Derives dynamic persona based on target graph context.
  2. `MissionSignature`: Synthesizes a tactical objective.
  3. `ConstraintsSignature`: Enforces formatting, character limits, and behavioral mandates.
  4. `DecisionSignature`: Returns a strictly typed `CombatDecision` capable of issuing a text reply, an emoji reaction, or both.

#### B. RoastbotEngine (Legacy Text-Based Profiling)
The high-throughput legacy mode (`mode="legacy"`).
- **Execution**: Relies on brute-force text concatenation. It pulls recent group history and textual behavioral summaries from MongoDB, injecting them into a rigid prompt template.
- **Routing**: It bypasses the LangGraph triage logic, relying on hardcoded triggers (e.g., explicit pings or private messages) before firing a single-shot completion request to the LLM pool.

### 3. Concurrency and Non-Blocking Surveillance
A critical flaw in standard Python LLM applications is synchronous blocking. DSPy and standard HTTP requests freeze the event loop, starving the ASGI server of throughput. 

LARPAn1 circumvents this by wrapping all LLM generation logic inside `asyncio.to_thread()`. 
```python
# Executed within FastAPI BackgroundTasks
final_state = await asyncio.to_thread(compiled_vrag_agent.invoke, initial_state)
```
This isolates the synchronous CPU/IO blocking of LLM inference to a dedicated thread pool, allowing Uvicorn to continue accepting and routing hundreds of concurrent websocket or HTTP events from the platform bridges.

### 4. Dynamic LLM Load Balancing
Both Groq (utilized for background profiling and triage) and NVIDIA NIM (utilized for combat generation) are shielded by bespoke load balancers (`FailoverLMPool` and `NvidiaRoundRobinPool`).

Configuration relies on `pydantic-settings` to parse comma-separated lists of API keys from the environment:
```env
NVIDIA_API_KEYS="key_1,key_2,key_3"
GROQ_API_KEYS="key_1,key_2"
```
The balancers instantiate combinations of API keys and underlying models. If a `429 Rate Limit` or `5xx Server Error` is encountered during inference, the balancers automatically trap the exception, advance the internal pointer to the next available API key, and transparently retry the request.

### 5. MongoDB Repository Layer
Database operations have been extracted from the core execution loops into a dedicated Repository Layer (`ChatRepository`, `MemoryRepository`, `GraphRepository`). This manages the PyMongo connection pool centrally, ensuring connections are multiplexed efficiently and closed gracefully during the FastAPI lifespan teardown.

---

## Installation & Deployment

LARPAn1 is designed for standard containerized or bare-metal execution environments running Python 3.10+.

### Prerequisites
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Configuration
Create a `.env` file in the root directory. `pydantic-settings` will automatically map these to the `Settings` class with strict type coercion.

```env
# MongoDB Atlas or Local Connection String
MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/psi09"

# Comma-Separated API Keys
NVIDIA_API_KEYS="your_nvidia_key_1,your_nvidia_key_2"
GROQ_API_KEYS="your_groq_key_1,your_groq_key_2"

# Application Tuning (Optional Overrides)
MEMORY_TTL=500
GROUP_HISTORY_MAX_MESSAGES=50000
```

### Ignition
Start the ASGI server using Uvicorn. The engine will bind to `0.0.0.0:7860` and establish the MongoDB connection pools on startup.

```bash
python run.py
```

## Legacy Note (PSI-09)
While the core architecture has been completely rewritten into LARPAn1, the underlying spirit of this engine remains deeply indebted to the original **PSI-09** proof-of-concept scripts:
- [PSI-09 Production (Legacy Roastbot)](https://github.com/sudoboneman/PSI-09-production)
- [PSI-09 vRAG (Experimental Graph Engine)](https://github.com/sudoboneman/PSI-09-vRAG)

The legacy `RoastbotEngine` continues to honor the original prompt matrices that gave PSI-09 its infamous reputation.

---

> *"They type to fill the void. We type to widen it."*  
> **— LARPAn1 Protocol**
