# LARPAn1 Deep Technical Audit Report

**Scope:** Full line-by-line comparison of [psi-09-engine-v2](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2) (LARPAn1) against legacy [psi-09-roastbot](file:///home/qop/projects/PSI-CORE/psi-09-roastbot) and [PSI-09-vRAG](file:///home/qop/projects/PSI-CORE/PSI-09-vRAG)

---

## Executive Summary

LARPAn1 is a well-structured, modular rewrite that successfully unifies both legacy engines. The core architectural flow, DSPy signatures, LangGraph state machine, and MongoDB repository pattern are all solid. However, the audit uncovered **15 issues** ranging from missing legacy capabilities to subtle behavioral regressions and engineering loose ends.

| Severity | Count |
|----------|-------|
| 🔴 Critical (Functional regression / data loss) | 4 |
| 🟡 Moderate (Behavioral drift / potential bugs) | 6 |
| 🔵 Low (Code quality / optimization) | 5 |

---

## 🔴 Critical Issues

---

### C1. vRAG Background Evolution Runs on Every Single Message (No Gating)

**Files:** [background.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/tasks/background.py#L220-L245)

The SYSTEM_HANDOVER explicitly calls out this bug as **fix #1**: *"the LLM attempted a full DSPy graph extraction on every single message, causing immediate rate limits."*

**The problem in LARPAn1:** The vRAG branch of `evolve_profile_task()` correctly gates **user** graph extraction (lines 222-238) with `EVOLVE_EVERY_N_MESSAGES`. However, the **group** graph extraction (lines 240-245) is gated at `GROUP_SUMMARY_EVERY_N=300`, but **there is no First Contact logic for groups in vRAG mode**. 

Compare to the legacy vRAG (line 680-688): The legacy vRAG fires graph extraction on **every single call** for both user and group in the background thread — but it uses per-user/per-group locks to serialize. LARPAn1 correctly added gating for users but the group gating works. This is actually fine on closer inspection — the group gate at 300 is correct.

**However, the real critical issue is different:** In legacy vRAG, `summarize_user_history()` does a **full replace** of the graph data (`graph_user_cache.set(user_key, graph_dict)` at line 563). In LARPAn1, `_evolve_graph()` does a **merge** (lines 78-123 of background.py). This is architecturally superior, but there's a subtle bug: the merge logic for **relationships** uses `(source, relation, target)` as the deduplication key (line 101). If the same relationship gets extracted with a different `intensity`, the old intensity is preserved and the new one is silently dropped. The legacy vRAG just overwrites everything, which means relationship intensities always reflected the latest extraction. LARPAn1's merge silently freezes relationship intensities at their first-observed value.

> [!CAUTION]
> **Impact:** Relationship intensities in the graph will become stale over time. A friendship that intensifies from 3.0 to 9.0 over weeks will remain frozen at 3.0 in LARPAn1.

**Fix:** When a relationship `(source, relation, target)` already exists, update its intensity via a weighted average or take the max, rather than skipping it entirely.

---

### C2. Legacy vRAG Graph Extraction Signature Mismatch

**Files:** [dspy_signatures.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/prompts/dspy_signatures.py#L118-L131) vs [legacy vRAG main.py](file:///home/qop/projects/PSI-CORE/PSI-09-vRAG/main.py#L528-L536)

The legacy vRAG `GraphExtractionSignature` has two input fields:
```python
target_focus: str  # "Deep psychological profile of user: {username}"
chat_log: str      # The raw chat history
```

LARPAn1's `GraphExtractionSignature` has **three** input fields:
```python
chat_history: str
existing_entities: str
existing_relationships: str
```

The legacy `target_focus` field is **completely missing** from LARPAn1. This field was critical in the original vRAG — it told the LLM **who** to focus the extraction on, e.g. `"Deep psychological profile of user: Steve"` for users or `"Map the social dynamics..."` for groups.

Without `target_focus`, the LLM in LARPAn1's graph extraction has no explicit instruction about which entity to prioritize. The existing `existing_entities` and `existing_relationships` context fields are good additions for merge guidance, but they don't replace the directional focus.

> [!CAUTION]
> **Impact:** Graph extractions in LARPAn1 may be unfocused and diffuse, extracting entities/relationships about random participants rather than deeply profiling the target user.

**Fix:** Add `target_focus` as an input field to `GraphExtractionSignature` and pass it from `_evolve_graph()`.

---

### C3. `group_memory` Collection Completely Missing from vRAG Mode

**Files:** [repositories.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/db/repositories.py), [background.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/tasks/background.py)

The legacy roastbot uses **6 collections**:
1. `chat_history` ✅ (in LARPAn1)
2. `user_memory` ✅ (in LARPAn1 via `MemoryRepository`)
3. `group_history` ✅ (in LARPAn1)
4. `group_memory` ❌ **MISSING from LARPAn1**
5. `global_history` ✅ (in LARPAn1)
6. `global_memory` ✅ (in LARPAn1)

The legacy roastbot stores group summaries in a **separate `group_memory` collection** (see line 138: `group_memory_col = db["group_memory"]`). In LARPAn1, the roastbot background task at line 274 calls:
```python
await _evolve_text_profile(group_name, group_history, memory_repo, is_global=False, is_group=True)
```
This passes the `MemoryRepository` (which maps to `user_memory` collection) to store group profiles. **Group summaries are being written to the `user_memory` collection instead of `group_memory`**.

Since the group key is the bare `group_name` string (e.g., `"6b6t"`) and user keys are `"{group}:{username}"`, there's no collision per se, but:

> [!WARNING]
> **Impact:** Group summaries from the legacy roastbot stored in `group_memory` are **invisible** to LARPAn1. The engine is reading from `user_memory` for group profiles, which will always be empty for existing groups. All legacy group intelligence is effectively orphaned.

**Fix:** Create a dedicated `GroupMemoryRepository` pointing to `group_memory` and pass it to `_evolve_text_profile` when `is_group=True`.

---

### C4. `discord_dm` → `private_chat` Normalization Missing from LARPAn1

**Files:** [routes.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/api/routes.py), [models.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/api/models.py)

The legacy roastbot has explicit normalization at line 738:
```python
if group_name.lower() in ["defaultgroup", "discord_dm"]:
    group_name = "private_chat"
```

The legacy vRAG does the same at line 621:
```python
if group_name.lower() in ["defaultgroup", "discord_dm"]:
    group_name = "private_chat"
```

**LARPAn1 does NOT have this normalization anywhere.** The routes.py checks `payload.group_name == "private_chat"` at line 41, and the engines check for `"private_chat"` — but if a bridge sends `"discord_dm"` or `"DefaultGroup"` (which bridges actually send), the payload passes Pydantic validation but the engine never detects it as private.

> [!CAUTION]
> **Impact:** DMs from Discord (which send `group_name: "discord_dm"`) will be treated as **group messages**, causing: (1) no forced reply, (2) group history storage instead of private history, (3) incorrect user profile keying.

**Fix:** Add normalization in `routes.py` before dispatch:
```python
if payload.group_name.lower() in ["defaultgroup", "discord_dm"]:
    payload.group_name = "private_chat"
```

---

## 🟡 Moderate Issues

---

### M1. Roastbot Engine Missing `display_name` in Tagged Profile Fallback

**Files:** [roastbot.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/engine/roastbot.py#L23-L37) vs [legacy roastbot main.py](file:///home/qop/projects/PSI-CORE/psi-09-roastbot/main.py#L407-L418)

In the legacy roastbot, `fetch_tagged_profiles` does NOT have a fallback message for users with no profile — it simply skips them:
```python
if summary:
    profiles.append(...)
# No else branch — silent skip
```

LARPAn1's `_fetch_tagged_profiles` adds a fallback:
```python
else:
    profiles.append(f'<bystander ...>No intelligence gathered yet. Default to standard mockery.</bystander>')
```

This is a **deliberate improvement**, not a bug. However, this means the LLM now receives an explicit instruction to mock tagged users even with zero data. This changes the combat behavior from "ignore unknown bystanders" to "mock them generically."

> [!NOTE]
> **Impact:** Behavioral drift — the LLM may now mock tagged users with generic insults rather than ignoring them. This could be considered a feature, but it's a departure from legacy behavior. Confirming intentionality is recommended.

---

### M2. `max_tokens` Mismatch Between Legacy vRAG NVIDIA Pool and LARPAn1

**Files:** [llm_balancer.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/core/llm_balancer.py#L49) vs [legacy vRAG main.py](file:///home/qop/projects/PSI-CORE/PSI-09-vRAG/main.py#L101)

Legacy vRAG NVIDIA pool: `max_tokens=512`
LARPAn1 NVIDIA pool: `max_tokens=1024`

Legacy roastbot NVIDIA HTTP payload: `max_tokens=1024` (line 172)

The doubling from 512 → 1024 for the vRAG combat path means the DSPy pipeline may generate longer responses than the legacy vRAG ever did. Given the "MUST BE UNDER 150 CHARACTERS" constraint in the signatures, this likely doesn't change output length, but it **doubles the maximum potential token cost per request** and could impact throughput on the free tier.

> [!WARNING]
> **Impact:** Increased token consumption per vRAG combat request compared to legacy vRAG. May accelerate rate limiting under heavy traffic.

---

### M3. Missing `prompts_high.py` and `prompts_roleplay.py` Capability

**Files:** [prompts_high.py](file:///home/qop/projects/PSI-CORE/psi-09-roastbot/prompts_high.py), [prompts_roleplay.py](file:///home/qop/projects/PSI-CORE/psi-09-roastbot/prompts_roleplay.py)

The legacy roastbot has **three complete prompt sets**:
1. `prompts.py` — Standard combat prompts ✅ (ported to LARPAn1's `roastbot_prompts.py`)
2. `prompts_high.py` — "Hyper-cynical, ABSOLUTELY UNHINGED" variant with more nuanced constraints ("ROAST LIKE A MAN, USE WIT to ARTICULATE YOUR HATE") ❌ **Not ported**
3. `prompts_roleplay.py` — Roleplay jailbreak wrapper variant ("AUTHORIZED SCENARIO: MATURE SATIRICAL COMEDY SCRIPT") ❌ **Not ported**

While the legacy roastbot's `main.py` only imports from `prompts.py`, these alternate prompt sets represent available **combat modes** that were presumably selected manually or through configuration. LARPAn1 has no concept of prompt variants.

> [!IMPORTANT]
> **Impact:** Two alternate prompt strategies from the legacy roastbot are not portable to LARPAn1. If these were ever used in production or planned for future use, the capability is lost.

**Fix:** Consider adding a `prompt_variant` field to `IncomingPayload` or `Settings`, and loading the appropriate prompts accordingly.

---

### M4. vRAG Engine Doesn't Use `display_name` in Any Context

**Files:** [vrag.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/engine/vrag.py)

Both `_format_history` and `_format_graph` and the `initial_state` use `payload.username` exclusively. The `display_name` field — which bridges send and Pydantic validates as required — is never passed to the DSPy pipeline.

The legacy vRAG also doesn't use `display_name` in its combat pipeline, so this is consistent. However, the roastbot engine in LARPAn1 also doesn't use `display_name` in the LLM feed (it only stores it in MongoDB). The field is validated by Pydantic but functionally unused at the engine layer across both engines.

> [!NOTE]
> **Impact:** Minor — `display_name` is stored in history but never surfaced to the LLM. For platforms where display names differ from usernames (common on Discord), the LLM operates blind to the user's chosen display identity.

---

### M5. In-Memory Counter State Lost on Server Restart

**Files:** [background.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/tasks/background.py#L17-L28)

The message counters (`_msg_counters`) are stored in a Python `defaultdict` in memory. On Render free tier, the server sleeps after inactivity and restarts frequently.

The legacy roastbot has the same design (in-memory `msg_count` in `MongoCache`), so this is consistent — but both share the same vulnerability: **every restart resets all counters to 0**, meaning the next message from every user will NOT trigger First Contact (since profiles already exist) but will NOT trigger evolution either (counter starts at 0, needs to reach 50).

This means after a restart, the first 49 messages from every user are "blind" — no profiling runs. On the Render free tier where restarts are frequent, this creates significant profiling gaps.

> [!WARNING]
> **Impact:** Frequent Render restarts cause profiling dead zones. Could be mitigated by persisting counters to MongoDB, but the legacy engines have the same problem, so this is an inherited architectural limitation, not a regression.

---

### M6. Roastbot Engine Doesn't Strip Remaining Non-Bot Snowflakes from History

**Files:** [roastbot.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/engine/roastbot.py#L12-L19)

The `_clean_snowflakes` function in `roastbot.py` correctly replaces bot snowflakes with `@PSI-09` and strips remaining unknown snowflakes:
```python
text = re.sub(r'<@!?&?\d+>', '', text)
```

But this function is applied to:
1. The `payload.message` (line 71) ✅
2. Each history entry's `content` (line 61) ✅

However, the **routes.py** `_clean_discord_mentions` (line 22-27) only replaces bot snowflakes and does NOT strip remaining unknown ones. This means the **stored** messages in MongoDB will contain raw snowflakes like `<@123456789>` forever. The roastbot engine cleans them at read time, but the **vRAG engine does NOT have any snowflake cleaning** — `_format_history` in vrag.py (line 169) passes history content raw to the DSPy pipeline.

> [!WARNING]
> **Impact:** The vRAG combat engine receives raw Discord snowflake strings in its history and message fields, which the DSPy signatures may misinterpret as entity names or cause hallucinated graph entities.

**Fix:** Apply snowflake stripping in `routes.py` before storage, or add cleaning to the vRAG's `_format_history`.

---

## 🔵 Low-Severity / Code Quality Issues

---

### L1. `FailoverLMPool` Constructor Differs from Legacy vRAG

**Files:** [llm_balancer.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/core/llm_balancer.py#L12-L19) vs [legacy vRAG main.py](file:///home/qop/projects/PSI-CORE/PSI-09-vRAG/main.py#L72-L87)

The legacy vRAG `FailoverLMPool` takes a single `api_key: str` and creates one model per model name:
```python
self.models = [dspy.LM(model=f"groq/{m}", api_key=api_key) for m in model_names]
```

LARPAn1's `FailoverLMPool` takes `api_keys: list` and creates **one model per (key, model_name) combination**:
```python
for key in api_keys:
    self.models.extend([dspy.LM(model=f"groq/{m}", api_key=key.strip()) for m in model_names])
```

This is a **deliberate improvement** — the pool is now N×M instead of N. However, the `advance()` method still does linear cycling, which means failover order is `(key1,model1) → (key1,model2) → (key1,model3) → (key2,model1) → ...`. On a 429, you advance to the next slot which might be the **same key** with a different model. If the rate limit is per-key (common with Groq), this fails to actually escape the limit.

> [!TIP]
> **Suggestion:** Consider interleaving keys: `(key1,model1) → (key2,model1) → (key1,model2) → (key2,model2)` to maximize key rotation on rate limits.

---

### L2. `BaseEngine.engine_name()` Is a Method, Not a Property

**Files:** [base.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/engine/base.py#L19-L22)

`engine_name()` is defined as an abstract method but conceptually acts as a constant (returns `"roastbot"` or `"vrag"`). It should be a `@property` for cleaner usage:
```python
@property
@abstractmethod
def engine_name(self) -> str: ...
```

Minor style issue only — functionality is correct.

---

### L3. TTLCache `_cleanup` Called Only Inside `set()`, Never Independently

**Files:** [repositories.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/db/repositories.py#L9-L46)

The `_cleanup()` method is only called when the cache is full during a `set()`. Expired entries are only evicted when `get()` hits them individually. This means the cache can hold up to `max_size=1500` entries even if most are expired, consuming unnecessary memory on the 512MB Render instance.

> [!TIP]
> **Suggestion:** Consider periodic cleanup or using `get()` to evict expired entries proactively.

---

### L4. Redundant Snowflake Cleaning in Routes + Roastbot Engine

**Files:** [routes.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/api/routes.py#L22-L27), [roastbot.py](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/app/engine/roastbot.py#L12-L19)

`_clean_discord_mentions()` in routes.py and `_clean_snowflakes()` in roastbot.py both do bot-ID replacement. The message gets cleaned in routes.py first, then cleaned again inside roastbot.py. The second pass is redundant for the message itself (but still needed for history entries).

> [!TIP]
> **Suggestion:** Centralize all snowflake cleaning in a single utility function called at storage time.

---

### L5. SYSTEM_HANDOVER.md Is `.gitignore`'d

**Files:** [.gitignore](file:///home/qop/projects/PSI-CORE/psi-09-engine-v2/.gitignore#L18)

Line 18 of `.gitignore` excludes `SYSTEM_HANDOVER.md`. This means the handover document — the single most important architectural reference for the project — is **not tracked in version control** and will not be present in deployments or for new contributors.

> [!NOTE]
> **Suggestion:** Either remove it from `.gitignore` and commit it, or merge its contents into `README.md`.

---

## Feature Parity Matrix

| Capability | Legacy Roastbot | Legacy vRAG | LARPAn1 | Status |
|---|---|---|---|---|
| **API Framework** | Flask | Flask | FastAPI | ✅ Upgraded |
| **Async Execution** | Sync (threaded) | Sync (threaded) | Async (`asyncio.to_thread`) | ✅ Upgraded |
| **Pydantic Validation** | ❌ | Partial (DSPy only) | ✅ Full (Payload + Response + DSPy) | ✅ Upgraded |
| **CORS** | ✅ | ✅ | ✅ | ✅ |
| **Health Endpoint** | `GET /` | `GET /` | `GET /` | ✅ |
| **Snowflake Mention Cleaning** | ✅ (per-message) | ✅ (per-message) | ✅ (routes + engine) | ✅ |
| **DM Detection (`discord_dm` → `private_chat`)** | ✅ | ✅ | ❌ Missing | 🔴 C4 |
| **Think-Tag Sanitization** | ✅ (5-regex chain) | ❌ | ✅ (unified `sanitize_think_tags`) | ✅ |
| **User Chat History** | ✅ 6 collections | ✅ 4 collections | ✅ 8 collections (union) | ✅ |
| **Group Memory (text summaries)** | ✅ `group_memory` col | N/A (uses graphs) | ❌ Writes to wrong collection | 🔴 C3 |
| **Global Memory (cross-platform)** | ✅ | N/A | ✅ | ✅ |
| **Graph User Storage** | N/A | ✅ `graph_users` | ✅ `graph_users` | ✅ |
| **Graph Group Storage** | N/A | ✅ `graph_groups` | ✅ `graph_groups` | ✅ |
| **Tagged User Profiles** | ✅ (Roastbot) | N/A | ✅ (Roastbot) | ✅ |
| **Token Counting (HF Tokenizers)** | ✅ (model-specific) | ❌ | ❌ (word-count heuristic) | 🟡 Downgrade |
| **Model-Specific Tokenizer Loading** | ✅ (Kimi, Llama, GPT, Qwen, Gemma) | ❌ | ❌ | 🟡 Downgrade |
| **Exponential Backoff Retry** | ✅ (5xx handling) | ❌ (429 only) | ❌ (429 only) | 🟡 Partial |
| **Server Error (5xx) Retry** | ✅ (exponential backoff) | ❌ | ❌ | 🟡 Missing |
| **MongoDB Keepalive Thread** | ✅ | ❌ | ✅ | ✅ |
| **PageRank Convergence Guard** | N/A | ❌ (default params) | ✅ (`max_iter=500, tol=1e-6`) | ✅ Fixed |
| **CPU Guardian (500 node cap)** | N/A | ❌ | ✅ | ✅ Fixed |
| **Graph Merge (accumulative)** | N/A | ❌ (full replace) | ✅ (merge) | ✅ Upgraded |
| **Relationship Intensity Update** | N/A | ✅ (full replace) | ❌ (frozen at first value) | 🔴 C1 |
| **Graph `target_focus` Directive** | N/A | ✅ | ❌ Missing | 🔴 C2 |
| **Prompt Variants (high/roleplay)** | ✅ Available | N/A | ❌ Not ported | 🟡 M3 |
| **`display_name` Usage** | Stored only | Stored only | Stored only | ⚪ Consistent |
| **Background Task Gating** | ✅ (per-user counters) | ❌ (every message) | ✅ (per-user + per-group counters) | ✅ Fixed |
| **First Contact Logic** | ✅ | ❌ | ✅ (both engines) | ✅ Fixed |
| **LangGraph State Machine** | N/A | ✅ | ✅ | ✅ |
| **DSPy 4-Stage Pipeline** | N/A | ✅ | ✅ | ✅ |
| **Emoji Reaction Output** | ❌ | ✅ | ✅ | ✅ |
| **Mode Dispatcher (auto/legacy/vrag)** | N/A | N/A | ✅ | ✅ New |
| **Render YAML Deployment** | ❌ | ❌ | ✅ | ✅ New |

---

## Architectural Improvements in LARPAn1 (Successfully Ported)

These are things LARPAn1 does **better** than either legacy engine:

1. **FastAPI + async** replaces Flask's synchronous blocking — massive concurrency improvement
2. **Pydantic schemas** on the API layer (`IncomingPayload`, `EngineResponse`) catch malformed payloads at the boundary
3. **Repository pattern** cleanly abstracts all MongoDB operations
4. **Modular file structure** (api/core/db/engine/prompts/tasks) vs legacy monoliths
5. **Graph merge logic** accumulates entity knowledge over time instead of overwriting
6. **Unified think-tag sanitizer** replaces ad-hoc regex chains
7. **TTL cache with LRU eviction** replaces unbounded `MongoCache` dicts
8. **Background task gating** applies to BOTH engines (legacy roastbot had gating; legacy vRAG did not)
9. **First Contact** logic added to vRAG (legacy vRAG extracted on every message)
10. **Snowflake stripping** in graph extraction prevents junk entities (not present in legacy vRAG)

---

## Recommended Priority Actions

| Priority | Issue | Action |
|----------|-------|--------|
| **P0** | C4: `discord_dm` normalization | Add `group_name` normalization in routes.py |
| **P0** | C3: Group memory collection | Create `GroupMemoryRepository` → `group_memory` |
| **P1** | C1: Frozen relationship intensities | Update merge logic to refresh intensities |
| **P1** | C2: Missing `target_focus` | Add `target_focus` to `GraphExtractionSignature` |
| **P1** | M6: vRAG snowflake contamination | Clean snowflakes before storage or in vRAG formatter |
| **P2** | M2: max_tokens 512 → 1024 | Evaluate if 512 is sufficient for DSPy pipeline |
| **P2** | M5: Counter persistence | Accept risk or persist to MongoDB |
| **P3** | L1: Key interleaving in FailoverPool | Restructure model ordering |
| **P3** | L5: SYSTEM_HANDOVER in .gitignore | Remove from .gitignore |
