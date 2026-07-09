# LARPAn1 Audit — Fix Execution Tasks

## P0 — Critical (Functional Regressions)

- [ ] **C4:** Add `discord_dm` / `DefaultGroup` → `private_chat` normalization in `routes.py`
- [ ] **C3:** Create `GroupMemoryRepository` pointing to `group_memory` collection; wire it into `background.py`

## P1 — Critical (Data Quality)

- [ ] **C1:** Fix frozen relationship intensities in graph merge logic (`background.py`)
- [ ] **C2:** Add `target_focus` input field to `GraphExtractionSignature`; pass it from `_evolve_graph()`
- [ ] **M6:** Clean snowflakes from vRAG history formatter + clean before MongoDB storage in `routes.py`

## P2 — Moderate

- [ ] **M2:** Reduce `max_tokens` in NVIDIA pool from 1024 → 512 to match legacy vRAG
- [ ] **L1:** Interleave keys in `FailoverLMPool` to maximize key rotation on rate limits

## P3 — Low

- [ ] **L5:** Remove `SYSTEM_HANDOVER.md` from `.gitignore`
- [ ] **L4:** Centralize snowflake cleaning into `utils.py`
