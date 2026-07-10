import logging
import asyncio
import threading
import re
import dspy
from collections import defaultdict
from datetime import datetime, timezone
from app.core.config import settings
from app.core.llm_balancer import background_pool
from app.core.utils import sanitize_think_tags
from app.db.repositories import ChatRepository, GroupHistoryRepository, MemoryRepository, GroupMemoryRepository, GraphRepository, GlobalHistoryRepository, GlobalMemoryRepository, CounterRepository
from app.prompts.roastbot_prompts import FIRST_CONTACT_PROMPT, EVOLUTION_PROMPT, GROUP_SUMMARY_PROMPT, GLOBAL_FIRST_CONTACT_PROMPT, GLOBAL_EVOLUTION_PROMPT
from app.prompts.dspy_signatures import GraphExtractionSignature

logger = logging.getLogger(__name__)

async def _evolve_graph(entity_key: str, history_docs: list, graph_repo: GraphRepository, is_user: bool = True):
    if not history_docs:
        return
        
    history_str = "\n".join([f"[{m.get('username', 'Unknown')}]: {m.get('content', '')}" for m in history_docs])

    
    if is_user:
        existing_graph = await asyncio.to_thread(graph_repo.get_user_graph, entity_key)
    else:
        existing_graph = await asyncio.to_thread(graph_repo.get_group_graph, entity_key)
    
    existing_entities_str = ", ".join([e.get("id", "") for e in existing_graph.get("entities", []) if isinstance(e, dict)])
    if not existing_entities_str:
        existing_entities_str = ", ".join([e for e in existing_graph.get("entities", []) if isinstance(e, str)])

    existing_rels_str = ", ".join([f"{r['source']} {r['relation']} {r['target']} (Intensity: {r.get('intensity', 5.0)})" for r in existing_graph.get("relationships", [])])
    
    if is_user:
        username = entity_key.split(":", 1)[1] if ":" in entity_key else entity_key
        target_focus_str = f"Deep psychological profile of user: {username}"
    else:
        target_focus_str = "Map the social dynamics, relationships, and alliances between all active users."
    
    extractor = dspy.Predict(GraphExtractionSignature)
    max_retries = len(background_pool.models) if background_pool.models else 1
    new_graph_data = None
    
    for attempt in range(max_retries):
        try:
            current_lm, current_index = background_pool.get_current()
            with dspy.context(lm=current_lm):
                res = await asyncio.to_thread(
                    extractor,
                    target_focus=target_focus_str,
                    chat_history=history_str,
                    existing_entities=existing_entities_str or "None",
                    existing_relationships=existing_rels_str or "None"
                )
            new_graph_data = res.extracted_graph
            break
        except Exception as e:
            logger.error(f"[BACKGROUND] Graph extraction failed: {e}")
            if "429" in str(e) or "rate limit" in str(e).lower():
                background_pool.advance(current_index)
            else:
                break
                
    if new_graph_data:
        existing_entities_dict = {}
        for e in existing_graph.get("entities", []):
            if isinstance(e, str):
                existing_entities_dict[e] = {"id": e, "type": "Unknown", "attributes": ""}
            elif isinstance(e, dict):
                # Purge the eager lock stub to prevent data pollution
                if e.get("attributes") == "Initializing...":
                    continue
                existing_entities_dict[e["id"]] = e
        
        for new_ent in new_graph_data.entities:
            if new_ent.id in existing_entities_dict:
                existing_attrs = existing_entities_dict[new_ent.id].get("attributes", "")
                if new_ent.attributes and new_ent.attributes not in existing_attrs:
                    existing_entities_dict[new_ent.id]["attributes"] = f"{existing_attrs} | {new_ent.attributes}".strip(" |")
            else:
                existing_entities_dict[new_ent.id] = {
                    "id": new_ent.id,
                    "type": new_ent.type,
                    "attributes": new_ent.attributes
                }
        
        merged_entities = list(existing_entities_dict.values())
        
        existing_rels = existing_graph.get("relationships", [])
        seen_rels = {(r["source"], r["relation"], r["target"]): i for i, r in enumerate(existing_rels)}
        
        for rel in new_graph_data.relationships:
            rel_tuple = (rel.source, rel.relation, rel.target)
            if rel_tuple in seen_rels:
                idx = seen_rels[rel_tuple]
                existing_rels[idx]["intensity"] = float(rel.intensity)
            else:
                seen_rels[rel_tuple] = len(existing_rels)
                existing_rels.append({
                    "source": rel.source, 
                    "relation": rel.relation, 
                    "target": rel.target,
                    "intensity": float(rel.intensity)
                })
                
        updated_graph = {
            "entities": merged_entities,
            "relationships": existing_rels,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        if is_user:
            await asyncio.to_thread(graph_repo.update_user_graph, entity_key, updated_graph)
        else:
            await asyncio.to_thread(graph_repo.update_group_graph, entity_key, updated_graph)
            
        logger.info(f"[BACKGROUND] Successfully extracted and updated graph for {entity_key}.")
    else:
        # First Contact Fallback: If extraction fails or yields nothing, inject a stub to break infinite loops.
        entities = existing_graph.get("entities", [])
        is_empty = not entities and not existing_graph.get("relationships")
        is_only_stub = len(entities) == 1 and isinstance(entities[0], dict) and entities[0].get("attributes") == "Initializing..."
        
        if is_empty or is_only_stub:
            logger.warning(f"[BACKGROUND] Empty vRAG extraction for {entity_key}. Injecting First Contact stub.")
            stub_graph = {
                "entities": [{"id": "System", "type": "Metadata", "attributes": "Initialized without entities."}],
                "relationships": [],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            if is_user:
                await asyncio.to_thread(graph_repo.update_user_graph, entity_key, stub_graph)
            else:
                await asyncio.to_thread(graph_repo.update_group_graph, entity_key, stub_graph)

async def _evolve_text_profile(entity_key: str, history_docs: list, memory_repo, is_global: bool = False, is_group: bool = False):
    # For individual profiles, we purely want the user's messages.
    # For group profiles, we want everything, but we should make sure the LLM knows which is which.
    if is_group:
        filtered_docs = history_docs
    else:
        filtered_docs = [m for m in history_docs if m.get('role') == 'user']
        
    if not filtered_docs:
        return
        
    old_summary = await asyncio.to_thread(memory_repo.get_profile, entity_key)
    
    # Handle New Group Stub
    if is_group and len(filtered_docs) < 6:
        stub = f"New group '{entity_key}' — Understand group dynamic and log observations."
        await asyncio.to_thread(memory_repo.update_profile, entity_key, stub)
        logger.info(f"[BACKGROUND] Initialized group stub for {entity_key}")
        return
        
    history_lines = []
    for m in filtered_docs:
        sender = "PSI-09" if m.get('role') == 'assistant' else m.get('username', 'Unknown')
        history_lines.append(f"[{sender}]: {m.get('content', '')}")
    history_str = "\n".join(history_lines)
    
    system_prompt = ""
    user_prompt = ""
    
    if is_group:
        # Groups always use the dedicated surveillance prompt — no first-contact distinction (matching legacy)
        system_prompt = GROUP_SUMMARY_PROMPT
        user_prompt = f"<chat_history>\n{history_str}\n</chat_history>"
    elif not old_summary or old_summary == "[INITIALIZING]":
        if is_global:
            system_prompt = GLOBAL_FIRST_CONTACT_PROMPT
            user_prompt = f"<cross_platform_history>\n{history_str}\n</cross_platform_history>"
        else:
            system_prompt = FIRST_CONTACT_PROMPT
            user_prompt = f"<chat_history>\n{history_str}\n</chat_history>"
    else:
        if is_global:
            system_prompt = GLOBAL_EVOLUTION_PROMPT.replace("{old_summary}", old_summary)
            user_prompt = f"<cross_platform_history>\n{history_str}\n</cross_platform_history>"
        else:
            system_prompt = EVOLUTION_PROMPT.replace("{old_summary}", old_summary)
            user_prompt = f"<chat_history>\n{history_str}\n</chat_history>"
            
    # Structured chat completion feed (Fix #3)
    llm_feed = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
        
    max_retries = len(background_pool.models) if background_pool.models else 1
    new_profile = ""
    
    for attempt in range(max_retries):
        try:
            current_lm, current_index = background_pool.get_current()
            # Send structured chat payload to Groq
            res = await asyncio.to_thread(current_lm, messages=llm_feed)
            
            if isinstance(res, list) and len(res) > 0:
                new_profile = res[0]
            else:
                new_profile = str(res)
                
            break
        except Exception as e:
            logger.error(f"[BACKGROUND] Profiling failed: {e}")
            if "429" in str(e) or "rate limit" in str(e).lower():
                background_pool.advance(current_index)
            else:
                break
                
    if new_profile:
        new_profile = sanitize_think_tags(new_profile)
        await asyncio.to_thread(memory_repo.update_profile, entity_key, new_profile)
        logger.info(f"[BACKGROUND] Successfully evolved profile for {entity_key}.")
    else:
        logger.warning(f"[BACKGROUND] Failed to generate profile for {entity_key}.")
        # First Contact Fallback: Break the infinite loop if LLM fails on first contact
        if not old_summary:
            logger.warning(f"[BACKGROUND] Injecting fallback profile stub for {entity_key}.")
            await asyncio.to_thread(memory_repo.update_profile, entity_key, "Insufficient data for initial profile.")


async def evolve_profile_task(user_key: str, group_name: str, global_key: str, mode: str):
    """
    Background task with message-count gating.
    Profiles only evolve after every N messages, matching the old psi-09-roastbot behavior.
    """
    chat_repo = ChatRepository()
    group_repo = GroupHistoryRepository()
    global_history_repo = GlobalHistoryRepository()
    counter_repo = CounterRepository()
    
    if mode in ["vrag", "auto"]:
        # vRAG uses graph extraction — gate per user and group
        user_count = counter_repo.record_activity(f"vrag:{user_key}")
        graph_repo = GraphRepository()
        
        # Check for First Contact
        existing_user_graph = await asyncio.to_thread(graph_repo.get_user_graph, user_key)
        is_first_contact = not existing_user_graph.get("entities") and not existing_user_graph.get("relationships")
        
        if is_first_contact:
            logger.info(f"[BACKGROUND] First Contact Graph Extraction triggered for {user_key}")
            # Eagerly inject stub to lock concurrent first contacts
            await asyncio.to_thread(graph_repo.update_user_graph, user_key, {"entities": [{"id": "System", "type": "Metadata", "attributes": "Initializing..."}], "relationships": []})
            evolution_time = datetime.now(timezone.utc)
            counter_repo.record_evolution(f"vrag:{user_key}", timestamp=evolution_time)
            user_history = await asyncio.to_thread(chat_repo.get_recent_history, user_key, limit=30)
            await _evolve_graph(user_key, user_history, graph_repo, is_user=True)
        elif user_count == settings.EVOLVE_EVERY_N_MESSAGES:
            logger.info(f"[BACKGROUND] Evolution Graph Extraction triggered for {user_key} (count={user_count})")
            evolution_time = datetime.now(timezone.utc)
            counter_repo.record_evolution(f"vrag:{user_key}", timestamp=evolution_time)
            user_history = await asyncio.to_thread(chat_repo.get_recent_history, user_key, limit=30)
            await _evolve_graph(user_key, user_history, graph_repo, is_user=True)
        
        if group_name != "private_chat":
            group_count = counter_repo.record_activity(f"vrag_group:{group_name}")
            existing_group_graph = await asyncio.to_thread(graph_repo.get_group_graph, group_name)
            is_group_first_contact = not existing_group_graph.get("entities") and not existing_group_graph.get("relationships")
            
            if is_group_first_contact and group_count == 10:
                logger.info(f"[BACKGROUND] vRAG Group First Contact triggered for {group_name}")
                await asyncio.to_thread(graph_repo.update_group_graph, group_name, {"entities": [{"id": "System", "type": "Metadata", "attributes": "Initializing..."}], "relationships": []})
                evolution_time = datetime.now(timezone.utc)
                counter_repo.record_evolution(f"vrag_group:{group_name}", timestamp=evolution_time)
                group_history = await asyncio.to_thread(group_repo.get_recent_history, group_name, limit=30)
                await _evolve_graph(group_name, group_history, graph_repo, is_user=False)
            elif group_count == settings.GROUP_SUMMARY_EVERY_N:
                logger.info(f"[BACKGROUND] vRAG Group Summary triggered for {group_name} (count={group_count})")
                evolution_time = datetime.now(timezone.utc)
                counter_repo.record_evolution(f"vrag_group:{group_name}", timestamp=evolution_time)
                group_history = await asyncio.to_thread(group_repo.get_recent_history, group_name, limit=80)
                await _evolve_graph(group_name, group_history, graph_repo, is_user=False)
            
    else:
        # Legacy Roastbot Text Profiling — gate per user, group, and global
        memory_repo = MemoryRepository()
        group_memory_repo = GroupMemoryRepository()
        global_memory_repo = GlobalMemoryRepository()
        
        # --- User Profile ---
        user_count = counter_repo.record_activity(f"rb:{user_key}")
        existing_user_profile = await asyncio.to_thread(memory_repo.get_profile, user_key)
        
        if existing_user_profile is None or existing_user_profile == "":
            # Eagerly write lock stub
            await asyncio.to_thread(memory_repo.update_profile, user_key, "[INITIALIZING]")
            logger.info(f"[BACKGROUND] First Contact triggered for {user_key}")
            evolution_time = datetime.now(timezone.utc)
            counter_repo.record_evolution(f"rb:{user_key}", timestamp=evolution_time)
            user_history = await asyncio.to_thread(chat_repo.get_recent_history, user_key, limit=30)
            await _evolve_text_profile(user_key, user_history, memory_repo, is_global=False)
        elif user_count == settings.EVOLVE_EVERY_N_MESSAGES:
            logger.info(f"[BACKGROUND] Evolution triggered for {user_key} (count={user_count})")
            evolution_time = datetime.now(timezone.utc)
            counter_repo.record_evolution(f"rb:{user_key}", timestamp=evolution_time)
            user_history = await asyncio.to_thread(chat_repo.get_recent_history, user_key, limit=30)
            await _evolve_text_profile(user_key, user_history, memory_repo, is_global=False)
        
        # --- Group Profile ---
        if group_name != "private_chat":
            group_count = counter_repo.record_activity(f"rb_group:{group_name}")
            existing_group_profile = await asyncio.to_thread(group_memory_repo.get_profile, group_name)
            
            if existing_group_profile is None or existing_group_profile == "":
                # Wait for 10 messages to bypass the legacy < 6 stub logic and guarantee a real LLM profile
                if group_count == 10:
                    await asyncio.to_thread(group_memory_repo.update_profile, group_name, "[INITIALIZING]")
                    logger.info(f"[BACKGROUND] Group First Contact triggered for {group_name}")
                    evolution_time = datetime.now(timezone.utc)
                    counter_repo.record_evolution(f"rb_group:{group_name}", timestamp=evolution_time)
                    group_history = await asyncio.to_thread(group_repo.get_recent_history, group_name, limit=30)
                    await _evolve_text_profile(group_name, group_history, group_memory_repo, is_global=False, is_group=True)
            elif group_count == settings.GROUP_SUMMARY_EVERY_N:
                logger.info(f"[BACKGROUND] Group Summary triggered for {group_name} (count={group_count})")
                evolution_time = datetime.now(timezone.utc)
                counter_repo.record_evolution(f"rb_group:{group_name}", timestamp=evolution_time)
                group_history = await asyncio.to_thread(group_repo.get_recent_history, group_name, limit=80)
                await _evolve_text_profile(group_name, group_history, group_memory_repo, is_global=False, is_group=True)
        
        # --- Global Profile ---
        global_count = counter_repo.record_activity(f"rb_global:{global_key}")
        existing_global_profile = await asyncio.to_thread(global_memory_repo.get_profile, global_key)
        
        if existing_global_profile is None or existing_global_profile == "":
            await asyncio.to_thread(global_memory_repo.update_profile, global_key, "[INITIALIZING]")
            logger.info(f"[BACKGROUND] Global First Contact triggered for {global_key}")
            evolution_time = datetime.now(timezone.utc)
            counter_repo.record_evolution(f"rb_global:{global_key}", timestamp=evolution_time)
            global_history = await asyncio.to_thread(global_history_repo.get_recent_history, global_key, limit=50)
            await _evolve_text_profile(global_key, global_history, global_memory_repo, is_global=True)
        elif global_count == settings.EVOLVE_EVERY_N_MESSAGES:
            logger.info(f"[BACKGROUND] Global Evolution triggered for {global_key} (count={global_count})")
            evolution_time = datetime.now(timezone.utc)
            counter_repo.record_evolution(f"rb_global:{global_key}", timestamp=evolution_time)
            global_history = await asyncio.to_thread(global_history_repo.get_recent_history, global_key, limit=50)
            await _evolve_text_profile(global_key, global_history, global_memory_repo, is_global=True)

async def hourly_sweep_task():
    """
    Called by Fastcron hourly. Finds all entities that had activity since their last evolution
    and forces an immediate evolution to catch up 'slow-burn' profiles.
    Applies jitter to prevent 429 rate limit spikes.
    """
    counter_repo = CounterRepository()
    chat_repo = ChatRepository()
    group_repo = GroupHistoryRepository()
    global_history_repo = GlobalHistoryRepository()
    
    graph_repo = GraphRepository()
    memory_repo = MemoryRepository()
    group_memory_repo = GroupMemoryRepository()
    global_memory_repo = GlobalMemoryRepository()

    candidates = await asyncio.to_thread(counter_repo.get_sweep_candidates)
    logger.info(f"[SWEEP] Found {len(candidates)} candidates for time-based evolution.")
    
    for candidate in candidates:
        key = candidate["_id"]
        
        try:
            sweep_time = datetime.now(timezone.utc)
            if key.startswith("vrag:"):
                user_key = key.split("vrag:", 1)[1]
                logger.info(f"[SWEEP] Evolving vRAG User {user_key}")
                history = await asyncio.to_thread(chat_repo.get_recent_history, user_key, limit=30)
                await _evolve_graph(user_key, history, graph_repo, is_user=True)
                counter_repo.record_evolution(key, timestamp=sweep_time)
                
            elif key.startswith("vrag_group:"):
                group_name = key.split("vrag_group:", 1)[1]
                logger.info(f"[SWEEP] Evolving vRAG Group {group_name}")
                history = await asyncio.to_thread(group_repo.get_recent_history, group_name, limit=80)
                await _evolve_graph(group_name, history, graph_repo, is_user=False)
                counter_repo.record_evolution(key, timestamp=sweep_time)
                
            elif key.startswith("rb:"):
                user_key = key.split("rb:", 1)[1]
                logger.info(f"[SWEEP] Evolving Legacy User {user_key}")
                history = await asyncio.to_thread(chat_repo.get_recent_history, user_key, limit=30)
                await _evolve_text_profile(user_key, history, memory_repo, is_global=False)
                counter_repo.record_evolution(key, timestamp=sweep_time)
                
            elif key.startswith("rb_group:"):
                group_name = key.split("rb_group:", 1)[1]
                logger.info(f"[SWEEP] Evolving Legacy Group {group_name}")
                history = await asyncio.to_thread(group_repo.get_recent_history, group_name, limit=80)
                await _evolve_text_profile(group_name, history, group_memory_repo, is_global=False, is_group=True)
                counter_repo.record_evolution(key, timestamp=sweep_time)
                
            elif key.startswith("rb_global:"):
                global_key = key.split("rb_global:", 1)[1]
                logger.info(f"[SWEEP] Evolving Legacy Global {global_key}")
                history = await asyncio.to_thread(global_history_repo.get_recent_history, global_key, limit=50)
                await _evolve_text_profile(global_key, history, global_memory_repo, is_global=True)
                counter_repo.record_evolution(key, timestamp=sweep_time)
        except Exception as e:
            logger.error(f"[SWEEP] Failed to evolve candidate {key}: {e}")
            
        # Jitter to avoid hammering APIs
        await asyncio.sleep(2)
        
    logger.info("[SWEEP] Completed hourly evolution sweep.")
