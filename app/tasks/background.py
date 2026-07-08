import logging
import asyncio
import dspy
from app.core.llm_balancer import background_pool
from app.db.repositories import ChatRepository, GroupHistoryRepository, MemoryRepository, GraphRepository, GlobalHistoryRepository, GlobalMemoryRepository
from app.prompts.roastbot_prompts import FIRST_CONTACT_PROMPT, EVOLUTION_PROMPT, GLOBAL_FIRST_CONTACT_PROMPT, GLOBAL_EVOLUTION_PROMPT
from app.prompts.dspy_signatures import GraphExtractionSignature

logger = logging.getLogger(__name__)

async def _evolve_graph(entity_key: str, history_docs: list, graph_repo: GraphRepository, is_user: bool = True):
    if len(history_docs) < 2:
        return
        
    history_str = "\n".join([f"[{m.get('username', 'Unknown')}]: {m.get('content', '')}" for m in history_docs])
    
    existing_graph = graph_repo.get_user_graph(entity_key) if is_user else graph_repo.get_group_graph(entity_key)
    
    existing_entities_str = ", ".join([e.get("id", "") for e in existing_graph.get("entities", []) if isinstance(e, dict)])
    if not existing_entities_str:
        existing_entities_str = ", ".join([e for e in existing_graph.get("entities", []) if isinstance(e, str)])

    existing_rels_str = ", ".join([f"{r['source']} {r['relation']} {r['target']} (Intensity: {r.get('intensity', 5.0)})" for r in existing_graph.get("relationships", [])])
    
    extractor = dspy.Predict(GraphExtractionSignature)
    max_retries = len(background_pool.models) if background_pool.models else 1
    new_graph_data = None
    
    for attempt in range(max_retries):
        try:
            current_lm, current_index = background_pool.get_current()
            with dspy.context(lm=current_lm):
                res = await asyncio.to_thread(
                    extractor,
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
            else:
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
        seen_rels = {(r["source"], r["relation"], r["target"]) for r in existing_rels}
        
        for rel in new_graph_data.relationships:
            rel_tuple = (rel.source, rel.relation, rel.target)
            if rel_tuple not in seen_rels:
                seen_rels.add(rel_tuple)
                existing_rels.append({
                    "source": rel.source, 
                    "relation": rel.relation, 
                    "target": rel.target,
                    "intensity": float(rel.intensity)
                })
                
        updated_graph = {
            "entities": merged_entities,
            "relationships": existing_rels
        }
        
        if is_user:
            graph_repo.update_user_graph(entity_key, updated_graph)
        else:
            graph_repo.update_group_graph(entity_key, updated_graph)
            
        logger.info(f"[BACKGROUND] Successfully extracted and updated graph for {entity_key}.")

async def _evolve_text_profile(entity_key: str, history_docs: list, memory_repo, is_global: bool = False):
    if len(history_docs) < 5:
        return
        
    history_str = "\n".join([f"[{m.get('username', 'Unknown')}]: {m.get('content', '')}" for m in history_docs])
    old_summary = memory_repo.get_profile(entity_key)
    
    prompt = ""
    if not old_summary:
        if is_global:
            prompt = GLOBAL_FIRST_CONTACT_PROMPT + f"\n\n<chat_history>\n{history_str}\n</chat_history>"
        else:
            prompt = FIRST_CONTACT_PROMPT + f"\n\n<chat_history>\n{history_str}\n</chat_history>"
    else:
        if is_global:
            prompt = GLOBAL_EVOLUTION_PROMPT.replace("{old_summary}", old_summary) + f"\n\n<chat_history>\n{history_str}\n</chat_history>"
        else:
            prompt = EVOLUTION_PROMPT.replace("{old_summary}", old_summary) + f"\n\n<chat_history>\n{history_str}\n</chat_history>"
        
    max_retries = len(background_pool.models) if background_pool.models else 1
    new_profile = ""
    
    for attempt in range(max_retries):
        try:
            current_lm, current_index = background_pool.get_current()
            res = await asyncio.to_thread(current_lm, prompt)
            
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
        memory_repo.update_profile(entity_key, new_profile)
        logger.info(f"[BACKGROUND] Successfully evolved profile for {entity_key}.")
    else:
        logger.warning(f"[BACKGROUND] Failed to generate profile for {entity_key}.")


async def evolve_profile_task(user_key: str, group_name: str, global_key: str, mode: str):
    """
    Background task to evolve the user's local memory profile and the group's global profile.
    """
    chat_repo = ChatRepository()
    group_repo = GroupHistoryRepository()
    global_history_repo = GlobalHistoryRepository()
    
    user_history = chat_repo.get_recent_history(user_key, limit=30)
    group_history = group_repo.get_recent_history(group_name, limit=80)
    global_history = global_history_repo.get_recent_history(global_key, limit=50)
    
    if mode == "vrag":
        graph_repo = GraphRepository()
        
        # Evolve User Graph
        await _evolve_graph(user_key, user_history, graph_repo, is_user=True)
        
        # Evolve Group Graph
        if group_name != "private_chat":
            await _evolve_graph(group_name, group_history, graph_repo, is_user=False)
            
    else:
        # Legacy Roastbot Text Profiling
        memory_repo = MemoryRepository()
        global_memory_repo = GlobalMemoryRepository()
        
        # Evolve User Profile
        await _evolve_text_profile(user_key, user_history, memory_repo, is_global=False)
        
        # Evolve Group Profile
        if group_name != "private_chat":
            await _evolve_text_profile(group_name, group_history, memory_repo, is_global=False)
            
        # Evolve Global Profile
        await _evolve_text_profile(global_key, global_history, global_memory_repo, is_global=True)
