import logging
import asyncio
import dspy
from app.core.llm_balancer import background_pool
from app.db.repositories import ChatRepository, MemoryRepository, GraphRepository
from app.prompts.roastbot_prompts import FIRST_CONTACT_PROMPT, EVOLUTION_PROMPT
from app.prompts.dspy_signatures import GraphExtractionSignature

logger = logging.getLogger(__name__)

async def evolve_profile_task(user_key: str, mode: str):
    """
    Background task to evolve the user's local memory profile.
    This runs asynchronously and does not block the API response.
    """
    if mode == "vrag":
        chat_repo = ChatRepository()
        graph_repo = GraphRepository()
        
        recent_history = chat_repo.get_recent_history(user_key, limit=30)
        if len(recent_history) < 2:
            return
            
        history_str = "\n".join([f"[{m.get('username', 'Unknown')}]: {m.get('content', '')}" for m in recent_history])
        
        existing_graph = graph_repo.get_user_graph(user_key)
        existing_entities_str = ", ".join([e.get("id", "") for e in existing_graph.get("entities", []) if isinstance(e, dict)])
        if not existing_entities_str:
            # Fallback for old string lists during transition
            existing_entities_str = ", ".join([e for e in existing_graph.get("entities", []) if isinstance(e, str)])

        existing_rels_str = ", ".join([f"{r['source']} {r['relation']} {r['target']} (Intensity: {r.get('intensity', 5.0)})" for r in existing_graph.get("relationships", [])])
        
        extractor = dspy.TypedPredictor(GraphExtractionSignature)
        
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
            
            graph_repo.update_user_graph(user_key, updated_graph)
            logger.info(f"[BACKGROUND] Successfully extracted and updated graph for {user_key}.")
        return
        
    # Legacy Roastbot Text Profiling
    chat_repo = ChatRepository()
    memory_repo = MemoryRepository()
    
    recent_history = chat_repo.get_recent_history(user_key, limit=30)
    if len(recent_history) < 5:
        # Not enough data to bother profiling yet
        return
        
    history_str = "\n".join([f"[{m.get('username', 'Unknown')}]: {m.get('content', '')}" for m in recent_history])
    old_summary = memory_repo.get_profile(user_key)
    
    prompt = ""
    if not old_summary:
        prompt = FIRST_CONTACT_PROMPT + f"\n\n<chat_history>\n{history_str}\n</chat_history>"
    else:
        prompt = EVOLUTION_PROMPT.replace("{old_summary}", old_summary) + f"\n\n<chat_history>\n{history_str}\n</chat_history>"
        
    max_retries = len(background_pool.models) if background_pool.models else 1
    new_profile = ""
    
    for attempt in range(max_retries):
        try:
            current_lm, current_index = background_pool.get_current()
            # Run the synchronous generation in a background thread
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
        memory_repo.update_profile(user_key, new_profile)
        logger.info(f"[BACKGROUND] Successfully evolved profile for {user_key}.")
    else:
        logger.warning(f"[BACKGROUND] Failed to generate profile for {user_key}.")
