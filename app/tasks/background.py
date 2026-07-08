import logging
import asyncio
from app.core.llm_balancer import background_pool
from app.db.repositories import ChatRepository, MemoryRepository
from app.prompts.roastbot_prompts import FIRST_CONTACT_PROMPT, EVOLUTION_PROMPT

logger = logging.getLogger(__name__)

async def evolve_profile_task(user_key: str, mode: str):
    """
    Background task to evolve the user's local memory profile.
    This runs asynchronously and does not block the API response.
    """
    if mode == "vrag":
        # In a fully fleshed vRAG system, this would extract Graph Entities and Relationships.
        # For now, we simulate the completion of a graph extraction task.
        logger.info(f"[BACKGROUND] Graph extraction triggered for {user_key} (vRAG mode).")
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
