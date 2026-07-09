from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.api.models import IncomingPayload, EngineResponse
from app.engine.dispatcher import dispatcher
from app.tasks.background import evolve_profile_task
from app.db.repositories import ChatRepository, GroupHistoryRepository, GlobalHistoryRepository
from app.core.config import settings
from datetime import datetime, timezone
import asyncio
import re

router = APIRouter()

# Singleton repository instances to avoid re-instantiation per request
_chat_repo = ChatRepository()
_group_repo = GroupHistoryRepository()
_global_repo = GlobalHistoryRepository()

@router.get("/")
async def health_check():
    return {"status": "psi-09 core interface active", "version": "2.0.0"}

def _clean_discord_mentions(text: str) -> str:
    """Replace Discord snowflake mention patterns with @PSI-09 for bot IDs."""
    for d_id in [settings.DISCORD_ID, settings.DISCORD_ID_2]:
        if d_id:
            text = re.sub(r'<@!?' + re.escape(str(d_id)) + r'>', '@PSI-09', text)
    return text

@router.post("/psi09", response_model=EngineResponse)
async def process_message(payload: IncomingPayload, background_tasks: BackgroundTasks):
    """
    Unified entrypoint for all platform bridges.
    Legacy order: Generate roast FIRST → Store user message → Store bot reply → Background evolution
    """
    try:
        # Sanitize Discord snowflake mentions before anything else
        payload.message = _clean_discord_mentions(payload.message)
        
        user_key = f"{payload.group_name}:{payload.username}"
        global_key = f"Global:{payload.username}"
        is_private = payload.group_name == "private_chat"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # --- 1. DISPATCH TO ENGINE FIRST (before storage, so history doesn't contain the current message) ---
        response = await dispatcher.dispatch(payload)
        
        # --- 2. STORE USER MESSAGE (after engine runs, matching legacy order) ---
        local_entry = {
            "role": "user",
            "user_id": payload.sender_id,
            "username": payload.username,
            "display_name": payload.display_name,
            "platform": payload.platform,
            "channel": payload.channel,
            "content": payload.message,
            "timestamp": timestamp
        }
        
        # Global entry enriched with platform/group provenance
        global_entry = local_entry.copy()
        global_entry["content"] = f"[Sent via {payload.platform} - {payload.group_name} #{payload.channel}] {payload.message}"
        
        await asyncio.to_thread(_chat_repo.store_message, user_key, local_entry)
        await asyncio.to_thread(_global_repo.store_message, global_key, global_entry)
        
        if not is_private:
            await asyncio.to_thread(_group_repo.store_message, payload.group_name, local_entry)

        # --- 3. STORE ASSISTANT REPLY (chronologically after user message) ---
        if response.reply:
            reply_entry = {
                "role": "assistant",
                "username": "PSI-09",
                "content": response.reply,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await asyncio.to_thread(_chat_repo.store_message, user_key, reply_entry)
            
            if not is_private:
                await asyncio.to_thread(_group_repo.store_message, payload.group_name, reply_entry)
        
        # --- 4. SCHEDULE BACKGROUND EVOLUTION ---
        background_tasks.add_task(evolve_profile_task, user_key, payload.group_name, global_key, payload.mode)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
