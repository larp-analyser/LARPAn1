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
    """
    try:
        # Sanitize Discord snowflake mentions before anything else
        payload.message = _clean_discord_mentions(payload.message)
        
        user_key = f"{payload.group_name}:{payload.username}"
        global_key = f"Global:{payload.username}"
        is_private = payload.group_name == "private_chat"
        
        # --- 1. Store User Message ---
        timestamp = datetime.now(timezone.utc).isoformat()
        
        local_entry = {
            "role": "user",
            "username": payload.username,
            "content": payload.message,
            "timestamp": timestamp
        }
        
        # Global entry enriched with platform/group provenance
        global_entry = {
            "role": "user",
            "username": payload.username,
            "content": f"[Sent via {payload.platform} - {payload.group_name} #{payload.channel}] {payload.message}",
            "timestamp": timestamp
        }
        
        await asyncio.to_thread(_chat_repo.store_message, user_key, local_entry)
        await asyncio.to_thread(_global_repo.store_message, global_key, global_entry)
        
        if not is_private:
            await asyncio.to_thread(_group_repo.store_message, payload.group_name, local_entry)

        # --- 2. Dispatch to Engine ---
        response = await dispatcher.dispatch(payload)
        
        # --- 3. Store Assistant Reply ---
        if response.reply:
            reply_timestamp = datetime.now(timezone.utc).isoformat()
            
            reply_entry = {
                "role": "assistant",
                "username": "PSI-09",
                "content": response.reply,
                "timestamp": reply_timestamp
            }
            
            await asyncio.to_thread(_chat_repo.store_message, user_key, reply_entry)
            
            if not is_private:
                await asyncio.to_thread(_group_repo.store_message, payload.group_name, reply_entry)
        
        # --- 4. Schedule Background Evolution ---
        background_tasks.add_task(evolve_profile_task, user_key, payload.group_name, global_key, payload.mode)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
