from fastapi import APIRouter, BackgroundTasks, HTTPException, Header
from app.api.models import IncomingPayload, EngineResponse
from app.engine.dispatcher import dispatcher
from app.tasks.background import evolve_profile_task, hourly_sweep_task
from app.teleprompter.optimizer import run_teleprompter_task
from app.db.repositories import ChatRepository, GroupHistoryRepository, GlobalHistoryRepository
from app.core.config import settings
from datetime import datetime, timezone
import asyncio
import re

from app.core.utils import normalize_bot_mentions

router = APIRouter()

# Singleton repository instances to avoid re-instantiation per request
_chat_repo = ChatRepository()
_group_repo = GroupHistoryRepository()
_global_repo = GlobalHistoryRepository()

@router.get("/")
async def health_check():
    return {"status": "an1 core interface active", "version": "2.0.0"}

@router.post("/cron/sweep")
async def trigger_hourly_sweep(background_tasks: BackgroundTasks, x_cron_secret: str = Header(None)):
    if not x_cron_secret or x_cron_secret != settings.CRON_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized cron trigger")
        
    background_tasks.add_task(hourly_sweep_task)
    return {"status": "sweep_scheduled"}

async def safe_teleprompter_task():
    await asyncio.to_thread(run_teleprompter_task)

@router.post("/cron/optimize")
async def trigger_optimization(background_tasks: BackgroundTasks, x_cron_secret: str = Header(None)):
    if not x_cron_secret or x_cron_secret != settings.CRON_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized cron trigger")
        
    background_tasks.add_task(safe_teleprompter_task)
    return {"status": "optimization_scheduled"}

@router.post("/an1", response_model=EngineResponse)
async def process_message(payload: IncomingPayload, background_tasks: BackgroundTasks):
    """
    Unified entrypoint for all platform bridges.
    Legacy order: Generate roast FIRST → Store user message → Store bot reply → Background evolution
    """
    try:
        # Sanitize only bot mentions, preserve all other raw IDs
        payload.message = normalize_bot_mentions(payload.message)
        
        if payload.group_name.lower() in ["defaultgroup", "discord_dm"]:
            payload.group_name = "private_chat"
            
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
                "username": "AN1",
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
