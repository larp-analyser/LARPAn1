# app/tasks/background.py

import logging
import asyncio
import dspy
from datetime import datetime, timezone
from app.core.config import settings
from app.core.llm_balancer import background_pool
from app.core.utils import sanitize_think_tags

# --- ADD THIS IMPORT ---
from app.db.vector_store import vector_db

from app.db.repositories import (
    ChatRepository, 
    GroupHistoryRepository, 
    MemoryRepository, 
    GroupMemoryRepository, 
    GraphRepository, 
    GlobalHistoryRepository, 
    GlobalMemoryRepository, 
    CounterRepository
)
# ... (Keep existing prompt imports) ...

# --- ADD THE NEW BACKFILL TASK BELOW THE IMPORTS ---
async def vector_backfill_task():
    """Scans all chat histories and bulk-embeds missing messages into the Vector database."""
    logger.info("[VECTOR_BACKFILL] Initiating semantic memory backfill sweep...")
    chat_repo = ChatRepository()
    
    try:
        # Fetch all user chat histories
        all_chats = await asyncio.to_thread(lambda: list(chat_repo.collection.find({})))
        messages_to_embed = []
        
        for doc in all_chats:
            for msg in doc.get("messages", []):
                # Only embed actual user messages (skip AN1 replies and system messages)
                if msg.get("role") == "user" and msg.get("content") and msg.get("username"):
                    messages_to_embed.append({
                        "username": msg["username"],
                        "content": msg["content"],
                        "timestamp": msg.get("timestamp", datetime.now(timezone.utc).isoformat())
                    })
                    
        if messages_to_embed:
            embedded_count = await asyncio.to_thread(vector_db.add_batch_messages, messages_to_embed)
            logger.info(f"[VECTOR_BACKFILL] Complete. Embedded {embedded_count} new historical receipts.")
        else:
            logger.info("[VECTOR_BACKFILL] No messages found to backfill.")
    except Exception as e:
        logger.error(f"[VECTOR_BACKFILL] Error during backfill: {e}")

# ... (Keep existing _evolve_graph and _evolve_text_profile unchanged) ...

# --- UPDATE EVOLVE PROFILE TASK FOR LIVE INGESTION ---
async def evolve_profile_task(user_key: str, group_name: str, global_key: str, mode: str):
    chat_repo = ChatRepository()
    
    # 1. LIVE VECTOR INGESTION: Grab the message the user literally just sent and embed it
    try:
        latest_msg = await asyncio.to_thread(chat_repo.get_recent_history, user_key, limit=1)
        if latest_msg and latest_msg[0].get("role") == "user":
            msg_data = latest_msg[0]
            await asyncio.to_thread(
                vector_db.add_message, 
                msg_data.get("username", "Unknown"), 
                msg_data.get("content", ""), 
                msg_data.get("timestamp", datetime.now(timezone.utc).isoformat())
            )
    except Exception as e:
        logger.error(f"[BACKGROUND] Failed to ingest live vector message: {e}")

    # ... (THE REST OF THE FUNCTION REMAINS EXACTLY THE SAME FROM HERE) ...
    group_repo = GroupHistoryRepository()
    global_history_repo = GlobalHistoryRepository()
    counter_repo = CounterRepository()
    
    if mode in ["vrag", "auto"]:
    # ... (Keep the rest of evolve_profile_task untouched) ...

# ... (Keep hourly_sweep_task untouched) ...
