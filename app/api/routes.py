from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.api.models import IncomingPayload, EngineResponse
from app.engine.dispatcher import dispatcher
from app.tasks.background import evolve_profile_task
from app.db.repositories import ChatRepository, GroupHistoryRepository, GlobalHistoryRepository
from datetime import datetime, timezone

router = APIRouter()

@router.get("/")
async def health_check():
    return {"status": "psi-09 core interface active", "version": "2.0.0"}

@router.post("/psi09", response_model=EngineResponse)
async def process_message(payload: IncomingPayload, background_tasks: BackgroundTasks):
    """
    Unified entrypoint for all platform bridges.
    """
    try:
        user_key = f"{payload.group_name}:{payload.username}"
        global_key = f"Global:{payload.username}"
        
        # Save the message to the DB BEFORE dispatching
        message_data = {
            "username": payload.username,
            "content": payload.message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        ChatRepository().store_message(user_key, message_data)
        GlobalHistoryRepository().store_message(global_key, message_data)
        
        if payload.group_name != "private_chat":
            GroupHistoryRepository().store_message(payload.group_name, message_data)

        # The dispatcher handles selecting the correct engine based on payload.mode
        response = await dispatcher.dispatch(payload)
        
        # Save Assistant Reply Persistence
        if response.reply:
            reply_data = {
                "username": "PSI-09",
                "content": response.reply,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            ChatRepository().store_message(user_key, reply_data)
            GlobalHistoryRepository().store_message(global_key, reply_data)
            
            if payload.group_name != "private_chat":
                GroupHistoryRepository().store_message(payload.group_name, reply_data)
        
        # Schedule the background evolution safely without blocking the response
        background_tasks.add_task(evolve_profile_task, user_key, payload.group_name, global_key, payload.mode)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
