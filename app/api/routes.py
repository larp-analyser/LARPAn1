from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.api.models import IncomingPayload, EngineResponse
from app.engine.dispatcher import dispatcher
from app.tasks.background import evolve_profile_task

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
        # The dispatcher handles selecting the correct engine based on payload.mode
        response = await dispatcher.dispatch(payload)
        
        # Schedule the background evolution safely without blocking the response
        user_key = f"{payload.group_name}:{payload.username}"
        background_tasks.add_task(evolve_profile_task, user_key, payload.mode)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
