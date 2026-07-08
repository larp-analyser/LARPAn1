from app.api.models import IncomingPayload, EngineResponse
from app.engine.base import BaseEngine
from app.engine.roastbot import RoastbotEngine
from app.engine.vrag import VRAGEngine

class EngineDispatcher:
    """
    Routes the incoming payload to the appropriate engine implementation based on
    the payload 'mode' configuration.
    """
    def __init__(self):
        self.roastbot = RoastbotEngine()
        self.vrag = VRAGEngine()
        
    async def dispatch(self, payload: IncomingPayload) -> EngineResponse:
        mode = payload.mode
        
        if mode == "legacy":
            engine = self.roastbot
        elif mode == "vrag":
            engine = self.vrag
        else:
            # "auto" defaults to vRAG as the new frontier standard
            engine = self.vrag
            
        try:
            return await engine.process(payload)
        except Exception as e:
            return EngineResponse(
                reply=None,
                reaction=None,
                engine_used="error",
                error=str(e)
            )

# Singleton instance to be used by the API routes
dispatcher = EngineDispatcher()
