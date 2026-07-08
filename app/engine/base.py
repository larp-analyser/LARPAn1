from abc import ABC, abstractmethod
from app.api.models import IncomingPayload, EngineResponse
from typing import Optional

class BaseEngine(ABC):
    """
    Abstract base class for all PSI-09 conversation engines.
    Ensures that every engine implements a standard interface for the bridges.
    """
    
    @abstractmethod
    async def process(self, payload: IncomingPayload) -> EngineResponse:
        """
        Process the incoming payload and return a standardized EngineResponse.
        This method must handle its own DB persistence and background task triggering.
        """
        pass
    
    @abstractmethod
    def engine_name(self) -> str:
        """Return the identifier for this engine"""
        pass
