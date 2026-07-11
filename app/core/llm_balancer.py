import threading
import logging
import dspy

logger = logging.getLogger(__name__)

class FailoverLMPool:
    """
    Manages a pool of models for fallback in case of rate limits (429).
    Useful for Groq where limits are strictly enforced on free/lower tiers.
    """
    def __init__(self, model_names: list, api_keys: list, pool_name: str):
        self.pool_name = pool_name
        self.models = []
        for m in model_names:
            for key in api_keys:
                if key:
                    self.models.append(dspy.LM(model=f"groq/{m}", api_key=key.strip(), max_tokens=2048))
        self.index = 0
        self.lock = threading.Lock()

    def get_current(self):
        with self.lock:
            if not self.models:
                raise ValueError(f"[{self.pool_name}] No API key or models configured.")
            return self.models[self.index], self.index

    def advance(self, failed_index: int):
        with self.lock:
            if self.index == failed_index:
                self.index = (self.index + 1) % len(self.models)
                logger.warning(f"[{self.pool_name}] Rate Limit! Failover triggered -> {self.models[self.index].model}")
            return self.models[self.index]


class NvidiaRoundRobinPool:
    """
    Distributes requests across multiple API keys sequentially.
    Useful for NVIDIA NIM to maximize total throughput.
    """
    def __init__(self, api_keys: list, model_name: str, pool_name: str):
        self.pool_name = pool_name
        self.models = []
        for key in api_keys:
            if key:
                self.models.append(dspy.LM(
                    model=f"openai/{model_name}",
                    api_base="https://integrate.api.nvidia.com/v1",
                    api_key=key,
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=2048
                ))
        self.index = 0
        self.lock = threading.Lock()

    def get_next(self):
        with self.lock:
            if not self.models:
                raise ValueError(f"[{self.pool_name}] No NVIDIA API keys configured.")
            current_model = self.models[self.index]
            self.index = (self.index + 1) % len(self.models)
            return current_model

from app.core.config import settings

triage_pool = FailoverLMPool(
    model_names=settings.TRIAGE_MODELS, 
    api_keys=settings.groq_keys_list, 
    pool_name="TRIAGE"
)

background_pool = FailoverLMPool(
    model_names=settings.BACKGROUND_MODELS, 
    api_keys=settings.groq_keys_list, 
    pool_name="BACKGROUND"
)

nvidia_combat_pool = NvidiaRoundRobinPool(
    api_keys=settings.nvidia_keys_list,
    model_name=settings.ROAST_MODELS[0],
    pool_name="COMBAT"
)
