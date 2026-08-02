import threading
import logging
import dspy
import time
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModularRoundRobinPool:
    """
    Round-robin pool with isolated Primary and NVIDIA queues.
    NVIDIA queue is dormant for general pools by default (use_nvidia_fallback = False).
    The use_nvidia_fallback flag is completely programmer-driven.
    When flag = True, it routes exclusively to the NVIDIA queue in round-robin.
    """
    def __init__(self, pool_name: str):
        self.pool_name = pool_name
        self.primary_lm_pool = []
        self.nvidia_lm_pool = []
        
        # Controlled flag: False by default (Strictly programmer-driven)
        self.use_nvidia_fallback = False
        
        self.primary_index = 0
        self.nvidia_index = 0
        self.lock = threading.Lock()

    def enable_nvidia_fallback(self, enable: bool = True):
        """Programmatically toggle the NVIDIA fallback flag on or off."""
        with self.lock:
            self.use_nvidia_fallback = enable
            logger.info(f"[{self.pool_name}] NVIDIA fallback flag manually set to: {self.use_nvidia_fallback}")

    def reset_pool(self):
        """Resets the flag back to False and restarts queue indices."""
        with self.lock:
            self.use_nvidia_fallback = False
            self.primary_index = 0
            self.nvidia_index = 0
            logger.info(f"[{self.pool_name}] Reset to default state (use_nvidia_fallback = False).")

    def add_provider(self, api_keys: list, model_pool: list, provider_prefix: str, is_nvidia: bool = False, **kwargs):
        """
        Registers provider models into either the Primary or NVIDIA pool queue.
        Iterates through all models for Key 1 before moving to Key 2.
        """
        api_keys = [k.strip() for k in (api_keys or []) if k and k.strip()]
        model_pool = [m.strip() for m in (model_pool or []) if m and m.strip()]

        if not api_keys or not model_pool:
            logger.info(f"[{self.pool_name}] Skipping provider prefix '{provider_prefix}' — Unconfigured.")
            return

        target_queue = self.nvidia_lm_pool if is_nvidia else self.primary_lm_pool

        for key in api_keys:
            for model_name in model_pool:
                full_model_name = model_name if model_name.startswith(provider_prefix) else f"{provider_prefix}{model_name}"
                try:
                    # THE FIX: Force the underlying client to NEVER retry on its own
                    lm = dspy.LM(
                        model=full_model_name,
                        api_key=key,
                        timeout=10.0,
                        max_retries=0,      
                        **kwargs
                    )
                    target_queue.append(lm)
                except Exception as e:
                    logger.error(f"[{self.pool_name}] Failed to initialize {full_model_name}: {e}")

    def get_next(self):
        """
        Gets the next model instance.
        If use_nvidia_fallback is True, strictly pulls from nvidia_lm_pool.
        Otherwise, pulls strictly from primary_lm_pool.
        Does NOT automatically flip flags.
        """
        with self.lock:
            if self.use_nvidia_fallback:
                if not self.nvidia_lm_pool:
                    raise RuntimeError(f"[{self.pool_name}] NVIDIA fallback active, but no NVIDIA instances are loaded!")

                current_lm = self.nvidia_lm_pool[self.nvidia_index]
                self.nvidia_index = (self.nvidia_index + 1) % len(self.nvidia_lm_pool)
                return current_lm

            if not self.primary_lm_pool:
                raise RuntimeError(f"[{self.pool_name}] No LM instances available in primary queue!")

            current_lm = self.primary_lm_pool[self.primary_index]
            self.primary_index = (self.primary_index + 1) % len(self.primary_lm_pool)
            return current_lm

    def execute_with_retry(self, dspy_program, *args, max_retries=None, **kwargs):
        with self.lock:
            active_pool_len = len(self.nvidia_lm_pool) if self.use_nvidia_fallback else len(self.primary_lm_pool)

        # Let it cycle through the ENTIRE pool before giving up
        max_attempts = max_retries or active_pool_len
        attempts = 0

        while attempts < max_attempts:
            try:
                lm = self.get_next()
                with dspy.context(lm=lm):
                    return dspy_program(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower().replace("_", " ").replace("-", " ")
                
                retry_triggers = [
                    "429", "rate limit", "ratelimit", "quota", 
                    "request too large", "empty or null", "jsonadapter", 
                    "failed to parse", "none", "500", "502", "503",
                    "json validate failed", "invalid request error"
                ]
                
                if any(trigger in error_str for trigger in retry_triggers):
                    attempts += 1
                    logger.warning(f"[{self.pool_name}] Rate limit or transient error on active model! Advancing instance ({attempts}/{max_attempts}).")
                    
                    # NEW SLEEP LOGIC: Micro-sleep. 
                    # Since we are swapping to a BRAND NEW key, we don't need to wait.
                    # A flat 0.5 seconds prevents CPU thrashing but cycles 18 keys in just 9 seconds.
                    time.sleep(0.5)
                else:
                    raise e

        raise RuntimeError(f"[{self.pool_name}] Exceeded max retries ({max_attempts}) across active pool.")

# 1. COMBAT POOL (Roasting Engine)
# Loaded directly into Primary queue -> Always uses NVIDIA models strictly.
nvidia_combat_pool = ModularRoundRobinPool(pool_name="COMBAT_NVIDIA")
nvidia_combat_pool.add_provider(
    api_keys=settings.nvidia_keys_list,
    model_pool=settings.NVIDIA_POOL,
    provider_prefix="openai/",
    is_nvidia=False, # Loaded directly into primary queue for roasting
    api_base="https://integrate.api.nvidia.com/v1",
    temperature=1.0,
    top_p=1.0,
    max_tokens=16384
)


# 2. TRIAGE POOL
triage_pool = ModularRoundRobinPool(pool_name="TRIAGE_POOL")

# Primary Queue (Google, Groq, OpenAI)
triage_pool.add_provider(
    api_keys=settings.google_keys_list,
    model_pool=settings.GOOGLE_POOL,
    provider_prefix="gemini/",
    temperature=0.5,
    max_tokens=2048
)
triage_pool.add_provider(
    api_keys=settings.groq_keys_list,
    model_pool=settings.GROQ_POOL,
    provider_prefix="groq/",
    temperature=0.5,
    max_tokens=2048
)
triage_pool.add_provider(
    api_keys=settings.openai_keys_list,
    model_pool=settings.OPENAI_POOL,
    provider_prefix="openai/",
    temperature=0.5,
    max_tokens=2048
)

# Dormant NVIDIA Queue (Activated strictly when programmer sets use_nvidia_fallback = True)
triage_pool.add_provider(
    api_keys=settings.nvidia_keys_list,
    model_pool=settings.NVIDIA_POOL,
    provider_prefix="openai/",
    is_nvidia=True,
    api_base="https://integrate.api.nvidia.com/v1",
    temperature=0.5,
    max_tokens=2048
)


# 3. BACKGROUND POOL
background_pool = ModularRoundRobinPool(pool_name="BACKGROUND_POOL")

# Primary Queue (Google, Groq, OpenAI)
background_pool.add_provider(
    api_keys=settings.google_keys_list,
    model_pool=settings.GOOGLE_POOL,
    provider_prefix="gemini/",
    temperature=0.5,
    max_tokens=2048
)
background_pool.add_provider(
    api_keys=settings.groq_keys_list,
    model_pool=settings.GROQ_POOL,
    provider_prefix="groq/",
    temperature=0.5,
    max_tokens=2048
)
background_pool.add_provider(
    api_keys=settings.openai_keys_list,
    model_pool=settings.OPENAI_POOL,
    provider_prefix="openai/",
    temperature=0.5,
    max_tokens=2048
)

# Dormant NVIDIA Queue (Activated strictly when programmer sets use_nvidia_fallback = True)
background_pool.add_provider(
    api_keys=settings.nvidia_keys_list,
    model_pool=settings.NVIDIA_POOL,
    provider_prefix="openai/",
    is_nvidia=True,
    api_base="https://integrate.api.nvidia.com/v1",
    temperature=0.5,
    max_tokens=2048
)
