from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional

class Settings(BaseSettings):
    # MongoDB
    MONGO_URI: str = Field(..., description="MongoDB Connection URI")

    # API Keys (Comma-separated for Round-Robin)
    GROQ_API_KEYS: str = ""
    NVIDIA_API_KEYS: str = ""
    HF_TOKEN: Optional[str] = None

    # Bot Identifiers
    BOT_NUMBER: Optional[str] = None
    DISCORD_ID: Optional[str] = None
    DISCORD_ID_2: Optional[str] = None

    # Application Tuning
    MEMORY_TTL: int = 500
    GROUP_HISTORY_MAX_MESSAGES: int = 50000
    GROUP_HISTORY_SLICE: int = 80
    MAX_HISTORY_MESSAGES: int = 30
    MAX_HISTORY_TOKENS: int = 400
    GROUP_HISTORY_TOKEN_LIMIT: int = 2000
    EVOLVE_EVERY_N_MESSAGES: int = 50
    GROUP_SUMMARY_EVERY_N: int = 300

    # Model Pools
    ROAST_MODELS: List[str] = ["meta/llama-3.1-70b-instruct"]
    BACKGROUND_MODELS: List[str] = [
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768"
    ]
    TRIAGE_MODELS: List[str] = [
        "llama-3.3-70b-versatile",
        "llama3-8b-8192"
    ]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def groq_keys_list(self) -> List[str]:
        return [k.strip() for k in self.GROQ_API_KEYS.split(",") if k.strip()]

    @property
    def nvidia_keys_list(self) -> List[str]:
        return [k.strip() for k in self.NVIDIA_API_KEYS.split(",") if k.strip()]

settings = Settings()
