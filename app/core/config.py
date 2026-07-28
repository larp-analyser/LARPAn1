from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional

class Settings(BaseSettings):
    # MongoDB
    MONGO_URI: str = Field(..., description="MongoDB Connection URI")

    # API Keys (Comma-separated strings from .env)
    NVIDIA_API_KEYS: str = ""
    GROQ_API_KEYS: str = ""
    GOOGLE_API_KEYS: str = ""
    OPENAI_API_KEYS: str = ""
    HF_TOKEN: Optional[str] = None

    # Bot Identifiers
    BOT_NUMBER: Optional[str] = None
    DISCORD_ID: Optional[str] = None
    DISCORD_ID_2: Optional[str] = None

    # Application Tuning
    CRON_SECRET: str = "default_dev_secret"
    MEMORY_TTL: int = 86400  # 24 Hours (utilizing 104GB RAM)
    GROUP_HISTORY_MAX_MESSAGES: int = 50000
    GROUP_HISTORY_SLICE: int = 80  # Feed up to 200 messages at once
    MAX_HISTORY_MESSAGES: int = 80 # User-specific history tracking
    MAX_HISTORY_TOKENS: int = 2000  # Massive personal context
    GROUP_HISTORY_TOKEN_LIMIT: int = 3800  # Massive group context
    EVOLVE_EVERY_N_MESSAGES: int = 50
    GROUP_SUMMARY_EVERY_N: int = 300

    # Provider Pools
    NVIDIA_POOL: List[str] = [
        "z-ai/glm-5.2"
    ]
    
    GROQ_POOL: List[str] = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant"
    ]
    
    GOOGLE_POOL: List[str] = [
        "gemini-3.1-flash-lite"
    ]
    
    OPENAI_POOL: List[str] = [
        "gpt-5.4-mini"
    ]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def nvidia_keys_list(self) -> List[str]:
        return [k.strip() for k in self.NVIDIA_API_KEYS.split(",") if k.strip()]

    @property
    def groq_keys_list(self) -> List[str]:
        return [k.strip() for k in self.GROQ_API_KEYS.split(",") if k.strip()]

    @property
    def google_keys_list(self) -> List[str]:
        return [k.strip() for k in self.GOOGLE_API_KEYS.split(",") if k.strip()]

    @property
    def openai_keys_list(self) -> List[str]:
        return [k.strip() for k in self.OPENAI_API_KEYS.split(",") if k.strip()]

settings = Settings()
