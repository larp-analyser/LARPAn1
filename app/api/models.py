from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class TaggedUser(BaseModel):
    id: str
    username: str
    display_name: str

class IncomingPayload(BaseModel):
    message: str = Field(..., description="The user's message text")
    sender_id: str = Field(..., description="Platform-specific user ID")
    username: str = Field(..., description="Username of the sender")
    display_name: str = Field(..., description="Display name of the sender")
    group_name: str = Field(..., description="Server/group name, or 'discord_dm'")
    channel: str = Field(..., description="Channel name")
    tagged_users: List[TaggedUser] = Field(default_factory=list, description="Array of tagged users")
    platform: Literal["discord", "whatsapp", "minecraft"] = Field(..., description="Origin platform")
    force_reply: bool = Field(default=False, description="Bypass mention checks and force engagement")
    mode: Literal["auto", "legacy", "vrag"] = Field(
        default="auto", 
        description="Select the underlying engine architecture. 'auto' defaults to vRAG with legacy fallback."
    )

class EngineResponse(BaseModel):
    reply: Optional[str] = Field(None, description="Text reply from the engine")
    reaction: Optional[str] = Field(None, description="Emoji reaction from the engine")
    engine_used: str = Field(..., description="Which engine generated the response (e.g., 'vrag', 'roastbot', 'triage_silence')")
    error: Optional[str] = Field(None, description="Error message if generation failed")
