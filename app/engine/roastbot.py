import re
import logging
import asyncio
from app.engine.base import BaseEngine
from app.api.models import IncomingPayload, EngineResponse
from app.core.llm_balancer import nvidia_combat_pool
from app.db.repositories import ChatRepository, GroupHistoryRepository, MemoryRepository, GlobalMemoryRepository
from app.prompts.roastbot_prompts import ROAST_PROMPT, GROUP_ROAST_PROMPT
from app.core.config import settings

logger = logging.getLogger(__name__)

def _fetch_tagged_profiles(tagged_users: list, global_repo: GlobalMemoryRepository, max_targets: int = 3) -> list:
    """Fetch global memory profiles for users tagged in the message."""
    profiles = []
    for u in tagged_users[:max_targets]:
        uid = u.id
        username = u.username
        if not username:
            continue
        memory_key = f"Global:{username}"
        summary = global_repo.get_profile(memory_key)
        if summary:
            profiles.append(f'<bystander username="{username}" numeric_id="{uid}">\n{summary.strip()}\n</bystander>')
    return profiles

def _sanitize_think_tags(text: str) -> str:
    """Comprehensive think-tag sanitization matching the old 5-regex chain."""
    if not text:
        return ""
    clean = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<\|?think\|?>.*?</\|?think\|?>\s*", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<\|?think\|?>.*", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<\|channel>thought.*?</channel\|>\s*", "", clean, flags=re.DOTALL | re.IGNORECASE)
    return clean.strip()

def _trim_history_by_tokens(history_docs: list, max_tokens: int) -> list:
    """Trim history to fit within a token budget using word-count estimation."""
    total = 0
    trimmed = []
    for m in reversed(history_docs):
        sender = m.get("username") or m.get("sender") or m.get("role") or "User"
        formatted = f"[{sender}]: {m.get('content', '')}"
        estimated_tokens = int(len(formatted.split()) * 1.5)
        if total + estimated_tokens > max_tokens:
            break
        trimmed.insert(0, m)
        total += estimated_tokens
    return trimmed


class RoastbotEngine(BaseEngine):
    """
    The Legacy Production Engine.
    Uses flat prompt injection and MongoDB text summaries.
    """
    def __init__(self):
        self.chat_repo = ChatRepository()
        self.group_repo = GroupHistoryRepository()
        self.memory_repo = MemoryRepository()
        self.global_repo = GlobalMemoryRepository()
    
    def engine_name(self) -> str:
        return "roastbot"
        
    def _format_history(self, history_docs: list) -> str:
        return "\n".join([f"[{m.get('username', 'Unknown')}]: {m.get('content', '')}" for m in history_docs])

    async def process(self, payload: IncomingPayload) -> EngineResponse:
        user_key = f"{payload.group_name}:{payload.username}"
        is_private = (payload.group_name == "private_chat")
        
        # 1. Triage / Engagement Check (Legacy Hardcoded Logic)
        will_reply = is_private or payload.force_reply or ("@PSI-09" in payload.message)
        if not will_reply:
            return EngineResponse(
                reply=None,
                reaction=None,
                engine_used="triage_silence"
            )
            
        # 2. Context Assembly
        user_history_docs = self.chat_repo.get_recent_history(user_key, limit=settings.MAX_HISTORY_MESSAGES)
        user_history_docs = _trim_history_by_tokens(user_history_docs, settings.MAX_HISTORY_TOKENS)
        
        local_group_profile = self.memory_repo.get_profile(user_key)
        global_omniscient_profile = self.global_repo.get_profile(f"Global:{payload.username}")
        
        prompt_text = ""
        if is_private:
            prompt_text = ROAST_PROMPT
            prompt_text += f"\n\n<chat_history>\n{self._format_history(user_history_docs)}\n</chat_history>"
            prompt_text += f"\n\n<local_group_profile>\n{local_group_profile}\n</local_group_profile>"
            prompt_text += f"\n\n<global_omniscient_profile>\n{global_omniscient_profile}\n</global_omniscient_profile>"
        else:
            group_history_docs = self.group_repo.get_recent_history(payload.group_name, limit=settings.GROUP_HISTORY_SLICE)
            group_history_docs = _trim_history_by_tokens(group_history_docs, settings.GROUP_HISTORY_TOKEN_LIMIT)
            
            group_summary = self.memory_repo.get_profile(payload.group_name)
            prompt_text = GROUP_ROAST_PROMPT
            prompt_text += f"\n\n<chat_history>\n{self._format_history(group_history_docs)}\n</chat_history>"
            prompt_text += f"\n\n<group_dynamic_summary>\n{group_summary}\n</group_dynamic_summary>"
            prompt_text += f"\n\n<local_group_profile>\n{local_group_profile}\n</local_group_profile>"
            prompt_text += f"\n\n<global_omniscient_profile>\n{global_omniscient_profile}\n</global_omniscient_profile>"
            
            # Tagged User Profiles (Gap 1 fix)
            tagged_profiles = _fetch_tagged_profiles(payload.tagged_users, self.global_repo)
            if tagged_profiles:
                joined_profiles = "\n\n".join(tagged_profiles)
                prompt_text += f"\n\n<tagged_member_profiles>\n{joined_profiles}\n</tagged_member_profiles>"
            
        # Add target
        prompt_text += f"\n\nTARGET USER: {payload.username}\nMESSAGE: {payload.message}"
        
        # 3. Execution via LLM Pool
        max_retries = len(nvidia_combat_pool.models) if nvidia_combat_pool.models else 1
        final_reply = ""
        
        for attempt in range(max_retries):
            try:
                current_lm = nvidia_combat_pool.get_next()
                # Run the synchronous DSPy LLM call in a background thread
                res = await asyncio.to_thread(current_lm, prompt_text) 
                
                if isinstance(res, list) and len(res) > 0:
                    final_reply = res[0]
                else:
                    final_reply = str(res)
                    
                # Comprehensive think-tag sanitization (Gap 5 fix)
                final_reply = _sanitize_think_tags(final_reply)
                    
                break
            except Exception as e:
                logger.error(f"[{self.engine_name()}] Generation failed on attempt {attempt+1}: {e}")
        
        return EngineResponse(
            reply=final_reply if final_reply else None,
            reaction=None, # Legacy engine does not support reactions
            engine_used=self.engine_name()
        )
