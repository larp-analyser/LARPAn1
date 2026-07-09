import re
import logging
import asyncio
from app.engine.base import BaseEngine
from app.api.models import IncomingPayload, EngineResponse
from app.core.llm_balancer import nvidia_combat_pool
from app.db.repositories import ChatRepository, GroupHistoryRepository, MemoryRepository, GlobalMemoryRepository
from app.prompts.roastbot_prompts import ROAST_PROMPT, GROUP_ROAST_PROMPT
from app.core.config import settings
from app.core.utils import sanitize_think_tags, trim_history_by_tokens

def _clean_snowflakes(text: str) -> str:
    """Replace Discord snowflake mention patterns with @PSI-09 for bot IDs, strip others."""
    for d_id in [settings.DISCORD_ID, settings.DISCORD_ID_2]:
        if d_id:
            text = re.sub(r'<@!?' + re.escape(str(d_id)) + r'>', '@PSI-09', text)
    # Strip remaining snowflake mentions the bot doesn't own
    text = re.sub(r'<@!?&?\d+>', '', text)
    return text

logger = logging.getLogger(__name__)

async def _fetch_tagged_profiles(tagged_users: list, global_repo: GlobalMemoryRepository, max_targets: int = 3) -> list:
    """Fetch global memory profiles for users tagged in the message."""
    profiles = []
    for u in tagged_users[:max_targets]:
        uid = u.id
        username = u.username
        if not username:
            continue
        memory_key = f"Global:{username}"
        summary = await asyncio.to_thread(global_repo.get_profile, memory_key)
        if summary:
            profiles.append(f'<bystander username="{username}" numeric_id="{uid}">\n{summary.strip()}\n</bystander>')
    return profiles




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
        lines = []
        for m in history_docs:
            role = m.get("role", "user")
            sender = "PSI-09" if role == "assistant" else m.get('username', 'Unknown')
            content = _clean_snowflakes(m.get('content', ''))
            chan = m.get("channel", "unknown")
            lines.append(f"[#{chan}] [{sender}]: {content}")
        return "\n".join(lines)

    async def process(self, payload: IncomingPayload) -> EngineResponse:
        user_key = f"{payload.group_name}:{payload.username}"
        is_private = (payload.group_name == "private_chat")
        
        # Clean the incoming message of Discord snowflakes
        clean_message = _clean_snowflakes(payload.message)
        
        # 1. Triage / Engagement Check (Legacy Hardcoded Logic)
        will_reply = is_private or payload.force_reply or ("@PSI-09" in clean_message)
        if not will_reply:
            return EngineResponse(
                reply=None,
                reaction=None,
                engine_used="triage_silence"
            )
            
        # 2. Context Assembly — Build structured messages list (matching legacy psi-09-roastbot)
        user_history_docs = await asyncio.to_thread(self.chat_repo.get_recent_history, user_key, limit=settings.MAX_HISTORY_MESSAGES)
        user_history_docs = trim_history_by_tokens(user_history_docs, settings.MAX_HISTORY_TOKENS)
        
        local_group_profile = await asyncio.to_thread(self.memory_repo.get_profile, user_key)
        global_omniscient_profile = await asyncio.to_thread(self.global_repo.get_profile, f"Global:{payload.username}")
        
        llm_feed = []
        
        if is_private:
            llm_feed.append({"role": "system", "content": f"<roast_prompt>\n{ROAST_PROMPT}\n</roast_prompt>"})
            
            if local_group_profile:
                llm_feed.append({"role": "system", "content": f"<local_group_profile>\n{local_group_profile.strip()}\n</local_group_profile>"})
            if global_omniscient_profile:
                llm_feed.append({"role": "system", "content": f"<global_omniscient_profile>\n{global_omniscient_profile.strip()}\n</global_omniscient_profile>"})
            
            history_text = self._format_history(user_history_docs)
            llm_feed.append({
                "role": "user",
                "content": (
                    f"<chat_history>\n{history_text}\n</chat_history>\n\n"
                    f"<active_target>\nTARGET USER: [{payload.username}]\nMESSAGE: {clean_message}\n</active_target>"
                )
            })
        else:
            group_history_docs = await asyncio.to_thread(self.group_repo.get_recent_history, payload.group_name, limit=settings.GROUP_HISTORY_SLICE)
            group_history_docs = trim_history_by_tokens(group_history_docs, settings.GROUP_HISTORY_TOKEN_LIMIT)
            group_summary = await asyncio.to_thread(self.memory_repo.get_profile, payload.group_name)
            
            llm_feed.append({"role": "system", "content": f"<roast_prompt>\n{GROUP_ROAST_PROMPT}\n</roast_prompt>"})
            
            if local_group_profile:
                llm_feed.append({"role": "system", "content": f"<local_group_profile>\n{local_group_profile.strip()}\n</local_group_profile>"})
            if global_omniscient_profile:
                llm_feed.append({"role": "system", "content": f"<global_omniscient_profile>\n{global_omniscient_profile.strip()}\n</global_omniscient_profile>"})
            if group_summary:
                llm_feed.append({"role": "system", "content": f"<group_dynamic_summary>\n{group_summary.strip()}\n</group_dynamic_summary>"})
            
            # Tagged User Profiles (Gap 1 fix)
            tagged_profiles = await _fetch_tagged_profiles(payload.tagged_users, self.global_repo)
            if tagged_profiles:
                joined_profiles = "\n\n".join(tagged_profiles)
                llm_feed.append({"role": "system", "content": f"<tagged_member_profiles>\n{joined_profiles}\n</tagged_member_profiles>"})
            
            history_text = self._format_history(group_history_docs)
            llm_feed.append({
                "role": "user",
                "content": (
                    f"<chat_history>\n{history_text}\n</chat_history>\n\n"
                    f"<active_target>\nTARGET USER: [{payload.username}]\nMESSAGE: {clean_message}\n</active_target>"
                )
            })
        
        # 3. Execution via LLM Pool — Structured chat-completion call
        max_retries = len(nvidia_combat_pool.models) if nvidia_combat_pool.models else 1
        final_reply = ""
        
        for attempt in range(max_retries):
            try:
                current_lm = nvidia_combat_pool.get_next()
                # Run the synchronous DSPy LLM call in a background thread with structured messages
                res = await asyncio.to_thread(current_lm, messages=llm_feed) 
                
                if isinstance(res, list) and len(res) > 0:
                    final_reply = res[0]
                else:
                    final_reply = str(res)
                    
                # Comprehensive think-tag sanitization (Gap 5 fix)
                final_reply = sanitize_think_tags(final_reply)
                    
                break
            except Exception as e:
                logger.error(f"[{self.engine_name()}] Generation failed on attempt {attempt+1}: {e}")
        
        return EngineResponse(
            reply=final_reply if final_reply else None,
            reaction=None, # Legacy engine does not support reactions
            engine_used=self.engine_name()
        )
