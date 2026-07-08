import logging
from app.engine.base import BaseEngine
from app.api.models import IncomingPayload, EngineResponse
from app.core.llm_balancer import nvidia_combat_pool
from app.db.repositories import ChatRepository, GroupHistoryRepository, MemoryRepository, GlobalMemoryRepository
from app.prompts.roastbot_prompts import ROAST_PROMPT, GROUP_ROAST_PROMPT

logger = logging.getLogger(__name__)

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
        user_history_docs = self.chat_repo.get_recent_history(user_key, limit=30)
        local_group_profile = self.memory_repo.get_profile(user_key)
        global_omniscient_profile = self.global_repo.get_profile(f"Global:{payload.username}")
        
        prompt_text = ""
        if is_private:
            prompt_text = ROAST_PROMPT
            prompt_text += f"\n\n<chat_history>\n{self._format_history(user_history_docs)}\n</chat_history>"
            prompt_text += f"\n\n<local_group_profile>\n{local_group_profile}\n</local_group_profile>"
            prompt_text += f"\n\n<global_omniscient_profile>\n{global_omniscient_profile}\n</global_omniscient_profile>"
        else:
            group_history_docs = self.group_repo.get_recent_history(payload.group_name, limit=80)
            group_summary = self.memory_repo.get_profile(payload.group_name) # Group profiles stored in memory_repo for this legacy
            prompt_text = GROUP_ROAST_PROMPT
            prompt_text += f"\n\n<chat_history>\n{self._format_history(group_history_docs)}\n</chat_history>"
            prompt_text += f"\n\n<group_dynamic_summary>\n{group_summary}\n</group_dynamic_summary>"
            prompt_text += f"\n\n<local_group_profile>\n{local_group_profile}\n</local_group_profile>"
            prompt_text += f"\n\n<global_omniscient_profile>\n{global_omniscient_profile}\n</global_omniscient_profile>"
            
        # Add target
        prompt_text += f"\n\nTARGET USER: {payload.username}\nMESSAGE: {payload.message}"
        
        # 3. Execution via LLM Pool
        max_retries = len(nvidia_combat_pool.models) if nvidia_combat_pool.models else 1
        final_reply = ""
        
        for attempt in range(max_retries):
            try:
                current_lm = nvidia_combat_pool.get_next()
                import asyncio
                # Run the synchronous DSPy LLM call in a background thread
                res = await asyncio.to_thread(current_lm, prompt_text) 
                
                if isinstance(res, list) and len(res) > 0:
                    final_reply = res[0]
                else:
                    final_reply = str(res)
                    
                # Clean up <think> tags if model uses chain of thought natively
                if "<think>" in final_reply:
                    final_reply = final_reply.split("</think>")[-1].strip()
                    
                break
            except Exception as e:
                logger.error(f"[{self.engine_name()}] Generation failed on attempt {attempt+1}: {e}")
        
        return EngineResponse(
            reply=final_reply if final_reply else None,
            reaction=None, # Legacy engine does not support reactions
            engine_used=self.engine_name()
        )
