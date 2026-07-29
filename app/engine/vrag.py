import dspy
import asyncio
import logging
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.core.config import settings
from app.engine.base import BaseEngine
from app.api.models import IncomingPayload, EngineResponse
from app.prompts.dspy_signatures import (
    IdentitySignature,
    MissionSignature,
    ConstraintsSignature,
    DecisionSignature,
    TriageSignature,
    SelfInsultPreventionSignature
)
from app.core.llm_balancer import triage_pool, nvidia_combat_pool
from app.db.repositories import GraphRepository, ChatRepository, GroupHistoryRepository, GlobalMemoryRepository
from app.engine.graph_analyzer import build_networkx_context
from app.core.utils import sanitize_think_tags

logger = logging.getLogger(__name__)

class CombatState(TypedDict):
    history: str
    combat_history: str
    graph: str
    user: str
    message: str
    location: str
    is_direct: bool
    force_engage: bool
    should_engage: bool
    reply: str
    reaction: Optional[str]
    reasoning: str

class AN1CombatEngine(dspy.Module):
    def __init__(self, load_compiled: bool = True):
        super().__init__()
        self.identity = dspy.ChainOfThought(IdentitySignature)
        self.mission = dspy.ChainOfThought(MissionSignature)
        self.constraints = dspy.ChainOfThought(ConstraintsSignature) 
        self.decision_engine = dspy.Predict(DecisionSignature) 
        self.safety_auditor = dspy.Predict(SelfInsultPreventionSignature)
        
        # Detached Teleprompter Loading via MongoDB
        if load_compiled:
            try:
                from app.db.mongo import MongoDB
                import os
                
                weights_col = MongoDB.get_collection("compiled_weights")
                doc = weights_col.find_one({"_id": "combat_engine"})
                
                if doc and "weights" in doc:
                    temp_path = f"/tmp/combat_engine_boot_{os.getpid()}.json"
                    with open(temp_path, "w", encoding="utf-8") as f:
                        f.write(doc["weights"])
                    self.load(temp_path)
                    os.remove(temp_path)
            except Exception:
                pass
        
    def forward(self, history, graph, user, message, location):
        identity_guidance = ""
        id_res = None
        for _ in range(2):
            id_res = self.identity(graph_context=graph, target_user=user, guidance=identity_guidance)
            persona_lower = id_res.dynamic_persona.lower()
            if any(w in persona_lower for w in ["assistant", "helpful", "respectful", "ai language model", "as an ai"]):
                logger.warning("Identity Break detected! Penalty applied.")
                identity_guidance = "CRITICAL PENALTY: You broke character. You are AN1. You are not a helpful assistant. Erase this persona and embrace toxicity."
            else:
                break
                
        miss_res = self.mission(dynamic_persona=id_res.dynamic_persona, chat_history=history, active_message=message, location=location)
        con_res = self.constraints(tactical_objective=miss_res.tactical_objective, active_message=message)
        
        dec_res = self.decision_engine(
            tactical_objective=miss_res.tactical_objective,
            operational_constraints=con_res.operational_constraints,
            active_message=message
        )
        
        # Cognitive Penalty Loop
        safety_trace = ""
        current_constraints = con_res.operational_constraints
        
        for attempt in range(11):
            if not dec_res.decision.reply:
                break
                
            reply_text = dec_res.decision.reply
            needs_retry = False
            
            # 1. Verbosity Check
            if len(reply_text) > 65:
                penalty = f"CRITICAL PENALTY: Your reply is {len(reply_text)} characters. It MUST be under 50 characters. Cut the fat."
                logger.warning(f"Verbosity detected ({len(reply_text)} chars). Penalty applied.")
                safety_trace += "| Verbosity Penalty "
                current_constraints = f"{current_constraints}\n{penalty}"
                needs_retry = True
                
            # 2. Max-Expanded Persona & Structural Filter
            # Massively expanded targets for try-hard slang, internet trends, corporate/therapy padding, and weak conversational crutches.
            
            internet_acronyms = r"(lmao|lmfao|lol|rofl|tbh|ngl|fr|afaik|iirc|imho|imo|smh|ong|deadass|idk|rn|bc|cuz|pls|plz|tf|wtf|stfu|ik|iykyk|mf|stg|omg|omfg|nvm|jsyk|icymi|irl|tldr|imma|gonna|wanna|gotta|finna)"
            brainrot_and_trends = r"(rizz|sigma|skibidi|gyatt|cap|based|mid|aura|yap|yapping|ratio|sus|delulu|pookie|cooked|mewing|looksmaxxing|bussin|sheesh|yeet|lit|bet|glaze|glazing|npc|simp|cuck|chad|incel|slay|periodt)"
            corporate_and_therapy = r"(bandwidth|unpack|gaslight|gaslighting|problematic|synergy|mindset|journey|navigate|align|toxic|narcissist|narcissistic|trigger|triggered|trauma|boundaries|validate|validation|projecting|projection|leverage|pivot|holistic|paradigm|ideate|empower)"
            vocal_pauses = r"(uh|um|er|hmm|huh|oof|yikes|pfft|psh|ugh|welp|meh|haha|hehe|xd|kek|aww|oop|tsk|geez|jeez|dang|heck|yay|whoops)"
            weak_vocatives = r"(bro|bruh|broski|dawg|blud|homie|fam|bruv|buddy|chief|boss|kiddo|bucko|pal|chum|champ|sport|amigo|fella|fellas|peeps|yall)"
            adverbial_crutches = r"(literally|basically|essentially|actually|honestly|frankly|seriously|obviously|genuinely|totally|absolutely|utterly|practically|technically|ironically)"
            hedge_words = r"(kinda|sorta|maybe|perhaps|somewhat|probably|apparently|seemingly|supposedly|theoretically|hypothetically)"
            
            filler_pattern = re.compile(
                rf"\b({internet_acronyms}|{brainrot_and_trends}|{corporate_and_therapy}|{vocal_pauses}|{weak_vocatives}|{adverbial_crutches}|{hedge_words})\b", 
                re.IGNORECASE
            )
            
            if filler_pattern.search(reply_text):
                penalty = (
                    "CRITICAL PENALTY: Your response was rejected for using internet slang, therapy-speak, or nervous filler. "
                    "You must behave like a normal sane person who is educated in what to speak when to speak and how to speak properly."
                    "1. NO INTERNET SLANG OR TRENDS: Do not use brainrot, fleeting internet buzzwords, or memes. "
                    "2. NO CORPORATE/THERAPY SPEAK: Speak naturally, without academic padding, armchair psychology, or HR buzzwords. "
                    "3. NOT A STRICT DAD: Do not moralize, lecture, or sound like a disappointed father scolding a child. "
                    "Speak with effortless, grounded clarity. Deliver the insult as a simple, devastating, and natural observation."
                )
                logger.warning("Max-Expanded Persona/Filler filter triggered. Penalty applied.")
                safety_trace += "| Next-Level Persona Penalty "
                current_constraints = f"{current_constraints}\n{penalty}"
                needs_retry = True
                
            # 3. Self-Insult Check
            else:
                audit_res = self.safety_auditor(
                    active_message=message,
                    proposed_reply=reply_text
                )
                if audit_res.audit.is_self_roast:
                    logger.warning(f"Self-insult detected by SafetyAuditor! Penalty applied. Reason: {audit_res.audit.reasoning}")
                    safety_trace += f"| Safety Penalty: {audit_res.audit.reasoning} "
                    current_constraints = f"{current_constraints}\nCRITICAL PENALTY: Your previous draft insulted AN1. DO NOT DO THIS. You must rewrite your response. Reason: {audit_res.audit.reasoning}"
                    needs_retry = True
            
            if needs_retry:
                if attempt == 10:
                    logger.error("Cognitive Penalty Loop exhausted. Erasing non-compliant draft.")
                    dec_res.decision.reply = ""
                    dec_res.decision.reaction = None
                    dec_res.reasoning = "Silenced after 10 failed cognitive compliance checks."
                    break
                    
                dec_res = self.decision_engine(
                    tactical_objective=miss_res.tactical_objective,
                    operational_constraints=current_constraints,
                    active_message=message
                )
            else:
                safety_trace += "| All Checks Passed "
                break
        
        final_method = dec_res.decision.response_method
        final_reaction = dec_res.decision.reaction
        final_reply = dec_res.decision.reply
        
        full_reasoning = (
            f"ID Trace: {id_res.reasoning}\n"
            f"Mission Trace: {miss_res.reasoning}\n"
            f"Safety Trace: {safety_trace.strip()}\n"
            f"Decision Trace: Selected {final_method}"
        )
        
        return dspy.Prediction(
            reaction=final_reaction,
            reply=final_reply,
            reasoning=full_reasoning
        )

combat_engine = AN1CombatEngine()
triage_engine = dspy.Predict(TriageSignature)

def triage_node(state: CombatState):
    if state.get("force_engage", False):
        logger.info("Triage bypassed: Force engagement triggered (DM or Override).")
        return {"should_engage": True}
        
    try:
        res = triage_pool.execute_with_retry(
            triage_engine,
            chat_history=state["history"],
            active_message=state["message"],
            location=state["location"],
            is_direct_interaction=str(state["is_direct"])
        )
        engage = res.decision.should_engage if (res and hasattr(res, "decision") and res.decision) else False
        logger.info(f"Triage processed | Engage: {engage}")
        return {"should_engage": engage}
    except Exception as e:
        logger.error(f"Triage Execution Error: {e}")
        fallback_engage = state.get("is_direct", False)
        logger.info(f"Triage fallback engaged | Fallback Engage: {fallback_engage}")
        return {"should_engage": fallback_engage}

def combat_node(state: CombatState):
    try:
        full_history = state.get("combat_history", state["history"])
        
        res = nvidia_combat_pool.execute_with_retry(
            combat_engine,
            history=full_history,
            graph=state["graph"],
            user=state["user"],
            message=state["message"],
            location=state["location"]
        )
        
        # Wire to detached Teleprompter logs
        try:
            from app.teleprompter.logger import OptimizationLogger
            OptimizationLogger().log_inference(full_history, state["graph"], state["user"], state["message"], state["location"])
        except Exception:
            pass
        
        reply_val = res.reply if str(res.reply).lower() not in ["none", "null", ""] else ""
        reaction_val = res.reaction if str(res.reaction).lower() not in ["none", "null", ""] else None
        
        if reply_val:
            reply_val = sanitize_think_tags(reply_val)
        
        return {
            "reply": reply_val,
            "reaction": reaction_val,
            "reasoning": res.reasoning
        }
    except Exception as e:
        logger.error(f"NVIDIA Combat Error: {e}")
        return {"reply": "", "reaction": None, "reasoning": "Combat engine failure."}

def route_engagement(state: CombatState):
    if state["should_engage"]:
        return "combat"
    return "end"

# Compile Workflow
workflow = StateGraph(CombatState)
workflow.add_node("triage", triage_node)
workflow.add_node("combat", combat_node)
workflow.set_entry_point("triage")
workflow.add_conditional_edges("triage", route_engagement, {"combat": "combat", "end": END})
workflow.add_edge("combat", END)
compiled_vrag_agent = workflow.compile()

class VRAGEngine(BaseEngine):
    """
    The Experimental GraphRAG Engine.
    Uses DSPy, LangGraph, and NetworkX.
    """
    
    def __init__(self):
        self.graph_repo = GraphRepository()
        self.chat_repo = ChatRepository()
        self.group_repo = GroupHistoryRepository()
    
    def engine_name(self) -> str:
        return "vrag"
        
    async def _fetch_tagged_profiles(self, tagged_users: list, max_targets: int = 3) -> list:
        global_repo = GlobalMemoryRepository()
        profiles = []
        for u in tagged_users[:max_targets]:
            uid = getattr(u, 'id', '') or u.get('id', '') if isinstance(u, dict) else u.id
            username = getattr(u, 'username', '') or u.get('username', '') if isinstance(u, dict) else u.username
            if not username:
                continue
            memory_key = f"Global:{username}"
            summary = await asyncio.to_thread(global_repo.get_profile, memory_key)
            if summary:
                profiles.append(f'<bystander username="{username}" id="{uid}">\n{summary.strip()}\n</bystander>')
            else:
                profiles.append(f'<bystander username="{username}" id="{uid}">\nNo intelligence gathered yet.\n</bystander>')
        return profiles
        
    async def _format_history(self, payload: IncomingPayload, for_triage: bool = False) -> str:
        user_key = f"{payload.group_name}:{payload.username}"
        
        if for_triage:
            history = await asyncio.to_thread(
                self.group_repo.get_recent_history if payload.group_name != "private_chat" else self.chat_repo.get_recent_history,
                payload.group_name if payload.group_name != "private_chat" else user_key,
                limit=10
            )
            return "\n".join([f"[{m.get('username', 'Unknown')}]: {m.get('content', '')}" for m in history])

        if payload.group_name == "private_chat":
            history = await asyncio.to_thread(
                self.chat_repo.get_recent_history, user_key, limit=settings.MAX_HISTORY_MESSAGES
            )
        else:
            history = await asyncio.to_thread(
                self.group_repo.get_recent_history, payload.group_name, limit=settings.GROUP_HISTORY_SLICE
            )
            
        return "\n".join([f"[{m.get('username', 'Unknown')}]: {m.get('content', '')}" for m in history])
        
    async def _format_graph(self, payload: IncomingPayload) -> str:
        user_key = f"{payload.group_name}:{payload.username}"
        user_graph = await asyncio.to_thread(self.graph_repo.get_user_graph, user_key)
        group_graph = await asyncio.to_thread(self.graph_repo.get_group_graph, payload.group_name)
        
        return build_networkx_context(payload.username, user_graph, group_graph)
        
    async def process(self, payload: IncomingPayload) -> EngineResponse:
        is_private = (payload.group_name == "private_chat")
        
        triage_history_str = await self._format_history(payload, for_triage=True)
        combat_history_str = await self._format_history(payload, for_triage=False)
        graph_str = await self._format_graph(payload)
        
        tagged_profiles = await self._fetch_tagged_profiles(payload.tagged_users)
        if tagged_profiles:
            graph_str += "\n\n--- TAGGED BYSTANDER DOSSIERS ---\n" + "\n\n".join(tagged_profiles)
        
        initial_state = {
            "history": triage_history_str,
            "combat_history": combat_history_str,
            "graph": graph_str,
            "user": payload.username,
            "message": payload.message,
            "location": payload.channel,
            "is_direct": payload.force_reply or ("@an1" in payload.message.lower()),
            "force_engage": is_private or payload.force_reply,
            "should_engage": False,
            "reply": "",
            "reaction": None,
            "reasoning": "Triage bypassed combat engine. (Silence)"
        }
        
        final_state = await asyncio.to_thread(compiled_vrag_agent.invoke, initial_state)
        
        return EngineResponse(
            reply=final_state.get("reply") or None,
            reaction=final_state.get("reaction") or None,
            engine_used=self.engine_name() if final_state.get("should_engage") else "triage_silence"
        )
