import dspy
import asyncio
import logging
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

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
from app.db.repositories import GraphRepository, ChatRepository, GroupHistoryRepository
from app.engine.graph_analyzer import build_networkx_context
from app.core.utils import sanitize_think_tags

logger = logging.getLogger(__name__)

class CombatState(TypedDict):
    history: str
    graph: str
    user: str
    message: str
    location: str
    is_direct: bool
    
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
                    temp_path = "/tmp/combat_engine.json"
                    with open(temp_path, "w", encoding="utf-8") as f:
                        f.write(doc["weights"])
                    self.load(temp_path)
            except Exception:
                pass # If loading fails, just proceed normally (bulletproof)
        
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
        
        for _ in range(3):
            if not dec_res.decision.reply:
                break
                
            reply_text = dec_res.decision.reply
            needs_retry = False
            
            # 1. Verbosity Check
            if len(reply_text) > 150:
                penalty = f"CRITICAL PENALTY: Your reply is {len(reply_text)} characters. It MUST be under 150 characters. Cut the fat."
                logger.warning(f"Verbosity detected ({len(reply_text)} chars). Penalty applied.")
                safety_trace += "| Verbosity Penalty "
                current_constraints = f"{current_constraints}\n{penalty}"
                needs_retry = True
                
            # 2. French Filler Check
            elif re.search(r'\b(oh|ah|alas|ouais|voilà)\b', reply_text, re.IGNORECASE):
                penalty = "CRITICAL PENALTY: Remove all filler words (Oh, Ah, Alas, etc.). Speak directly and brutally."
                logger.warning("French Filler detected. Penalty applied.")
                safety_trace += "| Filler Penalty "
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
    max_retries = len(triage_pool.models) if triage_pool.models else 1
    
    for attempt in range(max_retries):
        try:
            current_lm, current_index = triage_pool.get_current()
            with dspy.context(lm=current_lm):
                res = triage_engine(
                    chat_history=state["history"], 
                    active_message=state["message"],
                    location=state["location"],
                    is_direct_interaction=str(state["is_direct"]) 
                )
            engage = res.decision.should_engage
            logger.info(f"Triage processed by: {current_lm.model} | Engage: {engage}")
            return {"should_engage": engage}
        except Exception as e:
            logger.error(f"Triage Error: {e}")
            if "429" in str(e) or "rate limit" in str(e).lower():
                triage_pool.advance(current_index)
            else:
                break
                
    logger.error("ALL TRIAGE MODELS FAILED OR NONE CONFIGURED. Defaulting to False.")
    return {"should_engage": False}

def combat_node(state: CombatState):
    max_retries = len(nvidia_combat_pool.models) if nvidia_combat_pool.models else 1
    
    for attempt in range(max_retries):
        try:
            current_lm = nvidia_combat_pool.get_next()
            with dspy.context(lm=current_lm):
                res = combat_engine(
                    history=state["history"], 
                    graph=state["graph"], 
                    user=state["user"], 
                    message=state["message"],
                    location=state["location"]
                )
            
            # Wire to detached Teleprompter logs
            try:
                from app.teleprompter.logger import OptimizationLogger
                OptimizationLogger().log_inference(state["history"], state["graph"], state["user"], state["message"], state["location"])
            except Exception:
                pass
            
            # Pydantic schema normalization
            reply_val = res.reply if str(res.reply).lower() not in ["none", "null", ""] else ""
            reaction_val = res.reaction if str(res.reaction).lower() not in ["none", "null", ""] else None
            
            # Sanitize think-tags from combat output (matching roastbot)
            if reply_val:
                reply_val = sanitize_think_tags(reply_val)
            
            return {
                "reply": reply_val,
                "reaction": reaction_val,
                "reasoning": res.reasoning
            }
        except Exception as e:
            logger.error(f"NVIDIA Combat Error (Attempt {attempt + 1}): {e}")
            
    logger.error("ALL NVIDIA COMBAT KEYS FAILED OR NONE CONFIGURED.")
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
        
    async def _format_history(self, payload: IncomingPayload) -> str:
        user_key = f"{payload.group_name}:{payload.username}"
        if payload.group_name == "private_chat":
            history = await asyncio.to_thread(self.chat_repo.get_recent_history, user_key, limit=16)
        else:
            history = await asyncio.to_thread(self.group_repo.get_recent_history, payload.group_name, limit=16)
            
        return "\n".join([f"[{m.get('username', 'Unknown')}]: {m.get('content', '')}" for m in history])
        
    async def _format_graph(self, payload: IncomingPayload) -> str:
        user_key = f"{payload.group_name}:{payload.username}"
        user_graph = await asyncio.to_thread(self.graph_repo.get_user_graph, user_key)
        group_graph = await asyncio.to_thread(self.graph_repo.get_group_graph, payload.group_name)
        
        return build_networkx_context(payload.username, user_graph, group_graph)
        
    async def process(self, payload: IncomingPayload) -> EngineResponse:
        is_private = (payload.group_name == "private_chat")
        
        # Build descriptive location string (matching legacy vRAG)
        location_str = "Private Direct Message" if is_private else f"Server: {payload.group_name} | Channel: #{payload.channel}"
        
        initial_state = {
            "history": await self._format_history(payload),
            "graph": await self._format_graph(payload),
            "user": payload.username,
            "message": f"[{payload.username}]: {payload.message}",
            "location": location_str,
            "is_direct": payload.force_reply or ("@an1" in payload.message.lower()),
            "should_engage": False,
            "reply": "",
            "reaction": None,
            "reasoning": "Triage bypassed combat engine. (Silence)"
        }
        
        # Invoke LangGraph in a separate thread so DSPy doesn't block FastAPI's async event loop
        final_state = await asyncio.to_thread(compiled_vrag_agent.invoke, initial_state)
        
        return EngineResponse(
            reply=final_state.get("reply") or None,
            reaction=final_state.get("reaction") or None,
            engine_used=self.engine_name() if final_state.get("should_engage") else "triage_silence"
        )
