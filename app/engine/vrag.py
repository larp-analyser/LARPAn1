import dspy
import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.engine.base import BaseEngine
from app.api.models import IncomingPayload, EngineResponse
from app.prompts.dspy_signatures import (
    IdentitySignature,
    MissionSignature,
    ConstraintsSignature,
    DecisionSignature,
    TriageSignature
)
from app.core.llm_balancer import triage_pool, nvidia_combat_pool
from app.db.repositories import GraphRepository, ChatRepository, GroupHistoryRepository
from app.engine.graph_analyzer import build_networkx_context

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

class PSI09CombatEngine(dspy.Module):
    def __init__(self):
        super().__init__()
        self.identity = dspy.ChainOfThought(IdentitySignature)
        self.mission = dspy.ChainOfThought(MissionSignature)
        self.constraints = dspy.ChainOfThought(ConstraintsSignature) 
        self.decision_engine = dspy.Predict(DecisionSignature) 
        
    def forward(self, history, graph, user, message, location):
        id_res = self.identity(graph_context=graph, target_user=user)
        miss_res = self.mission(dynamic_persona=id_res.dynamic_persona, chat_history=history, active_message=message, location=location)
        con_res = self.constraints(tactical_objective=miss_res.tactical_objective, active_message=message)
        
        dec_res = self.decision_engine(
            tactical_objective=miss_res.tactical_objective,
            operational_constraints=con_res.operational_constraints,
            active_message=message
        )
        
        final_method = dec_res.decision.response_method
        final_reaction = dec_res.decision.reaction
        final_reply = dec_res.decision.reply
        
        full_reasoning = (
            f"ID Trace: {id_res.reasoning}\n"
            f"Mission Trace: {miss_res.reasoning}\n"
            f"Decision Trace: Selected {final_method}"
        )
        
        return dspy.Prediction(
            reaction=final_reaction,
            reply=final_reply,
            reasoning=full_reasoning
        )

combat_engine = PSI09CombatEngine()
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
            
            # Pydantic schema normalization
            reply_val = res.reply if str(res.reply).lower() not in ["none", "null", ""] else ""
            reaction_val = res.reaction if str(res.reaction).lower() not in ["none", "null", ""] else None
            
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
        
    def _format_history(self, payload: IncomingPayload) -> str:
        user_key = f"{payload.group_name}:{payload.username}"
        if payload.group_name == "private_chat":
            history = self.chat_repo.get_recent_history(user_key, limit=16)
        else:
            history = self.group_repo.get_recent_history(payload.group_name, limit=16)
            
        return "\n".join([f"[{m.get('username', 'Unknown')}]: {m.get('content', '')}" for m in history])
        
    def _format_graph(self, payload: IncomingPayload) -> str:
        user_key = f"{payload.group_name}:{payload.username}"
        user_graph = self.graph_repo.get_user_graph(user_key)
        group_graph = self.graph_repo.get_group_graph(payload.group_name)
        
        return build_networkx_context(payload.username, user_graph, group_graph)
        
    async def process(self, payload: IncomingPayload) -> EngineResponse:
        is_private = (payload.group_name == "private_chat")
        
        initial_state = {
            "history": self._format_history(payload),
            "graph": self._format_graph(payload),
            "user": payload.username,
            "message": payload.message,
            "location": payload.channel if not is_private else "Direct Message",
            "is_direct": payload.force_reply or "@PSI-09" in payload.message,
            "should_engage": False,
            "reply": "",
            "reaction": None,
            "reasoning": "Triage bypassed combat engine. (Silence)"
        }
        
        import asyncio
        
        # Invoke LangGraph in a separate thread so DSPy doesn't block FastAPI's async event loop
        final_state = await asyncio.to_thread(compiled_vrag_agent.invoke, initial_state)
        
        return EngineResponse(
            reply=final_state.get("reply") or None,
            reaction=final_state.get("reaction") or None,
            engine_used=self.engine_name() if final_state.get("should_engage") else "triage_silence"
        )
