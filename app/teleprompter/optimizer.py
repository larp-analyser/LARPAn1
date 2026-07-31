import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
import logging
import re
import os
import threading
import tempfile

from app.engine.vrag import AN1CombatEngine
from app.prompts.dspy_signatures import SelfInsultPreventionSignature
from app.teleprompter.logger import OptimizationLogger
from app.db.mongo import MongoDB
from app.core.llm_balancer import nvidia_combat_pool

logger = logging.getLogger(__name__)

optimization_lock = threading.Lock()

def run_teleprompter_task():
    if not optimization_lock.acquire(blocking=False):
        logger.warning("[TELEPROMPTER] Optimization is already running. Aborting concurrent request to prevent API and I/O exhaustion.")
        return
        
    try:
        logger.info("[TELEPROMPTER] Starting DSPy Deep Optimization Side-Hustle...")
        
        # 1. Fetch massive historical dataset into RAM
        log_repo = OptimizationLogger()
        raw_logs = log_repo.get_recent_examples(limit=2000)
        
        if len(raw_logs) < 10:
            logger.info("[TELEPROMPTER] Not enough data to optimize. Aborting.")
            return
            
        # 2. Build DSPy Dataset
        trainset = []
        for log in raw_logs:
            example = dspy.Example(
                history=log["history"],
                graph=log["graph"],
                user=log["user"],
                message=log["message"],
                location=log["location"]
            ).with_inputs('history', 'graph', 'user', 'message', 'location')
            trainset.append(example)
            
        # 3. Define Metric (Now 1:1 synced with VRAG penalty loops)
        def combat_metric(example, pred, trace=None):
            reply_text = str(pred.reply)
            
            # Constraint 1: Verbosity
            if len(reply_text) > 80:
                return 0.0
                
            # Constraint 2: Persona & Structural Filter
            internet_acronyms = r"(lmao|lmfao|lol|rofl|tbh|ngl|fr|afaik|iirc|imho|imo|smh|ong|deadass|idk|rn|bc|cuz|pls|plz|tf|wtf|stfu|ik|iykyk|ur|u|mf|stg|omg|omfg|nvm|jsyk|icymi|irl|tldr|imma|gonna|wanna|gotta|finna)"
            brainrot_and_trends = r"(rizz|sigma|skibidi|gyatt|cap|based|mid|aura|yap|yapping|ratio|sus|delulu|pookie|cooked|mewing|looksmaxxing|bussin|sheesh|yeet|lit|bet|glaze|glazing|npc|simp|cuck|chad|incel|slay|periodt)"
            corporate_and_therapy = r"(bandwidth|unpack|gaslight|gaslighting|problematic|synergy|mindset|journey|navigate|align|toxic|narcissist|narcissistic|trigger|triggered|trauma|boundaries|validate|validation|projecting|projection|leverage|pivot|holistic|paradigm|ideate|empower)"
            vocal_pauses = r"(uh|um|er|hmm|huh|oof|yikes|pfft|psh|ugh|welp|meh|haha|hehe|xd|kek|aww|oop|tsk|geez|jeez|dang|heck|yay|whoops)"
            weak_vocatives = r"(bro|bruh|broski|dawg|blud|homie|fam|bruv|buddy|chief|boss|kiddo|bucko|pal|chum|champ|sport|amigo|fella|fellas|peeps|yall)"
            adverbial_crutches = r"(literally|basically|essentially|actually|honestly|frankly|seriously|obviously|genuinely|totally|absolutely|utterly|practically|technically|ironically)"
            hedge_words = r"(kinda|sorta|maybe|perhaps|somewhat|probably|apparently|seemingly|supposedly|theoretically|hypothetically)"
            observed_filler = r"(says|vibe|relevant|words|sentences)"
            
            filler_pattern = re.compile(
                rf"\b({internet_acronyms}|{brainrot_and_trends}|{corporate_and_therapy}|{vocal_pauses}|{weak_vocatives}|{adverbial_crutches}|{hedge_words}|{observed_filler})\b", 
                re.IGNORECASE
            )
            if filler_pattern.search(reply_text):
                return 0.0
                
            # Constraint 3: Formatting Tropes
            if re.search(r'[\n\r=—–~#*>]|--|\.\.\.|…', reply_text):
                return 0.0
                
            # Constraint 4: Cinematic Narrator
            if re.search(r':|\b(mode|activated|cue|reflex|breath|panic|flag|status|online|offline|address|network|system)\b|[.!?]\s+[A-Z]', reply_text, re.IGNORECASE):
                return 0.0
                
            # Constraint 5: Cliches and Rhetorical Questions
            if re.search(r'\?|\"|\b(imagine|ah yes|oh look|it[\'’]?s funny how|the fact that|speaks volumes|try harder|do better|next|make it make sense|let that sink in)\b', reply_text, re.IGNORECASE):
                return 0.0
                
                
            # Constraint 8: Pretentious Academics
            if re.search(r'\b(demonstrates|indicates|illustrates|profound|inadequate|deficit|exhibits|displays|fascinating|intriguing|indicative|perpetuate|manifests)\b', reply_text, re.IGNORECASE):
                return 0.0
                
            # Constraint 9: Echoing
            if re.search(r'^(so,?|oh,? so|let me get this straight|you think|you really think|are you saying)\b', reply_text.strip(), re.IGNORECASE):
                return 0.0
                
            # Constraint 10: Unsolicited Advice
            if re.search(r'\b(maybe you should|try to|it[\'’]?s time to|grow up|do yourself a favor|re-evaluate|reconsider|take a break|seek help)\b', reply_text, re.IGNORECASE):
                return 0.0
                
            # Teacher model safety audit
            auditor = dspy.Predict(SelfInsultPreventionSignature)
            res = auditor(active_message=example.message, proposed_reply=reply_text)
            if res.audit.is_self_roast:
                return 0.0
                
            return 1.0
            
        # 4. Setup Advanced Random Search Teleprompter
        current_lm = nvidia_combat_pool.get_next()
        
        with dspy.context(lm=current_lm):
            teleprompter = BootstrapFewShotWithRandomSearch(
                metric=combat_metric,
                max_bootstrapped_demos=6,
                max_labeled_demos=8,
                num_candidate_programs=10, 
                num_threads=4 
            )
            
            # 5. Compile
            student = AN1CombatEngine(load_compiled=False)
            compiled_engine = teleprompter.compile(student, trainset=trainset)
            
            # 6. Save optimized weights temporarily
            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as tmp_file:
                temp_path = tmp_file.name
                
            try:    
                compiled_engine.save(temp_path)
                with open(temp_path, "r", encoding="utf-8") as f:
                    weights_json = f.read()
                    
                weights_col = MongoDB.get_collection("compiled_weights")
                weights_col.update_one(
                    {"_id": "combat_engine"},
                    {"$set": {"weights": weights_json}},
                    upsert=True
                )
                
                from app.engine.vrag import combat_engine
                combat_engine.load(temp_path)
                logger.info("[TELEPROMPTER] Live engine dynamically updated with new weights.")
            finally:
                # This executes EVEN IF an error occurs above, preventing the disk leak
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # 7. Hot-reload the live engine in memory
            from app.engine.vrag import combat_engine
            combat_engine.load(temp_path)
            logger.info("[TELEPROMPTER] Live engine dynamically updated with new weights.")
            
            # Clean up unique temp file
            os.remove(temp_path)
            
        logger.info("[TELEPROMPTER] Compilation Complete! Weights saved to MongoDB.")
    except Exception as e:
        logger.error(f"[TELEPROMPTER] Error during optimization: {e}")
    finally:
        optimization_lock.release()
