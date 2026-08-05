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
        
        auditor = dspy.Predict(SelfInsultPreventionSignature)
        
        # 3. Define Metric (Now 1:1 synced with VRAG penalty loops)
        def combat_metric(example, pred, trace=None):
            if not pred.reply or str(pred.reply) == "None":
                # If it's a reaction only, ensure a valid emoji was actually provided
                if not pred.reaction or str(pred.reaction) == "None":
                    return 0.0
                return 1.0 # Valid reaction-only pass
                
            reply_text = str(pred.reply)
            
            if len(reply_text) > 80:
                return 0.0
                
            if re.search(r'[\n\r=—–~#*>]|--|\.\.\.|…', reply_text):
                return 0.0
                
            if re.search(r':|\b(mode|activated|cue|reflex|breath|panic|flag|status|online|offline|address|network|system)\b|[.!?]\s+[A-Z]', reply_text, re.IGNORECASE):
                return 0.0
                
            if re.search(r'^(so,?|oh,? so|let me get this straight|you think|you really think|are you saying)\b', reply_text.strip(), re.IGNORECASE):
                return 0.0
                
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
                
                from app.engine.vrag import AN1CombatEngine
                import app.engine.vrag as vrag_module
                
                # Load weights into a new instance to prevent active inference corruption
                new_engine = AN1CombatEngine(load_compiled=False)
                new_engine.load(temp_path)
                
                # Atomically swap the global reference safely
                vrag_module.combat_engine = new_engine
                logger.info("[TELEPROMPTER] Live engine dynamically swapped with new optimized weights.")
            finally:
                # This executes EVEN IF an error occurs above, preventing the disk leak
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
        logger.info("[TELEPROMPTER] Compilation Complete! Weights saved to MongoDB.")
    except Exception as e:
        logger.error(f"[TELEPROMPTER] Error during optimization: {e}")
    finally:
        optimization_lock.release()
