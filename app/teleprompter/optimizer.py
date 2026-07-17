import dspy
from dspy.teleprompt import BootstrapFewShot
import logging
import re
import os
import threading
import tempfile

from app.engine.vrag import AN1CombatEngine
from app.prompts.dspy_signatures import SelfInsultPreventionSignature
from app.teleprompter.logger import OptimizationLogger
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
        logger.info("[TELEPROMPTER] Starting DSPy Optimization Side-Hustle...")
        
        # 1. Fetch historical raw inputs
        log_repo = OptimizationLogger()
        raw_logs = log_repo.get_recent_examples(limit=50)
        
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
            
        # 3. Define Metric
        def combat_metric(example, pred, trace=None):
            reply_text = str(pred.reply)
            
            if len(reply_text) > 150:
                return 0.0
                
            # Teacher model audit
            auditor = dspy.Predict(SelfInsultPreventionSignature)
            res = auditor(active_message=example.message, proposed_reply=reply_text)
            if res.audit.is_self_roast:
                return 0.0
                
            return 1.0
            
        # 4. Setup Teleprompter
        current_lm = nvidia_combat_pool.get_next()
        
        with dspy.context(lm=current_lm):
            teleprompter = BootstrapFewShot(
                metric=combat_metric,
                max_bootstrapped_demos=8,
                max_labeled_demos=8  # Kept low to preserve LLM creativity and prevent overfitting
            )
            
            # 5. Compile
            student = AN1CombatEngine(load_compiled=False)
            compiled_engine = teleprompter.compile(student, trainset=trainset)
            
            # 6. Save optimized weights temporarily using process-isolated unique file
            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as tmp_file:
                temp_path = tmp_file.name
                
            compiled_engine.save(temp_path)
            
            with open(temp_path, "r", encoding="utf-8") as f:
                weights_json = f.read()
                
            weights_col = MongoDB.get_collection("compiled_weights")
            weights_col.update_one(
                {"_id": "combat_engine"},
                {"$set": {"weights": weights_json}},
                upsert=True
            )
            
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
