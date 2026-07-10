import dspy
from dspy.teleprompt import BootstrapFewShot
import logging
import re
import os

from app.engine.vrag import AN1CombatEngine
from app.prompts.dspy_signatures import SelfInsultPreventionSignature
from app.teleprompter.logger import OptimizationLogger
from app.core.llm_balancer import nvidia_combat_pool

logger = logging.getLogger(__name__)

def run_teleprompter_task():
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
                
            if re.search(r'\b(oh|ah|alas|ouais|voilà)\b', reply_text, re.IGNORECASE):
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
                max_bootstrapped_demos=4,
                max_labeled_demos=16
            )
            
            # 5. Compile
            student = AN1CombatEngine()
            compiled_engine = teleprompter.compile(student, trainset=trainset)
            
            # 6. Save optimized weights
            save_dir = "app/teleprompter/compiled"
            os.makedirs(save_dir, exist_ok=True)
            compiled_engine.save(f"{save_dir}/combat_engine.json")
            
        logger.info("[TELEPROMPTER] Compilation Complete! Weights saved.")
    except Exception as e:
        logger.error(f"[TELEPROMPTER] Error during optimization: {e}")
