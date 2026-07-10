from app.db.mongo import MongoDB
from datetime import datetime, timezone

class OptimizationLogger:
    """
    Detached side-hustle logger. 
    Stores exact inputs fed to the Combat Engine so the Teleprompter can rebuild the context identically.
    """
    def __init__(self):
        self.collection = MongoDB.get_collection("optimization_logs")
        
    def log_inference(self, history: str, graph: str, user: str, message: str, location: str):
        self.collection.insert_one({
            "history": history,
            "graph": graph,
            "user": user,
            "message": message,
            "location": location,
            "timestamp": datetime.now(timezone.utc)
        })
        
    def get_recent_examples(self, limit: int = 100) -> list:
        cursor = self.collection.find().sort("timestamp", -1).limit(limit)
        return list(cursor)
