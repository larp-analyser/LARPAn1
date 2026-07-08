from app.db.mongo import MongoDB
from datetime import datetime, timezone
import time
import threading
from app.core.config import settings

UTC = timezone.utc

class TTLCache:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self.cache = {}
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['time'] < self.ttl:
                    return entry['value']
                else:
                    del self.cache[key]
            return None

    def set(self, key: str, value):
        with self.lock:
            self.cache[key] = {'value': value, 'time': time.time()}
            
    def delete(self, key: str):
        with self.lock:
            if key in self.cache:
                del self.cache[key]

# Shared TTL cache instances
_profile_cache = TTLCache(settings.MEMORY_TTL)
_graph_cache = TTLCache(settings.MEMORY_TTL)


class ChatRepository:
    def __init__(self):
        self.collection = MongoDB.get_collection("chat_history")
        
    def store_message(self, user_key: str, message_data: dict):
        self.collection.update_one(
            {"_id": user_key},
            {"$push": {"messages": message_data}},
            upsert=True
        )
        
    def get_recent_history(self, user_key: str, limit: int = 30) -> list:
        doc = self.collection.find_one({"_id": user_key}, {"messages": {"$slice": -limit}})
        return doc.get("messages", []) if doc else []

class GroupHistoryRepository:
    def __init__(self):
        self.collection = MongoDB.get_collection("group_history")

    def store_message(self, group_name: str, message_data: dict):
        self.collection.update_one(
            {"_id": group_name},
            {"$push": {"messages": message_data}},
            upsert=True
        )
        
    def get_recent_history(self, group_name: str, limit: int = 80) -> list:
        doc = self.collection.find_one({"_id": group_name}, {"messages": {"$slice": -limit}})
        return doc.get("messages", []) if doc else []

class MemoryRepository:
    """Handles Roastbot's text-based psychological profiles"""
    def __init__(self):
        self.collection = MongoDB.get_collection("user_memory")

    def get_profile(self, user_key: str) -> str:
        cached = _profile_cache.get(user_key)
        if cached is not None:
            return cached
            
        doc = self.collection.find_one({"_id": user_key})
        val = doc.get("summary", "") if doc else ""
        _profile_cache.set(user_key, val)
        return val
        
    def update_profile(self, user_key: str, summary: str):
        self.collection.update_one(
            {"_id": user_key},
            {"$set": {"summary": summary, "last_updated": datetime.now(UTC)}},
            upsert=True
        )
        _profile_cache.set(user_key, summary)

class GlobalHistoryRepository:
    def __init__(self):
        self.collection = MongoDB.get_collection("global_history")

    def store_message(self, global_key: str, message_data: dict):
        self.collection.update_one(
            {"_id": global_key},
            {"$push": {"messages": message_data}},
            upsert=True
        )
        
    def get_recent_history(self, global_key: str, limit: int = 80) -> list:
        doc = self.collection.find_one({"_id": global_key}, {"messages": {"$slice": -limit}})
        return doc.get("messages", []) if doc else []

class GlobalMemoryRepository:
    def __init__(self):
        self.collection = MongoDB.get_collection("global_memory")

    def get_profile(self, global_key: str) -> str:
        cached = _profile_cache.get(global_key)
        if cached is not None:
            return cached
            
        doc = self.collection.find_one({"_id": global_key})
        val = doc.get("summary", "") if doc else ""
        _profile_cache.set(global_key, val)
        return val
        
    def update_profile(self, global_key: str, summary: str):
        self.collection.update_one(
            {"_id": global_key},
            {"$set": {"summary": summary, "last_updated": datetime.now(UTC)}},
            upsert=True
        )
        _profile_cache.set(global_key, summary)

class GraphRepository:
    """Handles vRAG's entity and relationship graphs"""
    def __init__(self):
        self.users = MongoDB.get_collection("graph_users")
        self.groups = MongoDB.get_collection("graph_groups")
        
    def get_user_graph(self, user_key: str) -> dict:
        cached = _graph_cache.get(user_key)
        if cached is not None:
            return cached
            
        doc = self.users.find_one({"_id": user_key})
        val = doc.get("graph_data", {"entities": [], "relationships": []}) if doc else {"entities": [], "relationships": []}
        _graph_cache.set(user_key, val)
        return val

    def update_user_graph(self, user_key: str, graph_data: dict):
        self.users.update_one(
            {"_id": user_key},
            {"$set": {"graph_data": graph_data, "last_updated": datetime.now(UTC)}},
            upsert=True
        )
        _graph_cache.set(user_key, graph_data)

    def get_group_graph(self, group_name: str) -> dict:
        cached = _graph_cache.get(f"group_{group_name}")
        if cached is not None:
            return cached
            
        doc = self.groups.find_one({"_id": group_name})
        val = doc.get("graph_data", {"entities": [], "relationships": []}) if doc else {"entities": [], "relationships": []}
        _graph_cache.set(f"group_{group_name}", val)
        return val

    def update_group_graph(self, group_name: str, graph_data: dict):
        self.groups.update_one(
            {"_id": group_name},
            {"$set": {"graph_data": graph_data, "last_updated": datetime.now(UTC)}},
            upsert=True
        )
        _graph_cache.set(f"group_{group_name}", graph_data)
