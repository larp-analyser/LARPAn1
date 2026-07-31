import faiss
import numpy as np
from fastembed import TextEmbedding
import threading
import logging
import hashlib
import json
import os

from huggingface_hub import HfApi, hf_hub_download
from app.core.config import settings

logger = logging.getLogger(__name__)

class HFPersistence:
    """Handles background upload/download of serialized FAISS files to Hugging Face Datasets."""
    def __init__(self):
        self.api = HfApi()
        self.repo_id = settings.HF_DATASET_ID
        self.token = settings.HF_TOKEN

    def download_file(self, filename: str, dest_path: str) -> bool:
        if not self.token or not self.repo_id:
            return False
        try:
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id, 
                filename=filename, 
                repo_type="dataset", 
                token=self.token
            )
            # Move the cached download to our working destination
            os.system(f"cp {downloaded_path} {dest_path}")
            return True
        except Exception as e:
            logger.info(f"[VECTOR] {filename} not found in HF Dataset or download failed: {e}")
            return False

    def upload_file(self, filename: str, src_path: str):
        if not self.token or not self.repo_id:
            return
        try:
            self.api.upload_file(
                path_or_fileobj=src_path,
                path_in_repo=filename,
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
        except Exception as e:
            logger.error(f"[VECTOR] HF Dataset upload failed for {filename}: {e}")

class VectorMemory:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VectorMemory, cls).__new__(cls)
                cls._instance._init_store()
            return cls._instance

    def _init_store(self):
        logger.info("[VECTOR] Initializing FastEmbed (BAAI/bge-large-en-v1.5)...")
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
        self.dimension = 1024 
        
        self.metadata = [] 
        self.seen_hashes = set() 
        self.db_lock = threading.Lock()
        
        self.cloud_storage = HFPersistence()
        self._sync_timer = None

        logger.info("[VECTOR] Fetching FAISS index and metadata from Hugging Face...")
        
        index_loaded = False
        if self.cloud_storage.download_file("faiss_index.bin", "/tmp/faiss_index.bin"):
            try:
                self.index = faiss.read_index("/tmp/faiss_index.bin")
                index_loaded = True
            except Exception as e:
                logger.error(f"[VECTOR] Failed to read FAISS index from disk: {e}")
                
        if self.cloud_storage.download_file("vector_metadata.json", "/tmp/vector_metadata.json"):
            try:
                with open("/tmp/vector_metadata.json", "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                for m in self.metadata:
                    self.seen_hashes.add(self._generate_hash(m["username"], m["content"]))
            except Exception as e:
                logger.error(f"[VECTOR] Failed to read metadata JSON from disk: {e}")

        if not index_loaded:
            logger.info("[VECTOR] No historical index found on HF. Starting fresh.")
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            logger.info(f"[VECTOR] Successfully loaded {len(self.metadata)} receipts into RAM from HF.")
            
        logger.info("[VECTOR] Engine Online.")

    def _generate_hash(self, username: str, content: str) -> str:
        """Generates a unique fingerprint for a message to prevent duplicate embedding."""
        return hashlib.md5(f"{username}::{content}".encode('utf-8')).hexdigest()

    def _schedule_sync(self):
        """Debounces the Cloud upload to prevent API Rate Limits. Waits 30s after the chat settles."""
        with self.db_lock:
            if self._sync_timer is not None:
                self._sync_timer.cancel()
            self._sync_timer = threading.Timer(30.0, self.force_sync)
            self._sync_timer.start()

    def force_sync(self):
        """Compiles the RAM data to local disk and streams it to Hugging Face."""
        logger.info("[VECTOR] Compiling binary memory payload...")
        with self.db_lock:
            faiss.write_index(self.index, "/tmp/faiss_index.bin")
            with open("/tmp/vector_metadata.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata, f)
        
        # Executes network I/O outside of db_lock to avoid freezing the chat engine
        try:
            self.cloud_storage.upload_file("faiss_index.bin", "/tmp/faiss_index.bin")
            self.cloud_storage.upload_file("vector_metadata.json", "/tmp/vector_metadata.json")
            logger.info("[VECTOR] Hugging Face Backup complete.")
        except Exception as e:
            logger.error(f"[VECTOR] Hugging Face upload failed: {e}")

    def add_message(self, username: str, content: str, timestamp: str):
        """Embeds a single message, updates RAM, and schedules a background backup."""
        msg_hash = self._generate_hash(username, content)
        
        with self.db_lock:
            if msg_hash in self.seen_hashes:
                return # Skip duplicate

        text_to_embed = f"[{username}]: {content}"
        embeddings_generator = self.embedding_model.embed([text_to_embed])
        vector = next(embeddings_generator).astype(np.float32)
        
        with self.db_lock:
            self.index.add(np.array([vector]))
            self.metadata.append({
                "username": username,
                "content": content,
                "timestamp": timestamp
            })
            self.seen_hashes.add(msg_hash)
            
        self._schedule_sync()

    def add_batch_messages(self, messages: list):
        """Optimized batch ingestion for the backfill cronjob."""
        new_messages = []
        with self.db_lock:
            for msg in messages:
                msg_hash = self._generate_hash(msg["username"], msg["content"])
                if msg_hash not in self.seen_hashes:
                    new_messages.append(msg)
                    self.seen_hashes.add(msg_hash)

        if not new_messages:
            return 0

        texts_to_embed = [f"[{m['username']}]: {m['content']}" for m in new_messages]
        logger.info(f"[VECTOR] Embedding batch of {len(texts_to_embed)} historical messages...")
        
        embeddings_generator = self.embedding_model.embed(texts_to_embed)
        vectors = [vec.astype(np.float32) for vec in embeddings_generator]
        
        with self.db_lock:
            self.index.add(np.array(vectors))
            for i, msg in enumerate(new_messages):
                self.metadata.append({
                    "username": msg["username"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"]
                })
                
        self.force_sync()
        return len(new_messages)

    def search_similar(self, query: str, top_k: int = 5, username: str = None):
        query_vector = self.embedding_model.embed([query]) 
        query_vector = np.array(list(query_vector), dtype=np.float32) 
        
        distances, indices = self.index.search(query_vector, top_k * 3)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
                
            item = self.metadata[idx]
            if username and item.get("username", "").lower() != username.lower():
                continue
                
            results.append(item)
            if len(results) >= top_k:
                break
                
        return results

# Initialize the singleton immediately
vector_db = VectorMemory()
