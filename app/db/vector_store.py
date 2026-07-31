import faiss
import numpy as np
from fastembed import TextEmbedding
import threading
import logging
import hashlib
import json
import os
import re
from rank_bm25 import BM25Okapi

from huggingface_hub import HfApi, hf_hub_download
from app.core.config import settings

logger = logging.getLogger(__name__)

def tokenize_text(text: str) -> list:
    """Helper tokenizer for BM25 keyword matching."""
    return re.findall(r'\w+', text.lower())

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
        self.rolling_buffer = []  # Holds the last 2 raw message dicts for live sliding window
        self.db_lock = threading.Lock()
        self.bm25 = None
        
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
                
                # Pre-fill rolling buffer with up to last 2 messages
                for m in self.metadata[-2:]:
                    self.rolling_buffer.append({"username": m["username"], "content": m["content"]})
            except Exception as e:
                logger.error(f"[VECTOR] Failed to read metadata JSON from disk: {e}")

        if not index_loaded:
            logger.info("[VECTOR] No historical index found on HF. Starting fresh.")
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            logger.info(f"[VECTOR] Successfully loaded {len(self.metadata)} receipts into RAM from HF.")
            
        self._rebuild_bm25()
        logger.info("[VECTOR] Engine Online with BM25 + FAISS Hybrid RRF Search.")

    def _generate_hash(self, username: str, content: str) -> str:
        """Generates a unique fingerprint for a message to prevent duplicate embedding."""
        return hashlib.md5(f"{username}::{content}".encode('utf-8')).hexdigest()

    def _build_context_window(self, previous_msgs: list, current_username: str, current_content: str) -> str:
        """Formats target message with up to 2 preceding messages using strict XML tagging."""
        xml_parts = ["<context_window>"]
        for msg in previous_msgs:
            xml_parts.append(f"  <utterance speaker=\"{msg['username']}\">{msg['content']}</utterance>")
        xml_parts.append(f"  <target_utterance speaker=\"{current_username}\">{current_content}</target_utterance>")
        xml_parts.append("</context_window>")
        return "\n".join(xml_parts)

    def _rebuild_bm25(self):
        """Builds or refreshes the in-memory BM25 index."""
        if not self.metadata:
            self.bm25 = None
            return
        corpus = [
            tokenize_text(m.get("window_text") or f"{m['username']}: {m['content']}")
            for m in self.metadata
        ]
        self.bm25 = BM25Okapi(corpus)

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
        
        try:
            self.cloud_storage.upload_file("faiss_index.bin", "/tmp/faiss_index.bin")
            self.cloud_storage.upload_file("vector_metadata.json", "/tmp/vector_metadata.json")
            logger.info("[VECTOR] Hugging Face Backup complete.")
        except Exception as e:
            logger.error(f"[VECTOR] Hugging Face upload failed: {e}")

    def add_message(self, username: str, content: str, timestamp: str):
        """Embeds a 3-message XML sliding window, updates RAM, and schedules a background backup."""
        msg_hash = self._generate_hash(username, content)
        
        with self.db_lock:
            if msg_hash in self.seen_hashes:
                return

        with self.db_lock:
            buffer_copy = list(self.rolling_buffer)

        window_text = self._build_context_window(buffer_copy, username, content)
        
        embeddings_generator = self.embedding_model.embed([window_text])
        vector = next(embeddings_generator).astype(np.float32)
        
        with self.db_lock:
            self.index.add(np.array([vector]))
            self.metadata.append({
                "username": username,
                "content": content,
                "timestamp": timestamp,
                "window_text": window_text
            })
            self.seen_hashes.add(msg_hash)
            
            self.rolling_buffer.append({"username": username, "content": content})
            if len(self.rolling_buffer) > 2:
                self.rolling_buffer.pop(0)
                
            self._rebuild_bm25()
            
        self._schedule_sync()

    def add_batch_messages(self, messages: list):
        """Optimized batch ingestion with sliding window construction."""
        new_messages = []
        with self.db_lock:
            for msg in messages:
                msg_hash = self._generate_hash(msg["username"], msg["content"])
                if msg_hash not in self.seen_hashes:
                    new_messages.append(msg)
                    self.seen_hashes.add(msg_hash)

        if not new_messages:
            return 0

        windows_to_embed = []
        batch_metadata_entries = []
        
        batch_buffer = list(self.rolling_buffer)
        for msg in new_messages:
            win_text = self._build_context_window(batch_buffer, msg["username"], msg["content"])
            windows_to_embed.append(win_text)
            batch_metadata_entries.append({
                "username": msg["username"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
                "window_text": win_text
            })
            batch_buffer.append({"username": msg["username"], "content": msg["content"]})
            if len(batch_buffer) > 2:
                batch_buffer.pop(0)

        logger.info(f"[VECTOR] Embedding batch of {len(windows_to_embed)} historical context windows...")
        
        embeddings_generator = self.embedding_model.embed(windows_to_embed)
        vectors = [vec.astype(np.float32) for vec in embeddings_generator]
        
        with self.db_lock:
            self.index.add(np.array(vectors))
            self.metadata.extend(batch_metadata_entries)
            self.rolling_buffer = batch_buffer[-2:]
            self._rebuild_bm25()
                
        self.force_sync()
        return len(new_messages)

    def search_similar(self, query: str, top_k: int = 5, username: str = None):
        """Hybrid Retrieval via Reciprocal Rank Fusion (FAISS + BM25)."""
        if not self.metadata or self.index.ntotal == 0:
            return []

        # 1. FAISS Vector Search
        query_vector = self.embedding_model.embed([query]) 
        query_vector = np.array(list(query_vector), dtype=np.float32) 
        
        distances, faiss_indices = self.index.search(query_vector, top_k * 3)
        
        # 2. BM25 Keyword Search
        bm25_indices = []
        bm25_scores = []
        if self.bm25:
            tokenized_q = tokenize_text(query)
            if tokenized_q:
                bm25_scores = self.bm25.get_scores(tokenized_q)
                bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 3]

        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores = {}
        
        if len(faiss_indices) > 0:
            for rank, idx in enumerate(faiss_indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (60 + rank))
                    
        for rank, idx in enumerate(bm25_indices):
            if idx < len(self.metadata) and bm25_scores[idx] > 0:
                fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (60 + rank))

        sorted_candidates = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)
        
        results = []
        for idx in sorted_candidates:
            item = self.metadata[idx]
            if username and item.get("username", "").lower() != username.lower():
                continue
                
            results.append(item)
            if len(results) >= top_k:
                break
                
        return results

# Initialize the singleton immediately
vector_db = VectorMemory()
