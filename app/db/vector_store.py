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
        self.rolling_buffers = {}  # FIXED: Maps group_name to its own buffer
        self.db_lock = threading.Lock()
        self.upload_lock = threading.Lock()
        self.bm25 = None
        
        self.cloud_storage = HFPersistence()
        self._sync_timer = None

        logger.info("[VECTOR] Fetching FAISS index and metadata from Hugging Face...")
        
        index_loaded = False
        
        # 1. Properly download and load the FAISS binary
        if self.cloud_storage.download_file("faiss_index.bin", "/tmp/faiss_index.bin"):
            try:
                self.index = faiss.read_index("/tmp/faiss_index.bin")
                index_loaded = True
            except Exception as e:
                logger.error(f"[VECTOR] Failed to read FAISS index from disk: {e}")
                
        # 2. Properly download and load the JSON metadata
        if self.cloud_storage.download_file("vector_metadata.json", "/tmp/vector_metadata.json"):
            try:
                with open("/tmp/vector_metadata.json", "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                for m in self.metadata:
                    # Generate hashes
                    grp = m.get("group_name", "unknown")
                    ts = m.get("timestamp", "")
                    self.seen_hashes.add(self._generate_hash(grp, m["username"], m["content"], ts))
                    
                    # Rebuild sliding windows
                    if grp not in self.rolling_buffers:
                        self.rolling_buffers[grp] = []
                    self.rolling_buffers[grp].append({"username": m["username"], "content": m["content"]})
                    if len(self.rolling_buffers[grp]) > 2:
                        self.rolling_buffers[grp].pop(0)

            except Exception as e:
                logger.error(f"[VECTOR] Failed to read metadata JSON from disk: {e}")

        if not index_loaded:
            logger.info("[VECTOR] No historical index found on HF. Starting fresh.")
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            logger.info(f"[VECTOR] Successfully loaded {len(self.metadata)} receipts into RAM from HF.")
            
        self._rebuild_bm25()
        logger.info("[VECTOR] Engine Online with BM25 + FAISS Hybrid RRF Search.")

    def _generate_hash(self, group_name: str, username: str, content: str, timestamp: str) -> str:
        """Generates a unique fingerprint using temporal and channel metadata."""
        unique_string = f"{group_name}::{username}::{content}::{timestamp}"
        return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

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
        if not self.upload_lock.acquire(blocking=False):
            logger.info("[VECTOR] Backup already in progress. Skipping redundant sync.")
            return

        try:
            logger.info("[VECTOR] Compiling binary memory payload...")
            with self.db_lock:
                faiss.write_index(self.index, "/tmp/faiss_index.bin")
                with open("/tmp/vector_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(self.metadata, f)
            
            self.cloud_storage.upload_file("faiss_index.bin", "/tmp/faiss_index.bin")
            self.cloud_storage.upload_file("vector_metadata.json", "/tmp/vector_metadata.json")
            logger.info("[VECTOR] Hugging Face Backup complete.")
        except Exception as e:
            logger.error(f"[VECTOR] Hugging Face upload failed: {e}")
        finally:
            self.upload_lock.release()

    def add_message(self, group_name: str, username: str, content: str, timestamp: str):
        msg_hash = self._generate_hash(group_name, username, content, timestamp)
        
        with self.db_lock:
            if msg_hash in self.seen_hashes:
                return

        with self.db_lock:
            if group_name not in self.rolling_buffers:
                self.rolling_buffers[group_name] = []
            buffer_copy = list(self.rolling_buffers[group_name])

        window_text = self._build_context_window(buffer_copy, username, content)
        
        embeddings_generator = self.embedding_model.embed([window_text])
        vector = next(embeddings_generator).astype(np.float32)
        
        with self.db_lock:
            self.index.add(np.array([vector]))
            self.metadata.append({
                "group_name": group_name, # FIXED: Save group_name to metadata
                "username": username,
                "content": content,
                "timestamp": timestamp,
                "window_text": window_text
            })
            self.seen_hashes.add(msg_hash)
            
            self.rolling_buffers[group_name].append({"username": username, "content": content})
            if len(self.rolling_buffers[group_name]) > 2:
                self.rolling_buffers[group_name].pop(0)
                
            self._rebuild_bm25()
            
        self._schedule_sync()

    def add_batch_messages(self, group_name: str, messages: list, skip_sync: bool = False):
        new_messages = []
        with self.db_lock:
            for msg in messages:
                ts = msg.get("timestamp", "")
                msg_hash = self._generate_hash(group_name, msg["username"], msg["content"], ts)
                if msg_hash not in self.seen_hashes:
                    new_messages.append(msg)
                    self.seen_hashes.add(msg_hash)

        if not new_messages:
            return 0

        with self.db_lock:
            if group_name not in self.rolling_buffers:
                self.rolling_buffers[group_name] = []
            batch_buffer = list(self.rolling_buffers[group_name])

        windows_to_embed = []
        batch_metadata_entries = []
        
        for msg in new_messages:
            win_text = self._build_context_window(batch_buffer, msg["username"], msg["content"])
            windows_to_embed.append(win_text)
            batch_metadata_entries.append({
                "group_name": group_name, # FIXED: Save group_name to metadata
                "username": msg["username"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
                "window_text": win_text
            })
            batch_buffer.append({"username": msg["username"], "content": msg["content"]})
            if len(batch_buffer) > 2:
                batch_buffer.pop(0)

        logger.info(f"[VECTOR] Embedding batch of {len(windows_to_embed)} historical context windows for {group_name}...")
        
        embeddings_generator = self.embedding_model.embed(windows_to_embed)
        vectors = [vec.astype(np.float32) for vec in embeddings_generator]
        
        with self.db_lock:
            self.index.add(np.array(vectors))
            self.metadata.extend(batch_metadata_entries)
            self.rolling_buffers[group_name] = batch_buffer[-2:]
            self._rebuild_bm25()
                
        if not skip_sync:
            self.force_sync()
            
        return len(new_messages)

    def search_similar(self, query: str, top_k: int = 5, username: str = None, group_name: str = None):
        """Hybrid Retrieval via Reciprocal Rank Fusion (FAISS + BM25)."""
        if not self.metadata or self.index.ntotal == 0:
            return []

        # 1. FAISS Vector Search
        query_vector = self.embedding_model.embed([query]) 
        query_vector = np.array(list(query_vector), dtype=np.float32) 
        
        with self.db_lock:
            if self.index.ntotal == 0:
                return []
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
            if group_name and item.get("group_name", "") != group_name:
                continue
                
            results.append(item)
            if len(results) >= top_k:
                break
                
        return results

# Initialize the singleton immediately
vector_db = VectorMemory()
