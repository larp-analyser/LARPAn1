import faiss
import numpy as np
from fastembed import TextEmbedding
import threading
import logging
import hashlib
from app.db.mongo import MongoDB

logger = logging.getLogger(__name__)

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
        # Load the 1.3GB ONNX model into RAM
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
        
        # BGE-Large outputs 1024-dimensional vectors
        self.dimension = 1024 
        self.index = faiss.IndexFlatIP(self.dimension) # Inner Product for Cosine Similarity
        
        self.metadata = [] 
        self.seen_hashes = set() # O(1) deduplication check
        self.db_lock = threading.Lock()

        # MongoDB Persistence Collection
        self.collection = MongoDB.get_collection("vector_receipts")
        
        logger.info("[VECTOR] Fetching historical receipts from MongoDB to rebuild FAISS index...")
        all_records = list(self.collection.find({}))
        
        if all_records:
            vectors = []
            for record in all_records:
                vectors.append(record["vector"])
                meta = {
                    "username": record["username"],
                    "content": record["content"],
                    "timestamp": record["timestamp"]
                }
                self.metadata.append(meta)
                self.seen_hashes.add(self._generate_hash(record["username"], record["content"]))
            
            # Instantly rebuild the FAISS map in RAM
            self.index.add(np.array(vectors).astype(np.float32))
            logger.info(f"[VECTOR] Successfully loaded {len(all_records)} receipts into RAM.")
        else:
            logger.info("[VECTOR] No historical receipts found. Starting fresh.")
            
        logger.info("[VECTOR] Engine Online.")

    def _generate_hash(self, username: str, content: str) -> str:
        """Generates a unique fingerprint for a message to prevent duplicate embedding."""
        return hashlib.md5(f"{username}::{content}".encode('utf-8')).hexdigest()

    def add_message(self, username: str, content: str, timestamp: str):
        """Embeds a single message, adds to FAISS, and backs up to Mongo."""
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
            
            # Background cloud save
            self.collection.insert_one({
                "username": username,
                "content": content,
                "timestamp": timestamp,
                "vector": vector.tolist()
            })

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
        
        mongo_docs = []
        with self.db_lock:
            self.index.add(np.array(vectors))
            for i, msg in enumerate(new_messages):
                self.metadata.append({
                    "username": msg["username"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"]
                })
                mongo_docs.append({
                    "username": msg["username"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"],
                    "vector": vectors[i].tolist()
                })
                
            if mongo_docs:
                self.collection.insert_many(mongo_docs)
                
        return len(new_messages)

    def search_similar(self, query: str, top_k: int = 3) -> list:
        """Finds semantic receipts matching the query."""
        with self.db_lock:
            if self.index.ntotal == 0:
                return []
                
        embeddings_generator = self.embedding_model.embed([query])
        query_vector = next(embeddings_generator).astype(np.float32)
        
        with self.db_lock:
            distances, indices = self.index.search(np.array([query_vector]), top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    if distances[0][i] > 0.55: 
                        results.append(self.metadata[idx])
            return results

# Initialize the singleton immediately so the model is ready when the app boots
vector_db = VectorMemory()
