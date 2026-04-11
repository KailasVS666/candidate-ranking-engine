"""
models/vector_store.py
-----------------------
ChromaDB manager for persistent vector storage and similarity search.
Optimised to share the SentenceTransformer model instance with the rest of the app.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import os

import chromadb
from chromadb.utils import embedding_functions

from config.settings import VECTOR_DB_PATH, SENTENCE_TRANSFORMER_MODEL
from models.semantic_scorer import _load_model
from utils.logger import get_logger

logger = get_logger(__name__)


class SharedModelEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    ChromaDB embedding function that uses the globally cached 
    SentenceTransformer model from semantic_scorer.py.
    """
    def __call__(self, input: List[str]) -> List[List[float]]:
        try:
            model = _load_model()
            if model is None:
                logger.warning("Embedding model unavailable, using zero vectors.")
                return [[0.0] * 384 for _ in input]
                
            embeddings = model.encode(input, normalize_embeddings=True)
            
            if embeddings is None:
                logger.error("Model.encode returned None for input of length %d", len(input))
                return [[0.0] * 384 for _ in input]
                
            # Handle both numpy array and list return types
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            return embeddings
        except Exception as e:
            logger.exception(f"Critical error in embedding function: {e}")
            return [[0.0] * 384 for _ in input]



class VectorStoreManager:
    """
    Manages a ChromaDB collection for resume embeddings.
    """

    def __init__(self, collection_name: str = "resumes"):
        """
        Initialize the persistent ChromaDB client and get/create a collection.
        """
        self.path = VECTOR_DB_PATH
        os.makedirs(self.path, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at {self.path}")
        # Using a client with a slightly longer timeout settings for Windows stability
        self.client = chromadb.PersistentClient(path=self.path)
        
        # We use our custom embedding function to share the model instance
        self.embed_fn = SharedModelEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_resumes(
        self, 
        texts: List[str], 
        filenames: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add or update resumes in the vector store.
        """
        if not texts:
            return

        # Use filename as the unique ID
        ids = filenames
        
        if metadatas is None:
            metadatas = [{"filename": f} for f in filenames]

        logger.info(f"Upserting {len(texts)} resumes to Vector Store...")
        try:
            self.collection.upsert(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("Vector Store sync complete.")
        except Exception as e:
            logger.error(f"Error adding to Vector Store: {e}")

    def delete_resume(self, filename: str):
        """Remove a resume from the vector store by ID."""
        try:
            self.collection.delete(ids=[filename])
            logger.info(f"Deleted {filename} from Vector Store.")
        except Exception as e:
            logger.warning(f"Could not delete {filename} from Vector Store: {e}")

    def clear(self):
        """Clear the entire collection."""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector Store cleared.")
        except Exception as e:
            logger.error(f"Failed to clear Vector Store: {e}")

    def query_similar(self, query_text: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Search for the most semantically similar resumes to the query text.
        """
        logger.info(f"Querying Vector Store for top {n_results} matches...")
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
