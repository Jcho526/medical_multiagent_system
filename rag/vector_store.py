"""FAISS-based vector store for knowledge retrieval."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from config import config
from rag.embedding import EmbeddingProvider, get_embedding_provider


class FAISSVectorStore:
    """Minimal FAISS wrapper — supports add, search, persist."""

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.provider = embedding_provider
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._index = None

    # ── Build ────────────────────────────────────────────────────

    def build_from_knowledge_base(self, path: Optional[str] = None):
        """Load knowledge base JSON, embed each entry, build FAISS index."""
        path = path or config.KNOWLEDGE_BASE_PATH
        with open(path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # Create a searchable text for each document
        texts = [self._doc_to_text(doc) for doc in self.documents]
        vecs = self.provider.embed(texts)
        self.embeddings = np.array(vecs, dtype="float32")

        self._build_index()

    def _doc_to_text(self, doc: Dict) -> str:
        """Flatten a knowledge entry into a single searchable string."""
        parts = [
            doc.get("disease", ""),
            doc.get("category", ""),
            " ".join(doc.get("symptoms", [])),
            doc.get("description", ""),
            " ".join(doc.get("rehab_methods", [])),
        ]
        return " ".join(parts)

    def _build_index(self):
        """Build a FAISS L2 index from current embeddings."""
        try:
            import faiss
        except ImportError:
            # Fallback to numpy brute-force if faiss not installed
            self._index = None
            return
        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(self.embeddings)
        self._index = index

    # ── Search ───────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return the top-k most similar documents for a query."""
        if self.embeddings is None:
            raise RuntimeError("Store is empty — call build_from_knowledge_base first.")

        q_vec = np.array(self.provider.embed([query]), dtype="float32")
        top_k = min(top_k, len(self.documents))

        if self._index is not None:
            return self._search_faiss(q_vec, top_k)
        return self._search_numpy(q_vec, top_k)

    def _search_faiss(self, q_vec: np.ndarray, top_k: int) -> List[Dict]:
        import faiss
        distances, indices = self._index.search(q_vec, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def _search_numpy(self, q_vec: np.ndarray, top_k: int) -> List[Dict]:
        """Brute-force cosine similarity fallback."""
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = self.embeddings / norms
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        sims = (normed @ q_norm.T).squeeze()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]
