"""RAG Retriever — ties embedding + vector store together."""

from typing import List, Dict, Any, Optional
from rag.embedding import get_embedding_provider
from rag.vector_store import FAISSVectorStore
from config import config


class RAGRetriever:
    """High-level retrieval interface used by Agents."""

    def __init__(self):
        provider = get_embedding_provider()
        self.store = FAISSVectorStore(provider)

    def initialize(self, kb_path: Optional[str] = None):
        """Load knowledge base and build index. Call once at startup."""
        self.store.build_from_knowledge_base(kb_path)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return top-K relevant knowledge entries for the query."""
        top_k = top_k or config.RAG_TOP_K
        return self.store.search(query, top_k)

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved docs into a text block for LLM context."""
        parts = []
        for i, doc in enumerate(results, 1):
            parts.append(
                f"【知识{i}】疾病: {doc.get('disease', 'N/A')}\n"
                f"  类别: {doc.get('category', 'N/A')}\n"
                f"  症状: {', '.join(doc.get('symptoms', []))}\n"
                f"  描述: {doc.get('description', 'N/A')}\n"
                f"  康复方法: {'; '.join(doc.get('rehab_methods', []))}\n"
                f"  严重程度范围: {doc.get('severity_range', 'N/A')}"
            )
        return "\n\n".join(parts)
