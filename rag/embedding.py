"""Embedding wrapper — abstracts OpenAI / Ollama / local embedding providers."""

from typing import List
import numpy as np
from config import config


class EmbeddingProvider:
    """Interface for text embedding generation."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI-compatible embedding (works with OpenAI / DeepSeek / SiliconFlow etc.)."""

    def __init__(self):
        from openai import OpenAI
        kwargs = {"api_key": config.LLM_API_KEY}
        base = config.EMBEDDING_BASE_URL or config.LLM_BASE_URL
        if base:
            kwargs["base_url"] = base
        self.client = OpenAI(**kwargs)
        self.model = config.EMBEDDING_MODEL

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in resp.data]


class OllamaEmbedding(EmbeddingProvider):
    """Ollama native embedding — calls GET /api/embeddings.

    用法: EMBEDDING_PROVIDER=ollama EMBEDDING_MODEL=nomic-embed-text
    先拉模型: ollama pull nomic-embed-text
    """

    def __init__(self):
        self.base_url = (config.EMBEDDING_BASE_URL or "http://localhost:11434").rstrip("/")
        # 去掉 /v1 后缀（Ollama原生接口不走 /v1）
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = config.EMBEDDING_MODEL

    def embed(self, texts: List[str]) -> List[List[float]]:
        import json
        import urllib.request

        results = []
        for text in texts:
            payload = json.dumps({"model": self.model, "prompt": text}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            results.append(data["embedding"])
        return results


class LocalEmbedding(EmbeddingProvider):
    """Character-ngram + keyword embedding for Chinese text (no API needed).

    Uses character bigrams for Chinese and word-level for English.
    Production: replace with a real embedding model for better retrieval quality.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.vocab: dict = {}
        self._fitted = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Split into: CJK bigrams + ASCII words."""
        tokens = []
        cjk_chars = [c for c in text if '一' <= c <= '鿿']
        for i in range(len(cjk_chars) - 1):
            tokens.append(cjk_chars[i] + cjk_chars[i + 1])
        tokens.extend(cjk_chars)
        for word in text.split():
            if not all('一' <= c <= '鿿' for c in word):
                tokens.append(word.lower())
        return tokens

    def _build_vocab(self, texts: List[str]):
        all_tokens = set()
        for t in texts:
            all_tokens.update(self._tokenize(t))
        self.vocab = {w: i for i, w in enumerate(sorted(all_tokens))}

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self._build_vocab(texts)
            self._fitted = True
        vectors = []
        for t in texts:
            vec = np.zeros(self.dim)
            for tok in self._tokenize(t):
                if tok in self.vocab:
                    idx = self.vocab[tok] % self.dim
                    vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors


def get_embedding_provider() -> EmbeddingProvider:
    provider = config.EMBEDDING_PROVIDER.lower()

    if provider == "ollama":
        return OllamaEmbedding()

    if provider == "openai" and config.LLM_API_KEY:
        try:
            return OpenAIEmbedding()
        except ImportError:
            pass

    return LocalEmbedding()
