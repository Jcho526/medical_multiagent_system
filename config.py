"""Global configuration for the Medical Rehab Agent System."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM Settings ──────────────────────────────────────────
    # Provider: openai | deepseek | local
    # 所有provider均走OpenAI兼容SDK，区别仅在 base_url
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # 自动设置 base_url：未显式配置时按 provider 推断
    _explicit_base_url = os.getenv("LLM_BASE_URL")
    if _explicit_base_url:
        LLM_BASE_URL = _explicit_base_url
    elif LLM_PROVIDER == "deepseek":
        LLM_BASE_URL = "https://api.deepseek.com"
    elif LLM_PROVIDER == "local":
        LLM_BASE_URL = "http://localhost:11434/v1"
    else:
        LLM_BASE_URL = None

    # ── RAG Settings ──────────────────────────────────────────
    # EMBEDDING_PROVIDER: openai | local | ollama
    #   ollama: 调用Ollama的embedding接口（需先pull一个embedding模型）
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", None)  # 默认跟随LLM_BASE_URL
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
    KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base.json")

    # ── Server ────────────────────────────────────────────────
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))


config = Config()
