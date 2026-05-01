"""
LLM 连接验证脚本 —— 帮你快速确认大模型是否接入成功

用法:
  python3 check_llm.py              # 默认读 .env
  python3 check_llm.py --deepseek   # 快速切换 DeepSeek 测试
  python3 check_llm.py --local      # 快速切换本地 Ollama 测试
"""

import sys
import os


def check_llm():
    # Allow quick provider override via CLI
    if "--deepseek" in sys.argv:
        os.environ["LLM_PROVIDER"] = "deepseek"
        os.environ["LLM_MODEL"] = "deepseek-chat"
        os.environ["LLM_BASE_URL"] = "https://api.deepseek.com"
    elif "--local" in sys.argv:
        os.environ["LLM_PROVIDER"] = "local"
        os.environ["LLM_MODEL"] = "qwen2.5:7b"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434/v1"

    # Re-import to pick up env overrides
    for mod in list(sys.modules):
        if mod.startswith("config") or mod.startswith("rag") or mod.startswith("agents"):
            del sys.modules[mod]

    from config import config
    from agents.base import BaseAgent

    print("=" * 55)
    print("  LLM & Embedding 连接验证")
    print("=" * 55)
    print(f"  LLM Provider : {config.LLM_PROVIDER}")
    print(f"  LLM Model    : {config.LLM_MODEL}")
    print(f"  LLM Base URL : {config.LLM_BASE_URL or 'https://api.openai.com/v1'}")
    print(f"  Emb Provider : {config.EMBEDDING_PROVIDER}")
    print(f"  Emb Model    : {config.EMBEDDING_MODEL}")
    print()

    # ── Test LLM ────────────────────────────────────────────
    print("[1/2] 测试 LLM 连接...")
    result = BaseAgent.test_connection()
    if result["ok"]:
        print(f"      ✓ 连接成功！模型回复: {result['response']!r}")
    else:
        print(f"      ✗ 连接失败: {result['error']}")
        print()
        print("  排查建议:")
        err = result["error"].lower()
        if "auth" in err or "api_key" in err or "401" in err:
            print("    → API Key 不正确，检查 .env 中的 LLM_API_KEY")
        elif "connection" in err or "refused" in err:
            print("    → 无法连接服务器，检查 LLM_BASE_URL 和网络")
            if config.LLM_PROVIDER == "local":
                print("    → 确认 Ollama 已启动: ollama serve")
        elif "model" in err or "404" in err:
            print("    → 模型名称不正确，检查 LLM_MODEL")
            if config.LLM_PROVIDER == "local":
                print("    → 可用模型: 运行 ollama list 查看")
        else:
            print(f"    → 未知错误: {result['error']}")

    # ── Test Embedding ──────────────────────────────────────
    print()
    print(f"[2/2] 测试 Embedding (provider={config.EMBEDDING_PROVIDER})...")
    try:
        from rag.embedding import get_embedding_provider
        provider = get_embedding_provider()
        vecs = provider.embed(["测试文本"])
        dim = len(vecs[0])
        print(f"      ✓ Embedding成功，维度: {dim}，模型: {config.EMBEDDING_MODEL}")
    except Exception as e:
        print(f"      ✗ Embedding失败: {e}")
        if config.EMBEDDING_PROVIDER == "ollama":
            print(f"    → 先拉取embedding模型: ollama pull {config.EMBEDDING_MODEL}")
            print(f"    → 确认Ollama在运行: ollama list")
        else:
            print("    → 降级方案: 在 .env 中设置 EMBEDDING_PROVIDER=local")

    print()
    print("=" * 55)
    if result["ok"]:
        print("  ✓ 大模型已接入，可以运行: python3 main.py")
    else:
        print("  ✗ 请修复上述问题后重试")
    print("=" * 55)


if __name__ == "__main__":
    check_llm()
