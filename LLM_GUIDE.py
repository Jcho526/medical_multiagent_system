"""
大模型接入指南
===

本项目支持任何 OpenAI 兼容接口的大模型，统一通过 openai SDK 调用。
你只需要修改 .env 文件即可切换模型，无需改任何代码。

════════════════════════════════════════════════
方案一：OpenAI（最简单，质量最高）
════════════════════════════════════════════════

1. 获取 API Key: https://platform.openai.com/api-keys
2. 创建 .env 文件:

   LLM_PROVIDER=openai
   LLM_MODEL=gpt-4o-mini          # 推荐，便宜快速
   LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
   LLM_BASE_URL=                   # 留空即可
   EMBEDDING_PROVIDER=openai
   EMBEDDING_MODEL=text-embedding-3-small

3. 运行验证:
   python3 check_llm.py

4. 启动系统:
   python3 main.py


════════════════════════════════════════════════
方案二：DeepSeek（性价比极高，中文能力强）
════════════════════════════════════════════════

1. 获取 API Key: https://platform.deepseek.com/api_keys
2. 创建 .env 文件:

   LLM_PROVIDER=deepseek
   LLM_MODEL=deepseek-chat
   LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
   LLM_BASE_URL=https://api.deepseek.com   # 系统会自动设置
   EMBEDDING_PROVIDER=local                # DeepSeek无embedding接口，用本地
   EMBEDDING_MODEL=text-embedding-3-small

3. 快速验证:
   python3 check_llm.py --deepseek


════════════════════════════════════════════════
方案三：本地模型（Ollama，完全离线，免费）
════════════════════════════════════════════════

1. 安装 Ollama: https://ollama.com/download
2. 拉取模型:

   ollama pull qwen2.5:7b          # 推荐，中文能力好
   # 或
   ollama pull llama3.1:8b

3. 确认 Ollama 在运行:

   ollama serve                     # 启动服务
   ollama list                      # 查看已下载模型

4. 创建 .env 文件:

   LLM_PROVIDER=local
   LLM_MODEL=qwen2.5:7b
   LLM_API_KEY=not-needed
   LLM_BASE_URL=http://localhost:11434/v1
   EMBEDDING_PROVIDER=local
   EMBEDDING_MODEL=local

5. 快速验证:
   python3 check_llm.py --local


════════════════════════════════════════════════
方案四：国内API（硅基流动 / 通义 / 月之暗面）
════════════════════════════════════════════════

这些平台提供 OpenAI 兼容的 API，只需修改 base_url:

── 硅基流动 (SiliconFlow) ──
   LLM_PROVIDER=openai
   LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
   LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
   LLM_BASE_URL=https://api.siliconflow.cn/v1

── 通义千问 (DashScope) ──
   LLM_PROVIDER=openai
   LLM_MODEL=qwen-plus
   LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
   LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

── Moonshot (月之暗面) ──
   LLM_PROVIDER=openai
   LLM_MODEL=moonshot-v1-8k
   LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
   LLM_BASE_URL=https://api.moonshot.cn/v1


════════════════════════════════════════════════
架构说明：为什么只需改 .env？
════════════════════════════════════════════════

所有模型统一走 OpenAI SDK 的 Chat Completions 接口：

  ┌─────────┐    ┌──────────────┐    ┌──────────────┐
  │ .env配置 │───→│  config.py   │───→│  base.py     │
  │         │    │  读取环境变量  │    │  OpenAI(     │
  │         │    │  自动推断URL  │    │    api_key,  │
  │         │    │              │    │    base_url   │
  │         │    │              │    │  )           │
  └─────────┘    └──────────────┘    └──────────────┘
                                              │
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                         OpenAI API      DeepSeek API     Ollama API
                       (官方-default)   (base_url指定)  (base_url指定)

核心代码在 agents/base.py:
  self.client = OpenAI(api_key=..., base_url=...)
  response = self.client.chat.completions.create(model=..., messages=...)

这是 OpenAI SDK 的标准用法，任何兼容接口都能直接用。
"""

if __name__ == "__main__":
    print(__doc__)
