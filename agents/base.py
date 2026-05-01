"""Base Agent class — all agents inherit from this."""

from typing import Any
from config import config


class BaseAgent:
    """A thin wrapper around LLM calls with a designed system prompt."""

    name: str = "BaseAgent"
    system_prompt: str = ""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy-init OpenAI-compatible client.

        支持: OpenAI / DeepSeek / Ollama / vLLM / 任何OpenAI兼容接口
        所有provider统一走 openai SDK，仅 base_url 不同。
        """
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": config.LLM_API_KEY or "not-needed"}
            if config.LLM_BASE_URL:
                kwargs["base_url"] = config.LLM_BASE_URL
            self._client = OpenAI(**kwargs)
        return self._client

    def run(self, **kwargs) -> Any:
        """Execute the agent. Subclasses override this."""
        raise NotImplementedError

    def _call_llm(self, user_prompt: str) -> str:
        """Single-turn LLM call with system + user prompt."""
        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    @classmethod
    def test_connection(cls) -> dict:
        """Verify LLM connectivity. Returns {ok, model, provider, error?}."""
        try:
            agent = cls()
            resp = agent.client.chat.completions.create(
                model=config.LLM_MODEL,
                temperature=0,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return {
                "ok": True,
                "provider": config.LLM_PROVIDER,
                "model": config.LLM_MODEL,
                "base_url": config.LLM_BASE_URL or "https://api.openai.com/v1",
                "response": resp.choices[0].message.content,
            }
        except Exception as e:
            return {
                "ok": False,
                "provider": config.LLM_PROVIDER,
                "model": config.LLM_MODEL,
                "base_url": config.LLM_BASE_URL or "https://api.openai.com/v1",
                "error": str(e),
            }
