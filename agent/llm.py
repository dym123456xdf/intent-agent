"""
agent.llm
==========
LLM 客户端工厂：ChatOpenAI 兼容 MiniMax / DeepSeek / 通义 / OpenAI。

为什么用 ChatOpenAI 而不是 ChatMiniMax / ChatDeepSeek？
- LangChain 官方 ChatOpenAI 是 OpenAI 协议的事实标准
- MiniMax / DeepSeek 都声明 OpenAI 兼容（base_url 一换就能用）
- 一份代码 = 多家模型，方便切换 provider
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agent.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    获取 LLM 客户端（单例 — lru_cache 缓存）。

    为什么单例？
    - 多次 invoke 不应该每次都新建连接
    - 测试时可以 monkeypatch get_llm 替换 mock
    """
    if not LLM_API_KEY:
        raise RuntimeError(
            "MINIMAX_API_KEY 未设置。请先 export MINIMAX_API_KEY=sk-xxx "
            "或把 export 加到 ~/.zshrc。"
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE,  # 意图分类要稳定 → 0
        timeout=LLM_TIMEOUT,
        # max_retries 默认 6 会很慢；意图分类失败一般 retry 没用
        max_retries=1,
    )


def reset_llm_cache() -> None:
    """测试时清掉 lru_cache，重新走 get_llm()"""
    get_llm.cache_clear()