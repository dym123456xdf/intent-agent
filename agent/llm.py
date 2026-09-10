"""
agent.llm
==========
LLM 客户端工厂 — v2 改造（create_agent 范式）。

create_agent 内部会自己把 model.with_structured_output(schema) 接上 response_format，
所以这里只暴露一个 ChatOpenAI 实例，不再包装 structured_output — 减少一层冗余。
"""

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
            "AGNES_API_KEY 未设置。请先 export AGNES_API_KEY=sk-xxx "
            "或把 export 加到 ~/.zshrc（或写入 .env）。"
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE,  # 意图分类要稳定 → 0
        timeout=LLM_TIMEOUT,
        max_retries=1,
    )


def reset_llm_cache() -> None:
    """测试时清掉 lru_cache，重新走 get_llm()"""
    get_llm.cache_clear()