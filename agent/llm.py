"""
agent.llm
==========
LLM 客户端工厂:ChatOpenAI 兼容 AGNES / DeepSeek / 通义 / OpenAI。

为什么用 ChatOpenAI 而不是 ChatAgnes / ChatDeepSeek?
- LangChain 官方 ChatOpenAI 是 OpenAI 协议的事实标准
- AGNES / DeepSeek 都声明 OpenAI 兼容（base_url 一换就能用）
- 一份代码 = 多家模型，方便切换 provider

新写法：用 ChatOpenAI.with_structured_output() 替代 PydanticOutputParser。
- 旧式:prompt | llm | PydanticOutputParser(Schema),prompt 里要 partial format_instructions,
  parser 在链尾硬解析 JSON。reasoning 模型容易炸 → 需要 RobustPydanticOutputParser。
- 新式:prompt | llm.with_structured_output(Schema),provider 在生成时按 schema 约束
  (走原生 function_calling 或 json_schema),不需 format_instructions 占位,reasoning
  模型的处理也更干净。
- 缺点:依赖 provider 原生结构化输出能力;不支持时需退回旧写法。

模型:agnes-2.5-flash(AGNES 平台免费 + Agent 优化,中文强)。
"""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from agent.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT
from agent.models import IntentClassification


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
        # max_retries 默认 6 会很慢；意图分类失败一般 retry 没用
        max_retries=1,
    )


@lru_cache(maxsize=1)
def get_structured_llm() -> Runnable:
    """
    获取绑定了结构化输出 schema 的 LLM 客户端（单例 — lru_cache 缓存）。

    用途:替换旧的 `chain = prompt | llm | PydanticOutputParser(Schema)`。
    新写法:`chain = prompt | get_structured_llm()`,ainvoke() 直接返回 IntentClassification 实例,
    不再需要单独的 parser,也不再需要 prompt.partial(format_instructions=...) 占位。

    实现要点:
    - ChatOpenAI.with_structured_output(schema) 底层走 provider 的原生结构化输出
      (OpenAI 协议下优先 function_calling,json_schema 兜底)
    - 返回的 Runnable[LanguageModelInput, BaseModel] — ainvoke() 出 Pydantic 实例
    - 异常由 provider 抛(网络/schema 校验/解析),下游 try/except 统一降级
    """
    return get_llm().with_structured_output(IntentClassification)


def reset_llm_cache() -> None:
    """测试时清掉 lru_cache，重新走 get_llm()"""
    get_llm.cache_clear()
    get_structured_llm.cache_clear()