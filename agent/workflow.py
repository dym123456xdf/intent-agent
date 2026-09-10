"""
agent.workflow
==============
v2 改造:用 LangChain 1.0 的 create_agent 替代手写的 StateGraph。

create_agent 是 LangChain 1.0 构建智能体的标准方式（替代 langgraph.prebuilt.create_react_agent）。
它底层基于 LangGraph runtime 跑一个固定的 ReAct loop：
  1. 把 messages 喂给 model
  2. model 决定调一个 tool
  3. tool 返回结果
  4. 把结果塞回 messages
  5. 重复直到 model 不再调 tool（最后一次输出按 response_format schema 校验）

对意图识别场景的适配:
- model = AGNES (ChatOpenAI 兼容)
- tools = 10 个 handler（业务 stub + clarification + fallback）
- system_prompt = 意图清单 + 分类原则
- response_format = IntentClassification（最后输出按此 schema 校验）
"""

from functools import lru_cache

from langchain.agents import create_agent

from agent.llm import get_llm
from agent.models import IntentClassification
from agent.prompts import SYSTEM_PROMPT
from agent.tools import ALL_TOOLS


@lru_cache(maxsize=1)
def get_agent():
    """
    获取 create_agent 构建的 agent（单例 — lru_cache 缓存）。

    create_agent 每次调用都会重新构造一个 LangGraph 图，单例避免重复开销。
    """
    return create_agent(
        model=get_llm(),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=IntentClassification,
    )