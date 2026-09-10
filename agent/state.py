"""
agent.state
============
v2 改造（create_agent 范式）下，state 不再由我们自己维护 — create_agent 内部用 LangGraph 跑
一个固定的 ReAct loop（model ↔ tools），state schema 是 LangChain 内置的 AgentState。

本文件保留一份 AgentResult TypedDict，用于在 main.py / tests 里接收 agent 输出：
result["messages"]: list[BaseMessage] — 全程对话历史（含 tool calls）
result["structured_response"]: IntentClassification — 最后一次按 schema 校验的结构化输出

字段说明：
- routed_node: 从 agent 的 tool calls 推断出来的最终去向 handler 名（主入口用）
- fallback_reason: 如果 routed_node 是 fallback_handler，触发原因
- intent / confidence / slots ... : 从 structured_response 里平铺出来，UI 层直接读
"""

from typing import NotRequired, TypedDict


class AgentResult(TypedDict):
    """create_agent.invoke() 返回值的扁平化视图（main.py 用）"""
    # agent 的原始输出
    messages: list  # 全程对话历史
    structured_response: NotRequired[dict]  # IntentClassification 转 dict 后的视图（可能缺失）

    # 推断字段（main.py 从 tool_calls 提取）
    routed_node: NotRequired[str]  # 最后调用的 tool 名
    fallback_reason: NotRequired[str | None]

    # 平铺字段（main.py 从 structured_response 提取）
    current_intent: NotRequired[str]
    intent_confidence: NotRequired[float]
    intent_reasoning: NotRequired[str]
    extracted_slots: NotRequired[dict[str, str]]
    needs_clarification: NotRequired[bool]
    clarification_question: NotRequired[str | None]