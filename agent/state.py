"""
agent.state
============
LangGraph StateGraph 的 TypedDict state schema。

HARD RULE（来自 langgraph-state-and-reducers skill）：
- node 函数 return 的每个 key 都必须在这里声明
- 否则 LangGraph 1.2 会静默丢弃，导致下游节点读不到、路由永远走 fallback

字段说明：
- messages: 用户-客服的多轮对话（Annotated[list, add_messages] 用 LangGraph 自带的
  基于 id 去重的合并器；多轮场景必需）
- current_intent: 本轮分类出的意图（每轮 overwrite）
- intent_confidence: 当前意图的置信度
- intent_reasoning: 分类理由（便于 trace）
- extracted_slots: 抽取出的槽位（dict[str, str]）
- needs_clarification: 是否需要追问
- clarification_question: 追问的问题
- routed_node: 经 confidence 校验 + 路由函数后，下一步要去的目标节点名
- fallback_reason: 触发 fallback 的原因（None / "low_confidence" / "unknown_intent"）
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # —— 输入 ——
    # 多轮对话历史（add_messages reducer：按 id 去重 + 新消息覆盖旧消息）
    messages: Annotated[list, add_messages]

    # —— 意图分类节点写入 ——
    # 8 类意图之一
    current_intent: str
    # 置信度（0~1）
    intent_confidence: float
    # 一句话分类理由（trace 调试）
    intent_reasoning: str
    # 抽取出的槽位 { "order_id": "123456", "product_id": "P001" }
    extracted_slots: dict[str, str]
    # 是否需要追问
    needs_clarification: bool
    # 追问用户的问题
    clarification_question: str

    # —— 路由节点写入 ——
    # 路由决策：下一步要去的目标节点名（"order_query_handler" / "fallback_handler" 等）
    routed_node: str
    # 触发 fallback 的原因（None / "low_confidence" / "unknown_intent"）
    fallback_reason: str | None