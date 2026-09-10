"""
agent.state
============
LangGraph StateGraph 的 state schema。

主流写法（LangGraph 1.x 官方推荐）:
- 继承 langgraph.graph.MessagesState —— 它已经声明了 `messages: Annotated[list, add_messages]`
  字段（带 add_messages reducer 的 messages 列表），我们只需要叠加业务字段即可
- 不需要再手动 import add_messages / 写 Annotated[list, add_messages]

字段说明:
- messages: 用户-客服的多轮对话（继承自 MessagesState，已带 add_messages reducer;
  LangGraph 自带的基于 id 去重 + 新消息覆盖旧消息的合并器，多轮场景必需）
- current_intent: 本轮分类出的意图（每轮 overwrite）
- intent_confidence: 当前意图的置信度
- intent_reasoning: 分类理由（便于 trace）
- extracted_slots: 抽取出的槽位（dict[str, str]）
- needs_clarification: 是否需要追问
- clarification_question: 追问的问题
- routed_node: 经 confidence 校验 + 路由函数后，下一步要去的目标节点名
- fallback_reason: 触发 fallback 的原因（None / "low_confidence" / "unknown_intent"）
"""

from typing import NotRequired

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """
    业务 state 继承 LangGraph 官方的 MessagesState:
    - 自动获得 messages 字段（Annotated[list[AnyMessage], add_messages]）
    - 少 4 行 boilerplate（不用 import add_messages / 不用 Annotated[...]）
    - 与 LangSmith / LangGraph 官方示例对齐
    """
    # —— 意图分类节点写入 ——
    # 8 类意图之一
    current_intent: NotRequired[str]
    # 置信度（0~1）
    intent_confidence: NotRequired[float]
    # 一句话分类理由（trace 调试）
    intent_reasoning: NotRequired[str]
    # 抽取出的槽位 { "order_id": "123456", "product_id": "P001" }
    extracted_slots: NotRequired[dict[str, str]]
    # 是否需要追问
    needs_clarification: NotRequired[bool]
    # 追问用户的问题
    clarification_question: NotRequired[str]

    # —— 路由节点写入 ——
    # 路由决策：下一步要去的目标节点名（"order_query_handler" / "fallback_handler" 等）
    routed_node: NotRequired[str]
    # 触发 fallback 的原因（None / "low_confidence" / "unknown_intent"）
    fallback_reason: NotRequired[str | None]