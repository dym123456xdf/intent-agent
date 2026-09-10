"""
agent.result_extract
====================
从 create_agent 的输出里抽取 UI 层关心的字段。

create_agent.invoke() 返回:
  {
    "messages": [BaseMessage, ...],  # 含 AIMessage.tool_calls / HumanMessage 等
    "structured_response": IntentClassification | None,
  }

我们要把这个结构扁平化成 UI 友好的 dict:
  {
    "current_intent": str,
    "intent_confidence": float,
    "intent_reasoning": str,
    "extracted_slots": dict[str, str],
    "needs_clarification": bool,
    "clarification_question": str | None,
    "routed_node": str,           # 最后调用的业务 tool 名(跳过 IntentClassification 伪 tool)
    "fallback_reason": str | None,
  }

抽取规则:
- routed_node: 反向遍历 messages,跳过 IntentClassification 伪 tool,取最后一个真实业务 tool
- fallback_reason: routed_node == fallback_handler 时,从它的 args.reason 字段拿
"""

from typing import Any

from agent.models import IntentClassification


def _extract_routed_node(messages: list) -> str:
    """
    从 agent 的 messages 里提取 routed_node。

    create_agent 内部 loop 会调一系列 tool(包括一个隐藏的 IntentClassification 伪 tool,
    它用于触发 response_format 校验)。"真实"路由 handler 是 IntentClassification 之前的
    最后一个业务 tool。
    """
    # 从后往前找,跳过 IntentClassification 这个伪 tool
    for m in reversed(messages):
        if not hasattr(m, "tool_calls") or not m.tool_calls:
            continue
        for tc in reversed(m.tool_calls):
            name = tc.get("name", "")
            if name != "IntentClassification":
                return name
    return ""


def _extract_fallback_reason(messages: list, routed_node: str) -> str | None:
    """如果 routed_node 是 fallback_handler,从最后一次调用它的 args 里拿 reason"""
    if routed_node != "fallback_handler":
        return None
    for m in reversed(messages):
        if not hasattr(m, "tool_calls") or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            if tc.get("name") == "fallback_handler":
                return tc.get("args", {}).get("reason", "unknown")
    return None


def flatten_agent_result(result: dict) -> dict[str, Any]:
    """
    把 create_agent 的返回值扁平化为 UI 视图。

    Args:
        result: create_agent.ainvoke() 的返回值({"messages", "structured_response"})

    Returns:
        dict with keys: current_intent / intent_confidence / intent_reasoning /
        extracted_slots / needs_clarification / clarification_question /
        routed_node / fallback_reason
    """
    sr = result.get("structured_response")
    sr_dict = sr.model_dump() if isinstance(sr, IntentClassification) else {}

    # slots 在 Pydantic schema 里是 list[IntentSlot],dump 出来还是 list
    # UI 层想要 dict[name, value]
    slots_raw = sr_dict.get("slots", [])
    slots_dict = {s["name"]: s["value"] for s in slots_raw if isinstance(s, dict)}

    messages = result.get("messages", [])
    routed_node = _extract_routed_node(messages)
    fallback_reason = _extract_fallback_reason(messages, routed_node)

    return {
        "current_intent": sr_dict.get("intent", "N/A"),
        "intent_confidence": sr_dict.get("confidence", 0.0),
        "intent_reasoning": sr_dict.get("reasoning", ""),
        "extracted_slots": slots_dict,
        "needs_clarification": sr_dict.get("needs_clarification", False),
        "clarification_question": sr_dict.get("clarification_question"),
        "routed_node": routed_node,
        "fallback_reason": fallback_reason,
    }