"""
agent.nodes
============
LangGraph 节点函数。

每个节点 = 一个纯函数：state in → dict out。
LangGraph 自动把返回的 dict 合并到 state（按 reducer 规则）。

节点清单（本版本只做意图识别，下游 handler 留 hook）:
- classify_intent: 调用 LLM 做意图分类，写入 current_intent / confidence / slots ...
- intent_router: 纯逻辑路由（不调 LLM），根据 confidence + intent 决定 routed_node

v2 改动:升级到 with_structured_output(Schema)。
- chain = prompt | structured_llm（不再 pipe parser）
- ainvoke() 直接返回 IntentClassification 实例，不再需要 RobustPydanticOutputParser 剥 think 块
- provider 在生成时按 schema 约束（function_calling / json_schema）
"""

import logging

from agent.config import INTENT_CONFIDENCE_THRESHOLD
from agent.llm import get_structured_llm
from agent.prompts import get_intent_classify_prompt
from agent.state import AgentState

logger = logging.getLogger(__name__)


# ============================================================
# 节点 1: classify_intent
# ============================================================
async def classify_intent(state: AgentState) -> dict:
    """
    意图分类节点。

    输入: messages (对话历史 + 当前用户消息)
    输出: dict → 合并到 AgentState，包含 current_intent / confidence / reasoning /
         extracted_slots / needs_clarification / clarification_question

    实现要点:
    1. 从 state["messages"] 拿对话历史（add_messages reducer 已经合并好）
    2. chain = prompt | structured_llm 一行链式调用
    3. 用 .ainvoke() 异步调用 → 直接拿 IntentClassification 实例
    4. 把 Pydantic 对象转成 dict，匹配 AgentState schema
    5. LLM 解析失败时不要直接 raise — 写一个 FALLBACK_UNKNOWN 意图进 state，
       路由节点会再处理（这是生产级鲁棒性）
    """
    messages = state.get("messages", [])
    if not messages:
        # 没有用户消息 → 直接走 fallback（防御性编程）
        logger.warning("classify_intent 收到空 messages，跳过 LLM 调用")
        return _fallback_result(reason="empty_messages")

    # chain: prompt | structured_llm（新写法）
    # structured_llm 已通过 ChatOpenAI.with_structured_output(IntentClassification) 绑死 schema，
    # ainvoke() 直接返回 IntentClassification 实例，不需要 parser
    chain = get_intent_classify_prompt() | get_structured_llm()

    try:
        # 注入 messages 占位符（LangGraph 的 ChatPromptTemplate 占位机制）
        result = await chain.ainvoke({"messages": messages})
    except Exception as e:
        # LLM 解析失败 / 网络错误 / schema 校验失败 → 统一降级到 FALLBACK
        logger.exception("意图分类 LLM 调用失败: %s", e)
        return _fallback_result(reason=f"llm_error: {type(e).__name__}")

    # 把 Pydantic 模型转成 dict，匹配 AgentState
    slots_dict = {s.name: s.value for s in result.slots}
    logger.info(
        "意图分类结果: intent=%s, confidence=%.2f, slots=%s",
        result.intent, result.confidence, slots_dict,
    )
    return {
        "current_intent": result.intent,
        "intent_confidence": result.confidence,
        "intent_reasoning": result.reasoning,
        "extracted_slots": slots_dict,
        "needs_clarification": result.needs_clarification,
        "clarification_question": result.clarification_question,
        # 路由相关字段由 intent_router 节点写入
        "routed_node": "",  # 占位，路由节点会覆盖
        "fallback_reason": None,
    }


def _fallback_result(reason: str) -> dict:
    """统一的 fallback 返回结构"""
    return {
        "current_intent": "FALLBACK_UNKNOWN",
        "intent_confidence": 0.0,
        "intent_reasoning": f"意图分类失败或无输入: {reason}",
        "extracted_slots": {},
        "needs_clarification": False,
        "clarification_question": None,
        "routed_node": "fallback_handler",
        "fallback_reason": reason,
    }


# ============================================================
# 节点 2: intent_router（纯逻辑，无 LLM）
# ============================================================
def intent_router(state: AgentState) -> dict:
    """
    路由决策节点（纯逻辑）。

    输入: current_intent / intent_confidence / needs_clarification
    输出: routed_node / fallback_reason

    路由规则（按优先级）:
    1. needs_clarification=True → clarification_handler（追问用户）
    2. confidence < INTENT_CONFIDENCE_THRESHOLD → fallback_handler
    3. intent 映射到 next_node（来自 IntentDefinition.next_node）
    4. 未知 intent → fallback_handler

    为什么这一步要独立成节点（不要在 classify_intent 里直接路由）？
    - **可观测性**: LangSmith trace 里能看到路由决策的每一步
    - **可测试性**: 路由逻辑是纯函数，单测容易写
    - **可替换性**: 以后想换成 LLM replanner，只换这一个节点
    """
    intent = state.get("current_intent", "")
    confidence = state.get("intent_confidence", 0.0)
    needs_clarification = state.get("needs_clarification", False)

    # 规则 1: 需要追问
    if needs_clarification:
        logger.info("路由: needs_clarification=True → clarification_handler")
        return {"routed_node": "clarification_handler", "fallback_reason": None}

    # 规则 2: 置信度不够
    if confidence < INTENT_CONFIDENCE_THRESHOLD:
        logger.info(
            "路由: confidence=%.2f < threshold=%.2f → fallback_handler",
            confidence, INTENT_CONFIDENCE_THRESHOLD,
        )
        return {
            "routed_node": "fallback_handler",
            "fallback_reason": "low_confidence",
        }

    # 规则 3: intent → next_node 映射
    from agent.prompts import load_intent_registry
    registry = load_intent_registry()
    intent_def = next((i for i in registry.intents if i.code == intent), None)

    if intent_def is None:
        logger.warning("路由: 未知 intent=%s → fallback_handler", intent)
        return {
            "routed_node": "fallback_handler",
            "fallback_reason": "unknown_intent",
        }

    logger.info(
        "路由: intent=%s (%.2f) → %s",
        intent_def.code, confidence, intent_def.next_node,
    )
    return {"routed_node": intent_def.next_node, "fallback_reason": None}