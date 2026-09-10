"""
tests/test_intent.py
====================
v2 改造:用 create_agent 范式后的测试。

测试层次:
1. schema 加载 + Pydantic 校验 — 离线
2. tool 注册 + docstring 校验 — 离线
3. _flatten_result 输出形状 — 离线（用 mock agent 输出）
4. agent.invoke() 端到端 — 真实 LLM，需要 AGNES_API_KEY

v1 时代的 TestIntentRouter(纯逻辑路由)整个删除 — create_agent 内部用 LLM 选 tool,
不再有纯函数 router 节点。
"""

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.models import (
    IntentClassification,
    IntentSlot,
    IntentRegistry,
)
from agent.prompts import SYSTEM_PROMPT, load_intent_registry
from agent.result_extract import (
    _extract_fallback_reason,
    _extract_routed_node,
    flatten_agent_result,
)
from agent.tools import (
    ALL_TOOLS,
    account_handler,
    chitchat_handler,
    clarification_handler,
    complaint_handler,
    fallback_handler,
    logistics_handler,
    order_query_handler,
    payment_handler,
    promotion_handler,
    refund_handler,
)


# ============================================================
# 离线测试 — schema + tool 注册
# ============================================================

class TestIntentRegistry:
    """data/intents.json 加载 + Pydantic 校验"""

    def test_load_registry(self):
        registry = load_intent_registry()
        assert isinstance(registry, IntentRegistry)
        assert len(registry.intents) == 8, "应该有 8 类意图"

    def test_all_intents_have_code_name(self):
        registry = load_intent_registry()
        codes = {i.code for i in registry.intents}
        expected = {
            "ORDER_QUERY", "REFUND_AFTER_SALE", "PAYMENT_ISSUE",
            "LOGISTICS_DELIVERY", "ACCOUNT_MEMBER", "PROMOTION_COUPON",
            "COMPLAINT_SUGGESTION", "CHITCHAT_GREETING",
        }
        assert codes == expected, f"意图编码不匹配,缺少: {expected - codes}"

    def test_each_intent_has_examples(self):
        registry = load_intent_registry()
        for intent in registry.intents:
            assert len(intent.examples) >= 3, (
                f"{intent.code} 缺少足够的示例(当前 {len(intent.examples)} 个)"
            )

    def test_fallback_defined(self):
        registry = load_intent_registry()
        assert registry.fallback is not None
        assert registry.fallback.code == "FALLBACK_UNKNOWN"


class TestIntentClassificationSchema:
    """Pydantic schema 校验"""

    def test_valid_full(self):
        c = IntentClassification(
            intent="ORDER_QUERY",
            confidence=0.95,
            reasoning="用户问订单状态",
            slots=[IntentSlot(name="order_id", value="123456")],
            needs_clarification=False,
        )
        assert c.intent == "ORDER_QUERY"
        assert c.slots[0].name == "order_id"
        assert c.slots[0].value == "123456"

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            IntentClassification(intent="ORDER_QUERY", confidence=1.5, reasoning="x")
        with pytest.raises(Exception):
            IntentClassification(intent="ORDER_QUERY", confidence=-0.1, reasoning="x")

    def test_intent_must_be_valid(self):
        with pytest.raises(Exception):
            IntentClassification(
                intent="FOO_BAR",  # type: ignore[arg-type]
                confidence=0.5,
                reasoning="x",
            )


class TestToolRegistration:
    """10 个 @tool handler 注册 + 形状校验"""

    def test_all_tools_have_docstring(self):
        """每个 tool 必须有 docstring — LLM 选 tool 的唯一依据"""
        for tool in ALL_TOOLS:
            assert tool.description, f"{tool.name} 缺少 docstring"
            # docstring 第一行(到第一个句号/换行)应是中文业务描述
            first_line = tool.description.split("\n")[0].split("。")[0]
            assert len(first_line) >= 5, (
                f"{tool.name} docstring 太短: {tool.description[:80]}"
            )

    def test_all_tools_count(self):
        """必须有 10 个 tool:8 业务 + clarification + fallback"""
        assert len(ALL_TOOLS) == 10, f"期望 10 个 tool,实际 {len(ALL_TOOLS)}"

    def test_tool_names_match_intents(self):
        """tool 名必须对应意图清单"""
        registry = load_intent_registry()
        expected = {i.next_node for i in registry.intents} | {"clarification_handler", "fallback_handler"}
        actual = {t.name for t in ALL_TOOLS}
        assert actual == expected, f"tool 名与意图不匹配: 缺 {expected - actual}, 多 {actual - expected}"

    def test_tools_return_string(self):
        """所有 tool 都返回 str(stub 返回固定路由标记)"""
        result = order_query_handler.invoke({"order_id": "123456"})
        assert isinstance(result, str)
        assert "已路由" in result


# ============================================================
# _flatten_result 纯逻辑测试
# ============================================================

class TestFlattenResult:
    """agent.invoke() 输出 → UI 视图的扁平化函数"""

    def _make_message(self, role: str, tool_calls: list | None = None, content: str = "") -> MagicMock:
        m = MagicMock()
        m.tool_calls = tool_calls or []
        m.content = content
        return m

    def test_extract_routed_node_skips_intent_classification(self):
        """routed_node 应跳过 IntentClassification 伪 tool,选最后一个真实业务 tool"""
        messages = [
            self._make_message("ai", [{"name": "logistics_handler", "args": {}}]),
            self._make_message("tool", content="ok"),
            self._make_message("ai", [{"name": "IntentClassification", "args": {}}]),
            self._make_message("tool", content="ok"),
        ]
        assert _extract_routed_node(messages) == "logistics_handler"

    def test_extract_routed_node_multi_tool_chain(self):
        """多 tool 链时(物流→追问)取最后一个真实 tool"""
        messages = [
            self._make_message("ai", [{"name": "logistics_handler", "args": {}}]),
            self._make_message("tool"),
            self._make_message("ai", [{"name": "clarification_handler", "args": {}}]),
            self._make_message("tool"),
            self._make_message("ai", [{"name": "IntentClassification", "args": {}}]),
            self._make_message("tool"),
        ]
        assert _extract_routed_node(messages) == "clarification_handler"

    def test_extract_routed_node_empty(self):
        """无 tool_calls → 空字符串"""
        messages = [self._make_message("human")]
        assert _extract_routed_node(messages) == ""

    def test_extract_fallback_reason(self):
        """fallback_handler 的 reason 从 args 里拿"""
        messages = [
            self._make_message("ai", [{"name": "fallback_handler", "args": {"reason": "unknown_intent"}}]),
        ]
        assert _extract_fallback_reason(messages, "fallback_handler") == "unknown_intent"

    def test_extract_fallback_reason_not_fallback(self):
        """routed_node 不是 fallback → reason 应是 None"""
        messages = [
            self._make_message("ai", [{"name": "logistics_handler", "args": {}}]),
        ]
        assert _extract_fallback_reason(messages, "logistics_handler") is None

    def test_flatten_result_full(self):
        """完整 agent 输出 → UI 视图"""
        structured = IntentClassification(
            intent="LOGISTICS_DELIVERY",
            confidence=0.95,
            reasoning="追踪物流",
            slots=[IntentSlot(name="order_id", value="123")],
            needs_clarification=False,
        )
        messages = [
            self._make_message("human"),
            self._make_message("ai", [{"name": "logistics_handler", "args": {"order_id": "123"}}]),
            self._make_message("tool"),
            self._make_message("ai", [{"name": "IntentClassification", "args": {}}]),
            self._make_message("tool"),
        ]
        result = {
            "messages": messages,
            "structured_response": structured,
        }
        flat = flatten_agent_result(result)
        assert flat["current_intent"] == "LOGISTICS_DELIVERY"
        assert flat["intent_confidence"] == 0.95
        assert flat["extracted_slots"] == {"order_id": "123"}
        assert flat["routed_node"] == "logistics_handler"
        assert flat["fallback_reason"] is None

    def test_flatten_result_missing_structured(self):
        """agent 异常没拿到 structured_response → 字段默认值"""
        messages = [self._make_message("human")]
        flat = flatten_agent_result({"messages": messages})
        assert flat["current_intent"] == "N/A"
        assert flat["intent_confidence"] == 0.0
        assert flat["routed_node"] == ""


# ============================================================
# 端到端测试 — 真实 LLM
# ============================================================

@pytest.mark.skipif(
    not os.getenv("AGNES_API_KEY"),
    reason="需要 AGNES_API_KEY 才能跑端到端测试",
)
class TestAgentE2E:
    """调真实 AGNES LLM — 每条用例都要花几秒"""

    @pytest.mark.asyncio
    async def test_logistics_with_order_id(self):
        from langchain_core.messages import HumanMessage
        from agent.workflow import get_agent

        agent = get_agent()
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="我的订单123456到哪了")]
        })
        sr = result["structured_response"]
        assert sr.intent == "LOGISTICS_DELIVERY", f"应该是 LOGISTICS_DELIVERY, 实际 {sr.intent}"
        assert sr.confidence >= 0.6
        slot_dict = {s.name: s.value for s in sr.slots}
        assert "order_id" in slot_dict

    @pytest.mark.asyncio
    async def test_refund_intent(self):
        from langchain_core.messages import HumanMessage
        from agent.workflow import get_agent

        agent = get_agent()
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="我要申请退款,订单号是888999")]
        })
        sr = result["structured_response"]
        assert sr.intent == "REFUND_AFTER_SALE"
        slot_dict = {s.name: s.value for s in sr.slots}
        assert "order_id" in slot_dict

    @pytest.mark.asyncio
    async def test_chitchat(self):
        from langchain_core.messages import HumanMessage
        from agent.workflow import get_agent

        agent = get_agent()
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="你好")]
        })
        sr = result["structured_response"]
        assert sr.intent == "CHITCHAT_GREETING"

    @pytest.mark.asyncio
    async def test_missing_order_id_triggers_clarification(self):
        """信息缺失时 LLM 应主动调 clarification_handler"""
        from langchain_core.messages import HumanMessage
        from agent.workflow import get_agent

        agent = get_agent()
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="我的订单到哪了")]
        })
        sr = result["structured_response"]
        # 要么 needs_clarification,要么 routed_node 是 clarification_handler
        flat = flatten_agent_result(result)
        assert (
            sr.needs_clarification
            or flat["routed_node"] == "clarification_handler"
        ), f"应该触发追问,实际 intent={sr.intent}, routed={flat['routed_node']}"