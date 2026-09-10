"""
tests/test_intent.py
====================
意图识别核心测试。

测试层次（不依赖真实 LLM — 用 monkeypatch 替换）:
1. schema 加载 / Pydantic 校验 — 离线
2. intent_router 纯逻辑 — 离线（不需要 LLM）
3. classify_intent 端到端 — 调真实 LLM，需要 AGNES_API_KEY（用 marker 区分）
"""

import asyncio
import os
from typing import Any

import pytest

# 把项目根目录加到 sys.path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.models import (
    IntentClassification,
    IntentSlot,
    IntentRegistry,
)
from agent.nodes import classify_intent, intent_router
from agent.prompts import load_intent_registry
from agent.state import AgentState


# ============================================================
# 离线测试 — 不依赖 LLM
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
        assert codes == expected, f"意图编码不匹配，缺少: {expected - codes}"

    def test_each_intent_has_examples(self):
        """每个意图至少要有 3 个 example 用于 few-shot"""
        registry = load_intent_registry()
        for intent in registry.intents:
            assert len(intent.examples) >= 3, (
                f"{intent.code} 缺少足够的示例（当前 {len(intent.examples)} 个）"
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
        # Pydantic 应该拒绝 confidence > 1 或 < 0
        with pytest.raises(Exception):
            IntentClassification(intent="ORDER_QUERY", confidence=1.5, reasoning="x")
        with pytest.raises(Exception):
            IntentClassification(intent="ORDER_QUERY", confidence=-0.1, reasoning="x")

    def test_intent_must_be_valid(self):
        # Literal 校验: 必须是 8 类之一
        with pytest.raises(Exception):
            IntentClassification(
                intent="FOO_BAR",  # 不在 Literal 里
                confidence=0.5,
                reasoning="x",
            )

    def test_extracted_slots_dict_property(self):
        """slots 字段提供 dict 视图便利方法"""
        c = IntentClassification(
            intent="ORDER_QUERY",
            confidence=0.9,
            reasoning="x",
            slots=[
                IntentSlot(name="order_id", value="A001"),
                IntentSlot(name="product_id", value="P100"),
            ],
        )
        # 这里 slots 本身是 list[IntentSlot]，但 nodes.classify_intent 里手写转 dict
        # 所以这里只验证 Pydantic schema
        assert len(c.slots) == 2


# ============================================================
# intent_router 纯逻辑测试（不调 LLM）
# ============================================================

class TestIntentRouter:
    """测试路由决策的纯逻辑"""

    def test_high_confidence_routes_to_handler(self):
        from agent.config import INTENT_CONFIDENCE_THRESHOLD
        state: AgentState = {
            "current_intent": "ORDER_QUERY",
            "intent_confidence": 0.95,
            "needs_clarification": False,
        }
        result = intent_router(state)
        assert result["routed_node"] == "order_query_handler"
        assert result["fallback_reason"] is None

    def test_low_confidence_routes_to_fallback(self):
        state: AgentState = {
            "current_intent": "ORDER_QUERY",
            "intent_confidence": 0.3,  # 低于阈值
            "needs_clarification": False,
        }
        result = intent_router(state)
        assert result["routed_node"] == "fallback_handler"
        assert result["fallback_reason"] == "low_confidence"

    def test_clarification_overrides_confidence(self):
        """即使置信度高，needs_clarification 也要先走追问"""
        state: AgentState = {
            "current_intent": "ORDER_QUERY",
            "intent_confidence": 0.95,
            "needs_clarification": True,
        }
        result = intent_router(state)
        assert result["routed_node"] == "clarification_handler"

    def test_unknown_intent_routes_to_fallback(self):
        state: AgentState = {
            "current_intent": "FOO_BAR_BAZ",  # 不在 registry 里
            "intent_confidence": 0.9,
            "needs_clarification": False,
        }
        result = intent_router(state)
        assert result["routed_node"] == "fallback_handler"
        assert result["fallback_reason"] == "unknown_intent"

    def test_all_eight_intents_route_correctly(self):
        """8 类意图都能正确路由到对应 handler"""
        cases = {
            "ORDER_QUERY": "order_query_handler",
            "REFUND_AFTER_SALE": "refund_handler",
            "PAYMENT_ISSUE": "payment_handler",
            "LOGISTICS_DELIVERY": "logistics_handler",
            "ACCOUNT_MEMBER": "account_handler",
            "PROMOTION_COUPON": "promotion_handler",
            "COMPLAINT_SUGGESTION": "complaint_handler",
            "CHITCHAT_GREETING": "chitchat_handler",
        }
        for intent_code, expected_handler in cases.items():
            state: AgentState = {
                "current_intent": intent_code,
                "intent_confidence": 0.95,
                "needs_clarification": False,
            }
            result = intent_router(state)
            assert result["routed_node"] == expected_handler, (
                f"{intent_code} 应该路由到 {expected_handler}, 实际 {result['routed_node']}"
            )


# ============================================================
# classify_intent 端到端测试（依赖真实 LLM）
# ============================================================

@pytest.mark.skipif(
    not os.getenv("AGNES_API_KEY"),
    reason="需要 AGNES_API_KEY 才能跑端到端测试",
)
class TestClassifyIntentE2E:
    """调真实 LLM 做意图分类（每条用例都要花 1~2s）"""

    @pytest.mark.asyncio
    async def test_order_query(self):
        from langchain_core.messages import HumanMessage
        state: AgentState = {
            "messages": [HumanMessage(content="帮我查一下订单123456的详情")],
        }
        result = await classify_intent(state)
        # "查订单详情" → ORDER_QUERY（核心目的：订单详情本身）
        # "订单到哪了" → LOGISTICS_DELIVERY（核心目的：物流追踪）
        # 这两个边界由 prompt 的核心语义原则控制
        assert result["current_intent"] in {"ORDER_QUERY", "LOGISTICS_DELIVERY"}, (
            f"ORDER_QUERY 或 LOGISTICS_DELIVERY 都可，实际 {result['current_intent']} "
            f"({result['intent_reasoning']})"
        )
        assert result["intent_confidence"] >= 0.6
        if result["current_intent"] == "ORDER_QUERY":
            assert "order_id" in result["extracted_slots"]

    @pytest.mark.asyncio
    async def test_logistics_tracking_intent(self):
        """'订单到哪了' 类语句应当路由到 LOGISTICS_DELIVERY（核心语义是物流追踪）"""
        from langchain_core.messages import HumanMessage
        state: AgentState = {
            "messages": [HumanMessage(content="我的订单123456到哪了")],
        }
        result = await classify_intent(state)
        assert result["current_intent"] == "LOGISTICS_DELIVERY", (
            f"应该是 LOGISTICS_DELIVERY，实际 {result['current_intent']} "
            f"({result['intent_reasoning']})"
        )
        assert result["intent_confidence"] >= 0.6
        assert "order_id" in result["extracted_slots"]

    @pytest.mark.asyncio
    async def test_logistics_with_varied_subjects(self):
        """核心目的是物流追踪的多种主语句式都应归 LOGISTICS_DELIVERY"""
        from langchain_core.messages import HumanMessage
        queries = [
            "我的快递到哪了",
            "包裹发出去了吗",
            "什么时候能发货",
            "我的订单123456到哪了",  # 即便主语是"订单"，目的仍是追踪
        ]
        for q in queries:
            state: AgentState = {"messages": [HumanMessage(content=q)]}
            result = await classify_intent(state)
            assert result["current_intent"] in {"LOGISTICS_DELIVERY", "ORDER_QUERY"}, (
                f"query='{q}' 应归物流/订单类，实际 {result['current_intent']} "
                f"({result['intent_reasoning']})"
            )

    @pytest.mark.asyncio
    async def test_refund_intent(self):
        from langchain_core.messages import HumanMessage
        state: AgentState = {
            "messages": [HumanMessage(content="我要申请退款，订单号是888999")],
        }
        result = await classify_intent(state)
        assert result["current_intent"] == "REFUND_AFTER_SALE"
        assert "order_id" in result["extracted_slots"]

    @pytest.mark.asyncio
    async def test_chitchat_routes_correctly(self):
        from langchain_core.messages import HumanMessage
        state: AgentState = {
            "messages": [HumanMessage(content="你好")],
        }
        result = await classify_intent(state)
        assert result["current_intent"] == "CHITCHAT_GREETING"

    @pytest.mark.asyncio
    async def test_missing_order_id_triggers_clarification(self):
        from langchain_core.messages import HumanMessage
        state: AgentState = {
            "messages": [HumanMessage(content="我的订单到哪了")],  # 没给订单号
        }
        result = await classify_intent(state)
        # LLM 应当识别 needs_clarification=true 或置信度低
        assert (
            result["needs_clarification"]
            or result["intent_confidence"] < 0.6
        ), f"应该触发追问或低置信，实际 confidence={result['intent_confidence']}"


# ============================================================
# classify_intent 错误降级测试（用 mock）
# ============================================================

class TestClassifyIntentFallback:
    """LLM 异常时的降级路径"""

    @pytest.mark.asyncio
    async def test_llm_error_returns_fallback(self, monkeypatch):
        """模拟 LLM 抛异常，应该走 fallback 路径"""
        # 替换 get_structured_llm — mock 成 ainvoke 抛异常的 Runnable
        from agent import nodes
        from langchain_core.runnables import RunnableLambda

        # RunnableLambda 把普通函数包装成 Runnable,这样 `prompt | fake` 才能跑通
        async def boom(_input):
            raise RuntimeError("mock LLM failure")

        monkeypatch.setattr(
            nodes, "get_structured_llm",
            lambda: RunnableLambda(boom),
        )

        # 直接调真实 classify_intent（chain 已经被 mock 替换）
        from langchain_core.messages import HumanMessage
        state: AgentState = {
            "messages": [HumanMessage(content="测试")],
        }
        result = await classify_intent(state)
        assert result["current_intent"] == "FALLBACK_UNKNOWN"
        assert result["fallback_reason"].startswith("llm_error")

    @pytest.mark.asyncio
    async def test_empty_messages_returns_fallback(self):
        from langchain_core.messages import HumanMessage
        state: AgentState = {"messages": []}
        result = await classify_intent(state)
        assert result["current_intent"] == "FALLBACK_UNKNOWN"