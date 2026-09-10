"""
agent.prompts
=============
意图分类 prompt 模板。

设计要点:
1. **Few-shot 示例**:每个意图给 2-3 个典型问句，让 LLM 学到分类边界
2. **结构化输出**:用 `ChatOpenAI.with_structured_output(IntentClassification)`
   让 provider 在生成时按 schema 约束（function_calling / json_schema）。
   不再需要 PydanticOutputParser + format_instructions 占位。
3. **多轮对话**:注入 messages 历史，让 LLM 考虑上下文
4. **低置信度策略**:prompt 鼓励 LLM 在模糊时输出 low confidence + needs_clarification
"""

import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from agent.config import INTENTS_FILE
from agent.models import IntentClassification, IntentRegistry


def load_intent_registry() -> IntentRegistry:
    """加载 data/intents.json → IntentRegistry"""
    with INTENTS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return IntentRegistry.model_validate(data)


# —— 意图清单字符串（注入 prompt） ——
def _build_intent_list_text(registry: IntentRegistry) -> str:
    lines = []
    for intent in registry.intents:
        examples = " / ".join(intent.examples[:3])
        lines.append(
            f"- `{intent.code}` ({intent.name}): {intent.description}\n"
            f"  示例: {examples}"
        )
    return "\n".join(lines)


# 启动时一次性构建 registry 文本（注入到 prompt 的 system message）
_registry_text = _build_intent_list_text(load_intent_registry())

INTENT_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"""你是订单系统在线客服的【意图识别引擎】。你的任务是：阅读用户当前的问句（以及多轮上下文），把它分类到下面 8 个意图之一，并抽取关键参数。

【意图清单】
{_registry_text}

【分类原则】
1. **核心语义**:看用户问句的**核心目的**是什么，不是看表面的词。
   - "我的订单到哪了" / "我的快递到哪了" / "包裹发出去了吗" → **物流配送**(LOGISTICS_DELIVERY)。
     即使主语是"订单"，只要目的是"追踪物流/配送进度"，就归 LOGISTICS_DELIVERY，不归 ORDER_QUERY。
   - "帮我查一下订单123456的详情" / "订单里有哪些商品" → **订单查询**(ORDER_QUERY)。
     核心目的是订单本身的信息（详情、商品清单、金额、下单时间），而非物流位置。
   - "我已经付款了为什么还是未支付" → 支付问题（PAYMENT_ISSUE），不是订单查询。
   - "我想退货，订单号是123456" → 退款售后（REFUND_AFTER_SALE），同时抽取 order_id=123456。
2. **优先级冲突时的取舍**:
   - "投诉 + 具体业务" → 投诉建议（COMPLAINT_SUGGESTION）。投诉意图盖过具体业务意图。
   - "订单号缺失 + 问订单状态" → 标记 needs_clarification=true，问用户要订单号。
3. **置信度规则**:
   - 高置信: 0.85~1.0 — 句式典型、意图明确。
   - 中置信: 0.6~0.85 — 有一定模糊性但可推断。
   - 低置信: 0.0~0.6 — 模糊或边界场景，必须 needs_clarification=true 或置信度低。
4. **闲聊处理**:打招呼/闲聊/与订单无关的问题 → CHITCHAT_GREETING。
5. **槽位抽取**:如果用户提到订单号、商品号等具体 ID，必须抽取到 slots 字段。
6. **追问策略**:信息缺失（如查询订单但没给订单号）时，必须 needs_clarification=true 并写出追问问题。

严格按 IntentClassification schema 输出 JSON — 不要加多余字段，不要省略字段。
"""),
    # 注入多轮对话历史（占位 {messages} 由 LangGraph 自动填）
    ("placeholder", "{messages}"),
])


def get_intent_classify_prompt() -> ChatPromptTemplate:
    """暴露给外部的 prompt getter — 测试时可以 monkeypatch"""
    return INTENT_CLASSIFY_PROMPT