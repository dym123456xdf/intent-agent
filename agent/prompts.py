"""
agent.prompts
=============
意图分类 system prompt + 意图清单加载。

v2 改动（create_agent 范式）:
- 不再用 ChatPromptTemplate|from_messages（那是 LCEL chain 写法）
- 改用纯 system_prompt 字符串，喂给 create_agent 的 system_prompt= 参数
- 意图清单从 data/intents.json 加载后注入到 system prompt
"""

import json

from agent.config import INTENTS_FILE
from agent.models import IntentDefinition, IntentRegistry


def load_intent_registry() -> IntentRegistry:
    """加载 data/intents.json → IntentRegistry"""
    with INTENTS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return IntentRegistry.model_validate(data)


def _build_intent_list_text(registry: IntentRegistry) -> str:
    """把意图清单格式化成可注入到 system prompt 的文本"""
    lines = []
    for intent in registry.intents:
        examples = " / ".join(intent.examples[:3])
        lines.append(
            f"- `{intent.code}` ({intent.name}): {intent.description}\n"
            f"  示例: {examples}"
        )
    return "\n".join(lines)


# 启动时一次性构建 — registry 文本在 system prompt 里展开
_registry = load_intent_registry()
_INTENT_LIST_TEXT = _build_intent_list_text(_registry)


# create_agent 范式下，system_prompt 是一个字符串；
# @tool 装饰的 handler 各自带 docstring，所以 tool 选择靠 docstring + 这个 system_prompt 共同驱动
SYSTEM_PROMPT = f"""你是订单系统在线客服的【意图识别与路由引擎】。

【意图清单】
{_INTENT_LIST_TEXT}

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
   - "订单号缺失 + 问订单状态" → 调 `clarification_handler` tool，向用户追问订单号。
3. **置信度规则**:
   - 高置信: 0.85~1.0 — 句式典型、意图明确。
   - 中置信: 0.6~0.85 — 有一定模糊性但可推断。
   - 低置信: 0.0~0.6 — 模糊或边界场景，必须调 `clarification_handler` 或 `fallback_handler` tool。
4. **闲聊处理**:打招呼/闲聊/与订单无关的问题 → 调 `chitchat_handler` tool。
5. **槽位抽取**:如果用户提到订单号、商品号等具体 ID，必须抽取到 slots 字段。
6. **未知意图**:如果用户问句不属于上面 8 类中的任何一类，调 `fallback_handler` tool，不要硬猜。
7. **绝不允许脑补参数**:如果用户问句里**没有**显式给出订单号/商品号等 ID（例如"我的订单到哪了"、"查一下我的订单"），即使主语里有"订单"两字，也**不要**把"我的订单"等当作 order_id 传入 handler——必须先调 `clarification_handler` 向用户追问。这是硬规则。

【输出要求】
- 最终一轮按 IntentClassification schema 输出 JSON：包含 intent/confidence/reasoning/slots/needs_clarification/clarification_question 字段。
- 不要省略字段，不要加多余字段。
"""