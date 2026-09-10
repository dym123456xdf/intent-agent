"""
agent.models
============
意图识别模块的 Pydantic schema。

v2 改动（create_agent 范式）:
- IntentClassification: 仍然是 LLM 的结构化输出 schema,通过 create_agent 的 response_format 传入
- IntentCode / IntentSlot: 保留作为 schema 依赖
- IntentDefinition / IntentRegistry: 保留,仍然从 data/intents.json 加载并用于构造 @tool
"""

from typing import Literal

from pydantic import BaseModel, Field

# —— 8 大意图枚举 ——
# Literal 强制 LLM 只能输出这 8 个值之一，防止幻觉
IntentCode = Literal[
    "ORDER_QUERY",         # 订单查询
    "REFUND_AFTER_SALE",   # 退款售后
    "PAYMENT_ISSUE",       # 支付问题
    "LOGISTICS_DELIVERY",  # 物流配送
    "ACCOUNT_MEMBER",      # 账户会员
    "PROMOTION_COUPON",    # 优惠活动
    "COMPLAINT_SUGGESTION",# 投诉建议
    "CHITCHAT_GREETING",   # 闲聊寒暄
]


class IntentSlot(BaseModel):
    """从用户语句中抽取的关键参数（命名实体 / 槽位）"""
    name: str = Field(description="槽位名，如 order_id / product_id / user_id")
    value: str = Field(description="槽位值，如 '1234567890'")


class IntentClassification(BaseModel):
    """
    单条用户消息的意图分类结果（LLM 输出的 schema）。

    字段说明:
    - intent: 8 类意图之一
    - confidence: 模型对自己判断的置信度（0~1）
    - reasoning: 一句话解释为什么这样分类（用于 trace 调试）
    - slots: 从用户语句中抽取的关键参数（订单号、商品号等）
    - needs_clarification: 是否需要向用户追问（信息不足）
    - clarification_question: 需要追问的问题

    在 create_agent 范式下，这个 schema 通过 response_format=IntentClassification 传入 agent；
    agent 的最后一次输出会按此 schema 校验，最终结果在 result["structured_response"] 里。
    """
    intent: IntentCode = Field(description="识别出的意图编码（必须是 8 类之一）")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="置信度，0~1，越接近 1 表示越确定",
    )
    reasoning: str = Field(description="一句话解释为什么这样分类，便于 trace 调试")
    slots: list[IntentSlot] = Field(
        default_factory=list,
        description="从用户语句中抽取的槽位（订单号、商品号等）",
    )
    needs_clarification: bool = Field(
        default=False,
        description="信息不足时是否需要追问用户",
    )
    clarification_question: str | None = Field(
        default=None,
        description="追问用户的问题（needs_clarification=True 时必填）",
    )


class IntentDefinition(BaseModel):
    """意图定义（来自 data/intents.json）"""
    code: str
    name: str
    description: str
    examples: list[str] = Field(default_factory=list)
    required_slots: list[str] = Field(default_factory=list)
    next_node: str


class IntentRegistry(BaseModel):
    """意图注册表（来自 data/intents.json 的整体结构）"""
    version: str = "1.0.0"
    description: str = ""
    intents: list[IntentDefinition]
    fallback: IntentDefinition | None = None