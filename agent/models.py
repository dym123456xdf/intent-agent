"""
agent.models
============
意图识别模块的 Pydantic schema。

为什么用 Pydantic 而不是 TypedDict？
- Pydantic 在 langchain 输出解析器（langchain_core.output_parsers.PydanticOutputParser）
  里是一等公民，可以直接 .parse() + 注入到 prompt 做 format_instructions
- TypedDict 是 LangGraph state 用的；这里要做 LLM 结构化输出校验，要用 Pydantic
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

    字段说明：
    - intent: 8 类意图之一
    - confidence: 模型对自己判断的置信度（0~1）
    - reasoning: 一句话解释为什么这样分类（用于 trace 调试）
    - slots: 从用户语句中抽取的关键参数（订单号、商品号等）
    - needs_clarification: 是否需要向用户追问（信息不足）
    - clarification_question: 需要追问的问题
    """
    intent: IntentCode = Field(description="识别出的意图编码（必须是 8 类之一）")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="置信度，0~1，越接近 1 表示越确定"
    )
    reasoning: str = Field(description="一句话解释为什么这样分类，便于 trace 调试")
    slots: list[IntentSlot] = Field(
        default_factory=list,
        description="从用户语句中抽取的槽位（订单号、商品号等）"
    )
    needs_clarification: bool = Field(
        default=False,
        description="信息不足时是否需要追问用户"
    )
    clarification_question: str | None = Field(
        default=None,
        description="追问用户的问题（needs_clarification=True 时必填）"
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