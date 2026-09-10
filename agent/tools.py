"""
agent.tools
===========
意图识别后的 10 个业务 handler — 在 create_agent 范式下作为 @tool 暴露给 LLM。

范式转变(v2):
- 旧版:handler 是 LangGraph StateGraph 节点,router 决定调用哪个 handler
- 新版:handler 是 @tool,LLM 根据 system_prompt + tool docstring 自己选调哪个

每个 tool 都是占位 stub — 本仓库只交付意图识别层,handler 内部不实现真实业务逻辑。
返回固定 dict(代表"已路由")，让 create_agent 拿到 tool result 后能完成 loop。
"""

from langchain_core.tools import tool


@tool
def order_query_handler(order_id: str = "") -> str:
    """
    订单查询 — 处理用户查询订单详情 / 商品清单 / 金额 / 下单时间等订单本身信息的请求。

    Args:
        order_id: 订单号,例如 "123456"。如果用户没给,传空字符串。

    Returns:
        str: 已路由标记（占位 stub — 真实业务逻辑在后续迭代实现）
    """
    return f"已路由到 order_query_handler (order_id={order_id or '未提供'})"


@tool
def refund_handler(order_id: str = "") -> str:
    """
    退款售后 — 处理用户申请退款 / 退货 / 换货 / 查看退款进度 / 取消订单等售后请求。

    Args:
        order_id: 订单号,例如 "888999"。如果用户没给,传空字符串。

    Returns:
        str: 已路由标记
    """
    return f"已路由到 refund_handler (order_id={order_id or '未提供'})"


@tool
def payment_handler() -> str:
    """
    支付问题 — 处理用户支付失败 / 支付方式选择 / 发票申请 / 支付到账等支付相关问题。

    Returns:
        str: 已路由标记
    """
    return "已路由到 payment_handler"


@tool
def logistics_handler(order_id: str = "") -> str:
    """
    物流配送 — 处理用户咨询发货时间 / 快递公司 / 收货地址修改 / 签收问题 / 物流追踪等请求。

    Args:
        order_id: 订单号,例如 "123456"。如果用户没给,传空字符串。

    Returns:
        str: 已路由标记
    """
    return f"已路由到 logistics_handler (order_id={order_id or '未提供'})"


@tool
def account_handler() -> str:
    """
    账户会员 — 处理用户账号登录 / 注册 / 会员等级 / 积分 / 余额 / 个人信息修改等请求。

    Returns:
        str: 已路由标记
    """
    return "已路由到 account_handler"


@tool
def promotion_handler() -> str:
    """
    优惠活动 — 处理用户咨询优惠券 / 促销活动 / 折扣 / 满减规则等请求。

    Returns:
        str: 已路由标记
    """
    return "已路由到 promotion_handler"


@tool
def complaint_handler() -> str:
    """
    投诉建议 — 处理用户表达不满 / 投诉 / 表扬 / 产品改进建议等请求。投诉意图盖过具体业务意图。

    Returns:
        str: 已路由标记
    """
    return "已路由到 complaint_handler"


@tool
def chitchat_handler() -> str:
    """
    闲聊寒暄 — 处理用户打招呼 / 闲聊 / 与订单业务无关的问题（如"你好"、"在吗"、"今天天气怎么样"）。

    Returns:
        str: 已路由标记
    """
    return "已路由到 chitchat_handler"


@tool
def clarification_handler(question: str = "请提供更具体的信息") -> str:
    """
    信息追问 — 当用户意图明确但缺少必要信息（如问订单但没给订单号）时调用，向用户追问缺失信息。

    Args:
        question: 追问用户的问题文本，例如 "请提供订单号"。

    Returns:
        str: 已路由标记（包含追问问题）
    """
    return f"已路由到 clarification_handler (question={question})"


@tool
def fallback_handler(reason: str = "未识别意图") -> str:
    """
    兜底分支 — 用户意图不属于其他 9 类时调用。置信度 < 0.6 或完全未知场景。

    Args:
        reason: 触发 fallback 的原因描述，例如 "low_confidence" 或 "unknown_intent"。

    Returns:
        str: 已路由标记
    """
    return f"已路由到 fallback_handler (reason={reason})"


# 集中导出 — workflow.py 里直接 ALL_TOOLS
ALL_TOOLS = [
    order_query_handler,
    refund_handler,
    payment_handler,
    logistics_handler,
    account_handler,
    promotion_handler,
    complaint_handler,
    chitchat_handler,
    clarification_handler,
    fallback_handler,
]