"""
agent.workflow
===============
LangGraph StateGraph 编排 — Plan-and-Execute 模式的第一步（意图识别 + 路由）。

图结构（当前只做意图识别 + 路由，下游 handler 留 hook）:

    ┌───────┐
    │ START │
    └───┬───┘
        │
        ▼
    ┌──────────────┐
    │ classify_    │   ← 调 LLM 做意图分类
    │ intent       │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ intent_      │   ← 纯逻辑路由（不调 LLM）
    │ router       │
    └──┬────┬──┬───┘
       │    │  │
       ▼    ▼  ▼
   fallback clarification  order_query_*
   handler  handler       handler / ...

   （handlers 是占位 stub，本版本不实现具体业务处理）

路由函数设计要点：
- intent_router 节点写入 state["routed_node"]
- route_after_router 从 state 读 routed_node → 直接返回（已经是目标节点名）
- path_map 的 key == value（解耦不强求，但这里简洁）
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import classify_intent, intent_router
from agent.state import AgentState


def _route_after_router(state: AgentState) -> str:
    """
    条件边路由函数（after intent_router）。

    参数1 (state: AgentState): LangGraph 注入的当前 state。
    规则:
      - state["routed_node"] 是目标节点名（由 intent_router 写入）
      - 直接返回这个字符串作为 path_map 的 key
      - 必须保证返回的 key 在 path_map 里存在
    """
    routed = state.get("routed_node", "fallback_handler")
    return routed


# —— 下游 handler 占位 stub ——
# 本版本不实现具体业务处理，仅留 hook 给后续工作流扩展。
async def fallback_handler(state: AgentState) -> dict:
    """兜底分支：未识别意图"""
    return {"routed_node": "fallback_handler"}


async def clarification_handler(state: AgentState) -> dict:
    """追问分支：向用户索取缺失信息"""
    return {"routed_node": "clarification_handler"}


# 8 个业务 handler 占位
HANDLER_NODES = [
    "order_query_handler",
    "refund_handler",
    "payment_handler",
    "logistics_handler",
    "account_handler",
    "promotion_handler",
    "complaint_handler",
    "chitchat_handler",
]

# —— 把所有 handler 注册为同一个 stub 函数（占位） ——
async def _handler_stub(state: AgentState) -> dict:
    """
    占位 handler：实际项目里每个 handler 是独立实现（调业务系统 / DB / API）。
    本版本只演示意图识别 + 路由，handler 输出 "意图已路由到: X" 即可。
    """
    return {"routed_node": state.get("routed_node", "fallback_handler")}


def build_graph() -> StateGraph:
    """
    构建意图识别 StateGraph。

    返回未编译的 StateGraph 实例（方便测试时可视化 / 修改）。

    关键 API:
    - StateGraph(AgentState): 构造图，state schema 是 AgentState TypedDict
    - add_node(name, fn): 注册节点。name 是字符串门牌，fn 是可调用对象（无括号！）
    - add_edge(src, dst): 铺铁路，无条件从 src 到 dst
    - add_conditional_edges(src, route_fn, path_map): 建岔口，route_fn 返回值 → path_map 查表 → 跳节点
    - compile(): 烘焙，返回可执行的 runnable
    """
    workflow = StateGraph(AgentState)

    # ── 注册节点 ──
    # add_node(name, fn) — name 是字符串门牌，fn 是可调用对象（无括号！）
    # 参数1 (name: str): 节点的稳定字符串 ID（用于 edge 引用 + JSON 序列化）
    # 参数2 (fn: callable): 节点函数 (state: AgentState) -> dict
    # 规则: name 在同一图中必须唯一；fn 绝不能带括号（带括号会立即调用并注册返回值）
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("intent_router", intent_router)
    workflow.add_node("fallback_handler", fallback_handler)
    workflow.add_node("clarification_handler", clarification_handler)
    # 业务 handler 占位
    for h in HANDLER_NODES:
        workflow.add_node(h, _handler_stub)

    # ── 铺边 ──
    # add_edge(src, dst) — 从 src 无条件到 dst
    workflow.add_edge(START, "classify_intent")
    workflow.add_edge("classify_intent", "intent_router")

    # ── 建岔口 ──
    # add_conditional_edges(src, route_fn, path_map)
    # 参数1 (src: str): 上游节点名（"intent_router"）
    # 参数2 (route_fn: callable): 路由函数 (state) -> str
    # 参数3 (path_map: dict): 路由函数返回值 → 目标节点名 的映射
    #   key = 路由函数返回值的字符串（不是节点名！）
    #   value = 目标节点名（必须 add_node 注册过，否则 KeyError）
    # 规则: value 必须是 add_node 注册过的 name；key 可以是任意字符串
    workflow.add_conditional_edges(
        "intent_router",
        _route_after_router,
        {
            # key == value 风格（简洁但不强制）— 这里因为 router 直接吐出目标节点名
            "fallback_handler": "fallback_handler",
            "clarification_handler": "clarification_handler",
            "order_query_handler": "order_query_handler",
            "refund_handler": "refund_handler",
            "payment_handler": "payment_handler",
            "logistics_handler": "logistics_handler",
            "account_handler": "account_handler",
            "promotion_handler": "promotion_handler",
            "complaint_handler": "complaint_handler",
            "chitchat_handler": "chitchat_handler",
        },
    )

    # ── 业务 handler → END ──
    for h in HANDLER_NODES:
        workflow.add_edge(h, END)
    workflow.add_edge("fallback_handler", END)
    workflow.add_edge("clarification_handler", END)

    return workflow


# —— 编译好的 runnable 单例 ——
_compiled_graph = None


def get_compiled_graph():
    """
    获取已编译的图（单例）。

    编译一次后复用 — 否则每次 invoke 都会重新编译图（不致命但浪费）。
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph