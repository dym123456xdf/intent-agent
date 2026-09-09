"""
agent.routes
=============
路由函数。workflow.py 里 _route_after_router 实际上就放这里更合适，
但 langgraph 的 conditional_edges 需要一个 callable —
放 workflow.py 里 build_graph 时闭包捕获更紧凑，所以这里留作 future use。

后续如果要加"基于意图 + 上下文的复杂路由"（比如退款超 24h 才走人工），
可以扩展这个模块。
"""

from agent.state import AgentState


def route_by_intent_and_context(state: AgentState) -> str:
    """
    进阶路由函数（示例，未启用）。

    实际路由逻辑可以比单纯读 state["routed_node"] 更复杂:
    - 看 intent + context 判断是否升级人工
    - 看时间窗口（如退款超 X 小时转人工）
    - 看用户等级（VIP 走专属队列）

    返回: 目标节点名（path_map 的 value）
    """
    # 占位实现 — 跟 _route_after_router 一致
    return state.get("routed_node", "fallback_handler")