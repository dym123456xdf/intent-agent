"""
agent.nodes
============
v2 改造后,本文件不再承载核心节点逻辑 — create_agent 内部用一个固定 ReAct loop 取代了
classify_intent / intent_router 这两个节点。

保留此文件作为占位,方便对照历史 commit（3857603 之前）的代码结构。
新代码请走 agent.workflow.get_agent()。
"""

# v1 时代的 classify_intent / intent_router / _fallback_result 已删除。
# create_agent 内部已实现相同能力（tool-calling loop + response_format 校验）。
# 历史实现见 git log 9680f08 之前的版本。