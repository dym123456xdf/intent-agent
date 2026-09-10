"""
agent.session
=============
按 session_id 维护多轮对话状态 — 进程内字典，足够本仓库 demo 使用。

设计取舍：
- 进程内 dict（非 Redis/DB）—— 本项目是意图识别层 demo，不需要持久化；
  若日后上生产，按同样的 SessionStore 接口换实现即可。
- 存的是 langchain Messages（HumanMessage / AIMessage / ToolMessage）——
  create_agent 的 MessagesState 天然支持追加，跨请求累积。
- 不在这里调 LLM，只做 CRUD + 拷贝。

类比：session 就是一张草稿纸，每来一轮就把最新 user 输入抄上去，
跑完 agent 再把 agent 的中间产物 + 最终回复抄回来，下一轮继续往上加。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Iterable

from langchain_core.messages import BaseMessage


@dataclass
class Session:
    """一个会话的完整状态"""
    session_id: str
    messages: list[BaseMessage] = field(default_factory=list)
    # 最近一次分类的扁平化结果,前端"详情面板"展示用(同 session 多轮会刷新)
    last_result: dict | None = None


class SessionStore:
    """
    进程内的 session 存储。

    接口刻意保持极简,后面想接 Redis / SQLite 直接换实现即可。
    加 asyncio.Lock 是为了同一 session 并发请求时 messages 追加不丢帧。
    """

    def __init__(self) -> None:
        self._store: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str) -> Session:
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = Session(session_id=session_id)
            return self._store[session_id]

    async def append_messages(self, session_id: str, new_msgs: Iterable[BaseMessage]) -> None:
        async with self._lock:
            s = self._store.setdefault(session_id, Session(session_id=session_id))
            s.messages.extend(new_msgs)

    async def update_last_result(self, session_id: str, result: dict) -> None:
        async with self._lock:
            s = self._store.setdefault(session_id, Session(session_id=session_id))
            s.last_result = result

    async def list_sessions(self) -> list[dict]:
        """
        给前端侧边栏用 —— 按最近活跃倒序返回简要信息。
        title 取首条 user 消息前 20 字,没有就显示 session_id。
        """
        async with self._lock:
            items = list(self._store.values())

        def _summary(s: Session) -> dict:
            title = s.session_id
            for m in s.messages:
                if getattr(m, "type", None) == "human":
                    raw = m.content or ""
                    content = raw.strip().replace("\n", " ") if isinstance(raw, str) else str(raw)
                    if content:
                        title = content[:20] + ("…" if len(content) > 20 else "")
                    break
            # 用最后一条消息的 message id 当伪"时间戳"足够 demo 用
            last_id = getattr(s.messages[-1], "id", None) if s.messages else None
            return {
                "session_id": s.session_id,
                "title": title,
                "message_count": len(s.messages),
                "last_result": s.last_result,
                "last_message_id": last_id,
            }

        items.sort(key=lambda s: len(s.messages), reverse=True)
        return [_summary(s) for s in items]

    async def reset(self, session_id: str) -> bool:
        async with self._lock:
            return self._store.pop(session_id, None) is not None


# 模块级单例 —— FastAPI 进程内复用
STORE = SessionStore()
