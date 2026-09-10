"""
web_app.py
==========
订单客服意图识别 — 网页版入口（FastAPI + SSE）。

启动：
    /opt/anaconda3/envs/langgraph/bin/python -m uvicorn web_app:app --host 127.0.0.1 --8000 --reload

接口：
    GET  /                        聊天 UI 单文件 HTML
    GET  /api/sessions            列出所有 session（侧边栏）
    POST /api/sessions            新建一个 session,返回 session_id
    POST /api/chat                多轮对话（SSE 流式）
    POST /api/reset               清空指定 session

为什么 SSE 不用 WebSocket：
- 单向推流（server → browser）就够用,agent 输出天然单向
- SSE 走 HTTP/1.1,不需要额外协议升级,前端用 EventSource 一行搞定
- 流式看的是 agent 节点事件 + 最终结构化结果,不是双向聊天
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import AsyncIterator

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

load_dotenv(find_dotenv(usecwd=True), override=False)

from agent.result_extract import flatten_agent_result
from agent.session import STORE
from agent.workflow import get_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("web_app")

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(title="Intent Agent Web", version="1.0.0")


# ---------- 静态文件 ----------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ---------- Schemas ----------
class ChatRequest(BaseModel):
    session_id: str | None = None   # None → 自动新建
    message: str


class ResetRequest(BaseModel):
    session_id: str


class NewSessionResponse(BaseModel):
    session_id: str


# ---------- Session 管理 ----------
@app.get("/api/sessions")
async def list_sessions() -> dict:
    return {"sessions": await STORE.list_sessions()}


@app.post("/api/sessions", response_model=NewSessionResponse)
async def new_session() -> NewSessionResponse:
    sid = uuid.uuid4().hex[:12]
    await STORE.get_or_create(sid)
    return NewSessionResponse(session_id=sid)


@app.post("/api/reset")
async def reset_session(req: ResetRequest) -> dict:
    ok = await STORE.reset(req.session_id)
    return {"ok": ok, "session_id": req.session_id}


# ---------- 核心:多轮对话(SSE) ----------
def _sse(event: str, data: dict | str) -> str:
    """包装一个 SSE 事件"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_chat(session_id: str, message: str) -> AsyncIterator[str]:
    """
    SSE 事件流：
      1. event=node       每个 create_agent 节点事件(节点名 + 状态预览)
      2. event=result     最终扁平化分类结果
      3. event=error      异常信息
      4. event=done       收尾(前端关 EventSource)
    """
    session = await STORE.get_or_create(session_id)

    # 把 user 输入写进 session(同 session 下轮继续累加)
    human_msg = HumanMessage(content=message)
    await STORE.append_messages(session_id, [human_msg])

    try:
        agent = get_agent()

        # 把当前 session 的完整 messages 作为输入喂给 agent —— 多轮上下文就这样传
        inputs = {"messages": session.messages}

        final_result: dict | None = None
        async for event in agent.astream(inputs):
            # create_agent.astream() 每个事件是 {node_name: state_delta}
            for node_name, node_state in (event or {}).items():
                preview = {}
                if isinstance(node_state, dict):
                    for k, v in node_state.items():
                        # 把 BaseMessage 列表压成纯文本预览,避免 JSON 序列化炸掉
                        if k == "messages" and isinstance(v, list):
                            preview[k] = [
                                f"{getattr(m, 'type', '?')}: {str(getattr(m, 'content', ''))[:120]}"
                                for m in v
                            ]
                        else:
                            preview[k] = str(v)[:200]
                yield _sse("node", {"node": node_name, "preview": preview})

        # astream 跑完后再调一次 ainvoke 拿 structured_response(它只在终态有值)
        # —— 改:astream 在最后一步也会输出 structured_response,直接从最终 event 取
        # 为保险,这里再跑一次纯 ainvoke,只取 structured_response 字段
        final = await agent.ainvoke(inputs)
        final_result = flatten_agent_result(final)

        # 把 agent 这一轮产出的新 messages 也写回 session,供下一轮累积
        new_msgs = final.get("messages", [])
        await STORE.append_messages(session_id, new_msgs[len(session.messages):])
        await STORE.update_last_result(session_id, final_result)

        yield _sse("result", final_result)
        yield _sse("done", {"session_id": session_id})

    except Exception as e:
        logger.exception("chat failed: %s", e)
        yield _sse("error", {"message": str(e)})
        yield _sse("done", {"session_id": session_id})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(400, detail="message 不能为空")

    sid = req.session_id or uuid.uuid4().hex[:12]
    # 立刻建好 session,避免前端先 POST /api/chat 再 GET /api/sessions 看不到
    await STORE.get_or_create(sid)

    return StreamingResponse(
        _stream_chat(sid, req.message.strip()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # nginx 反代时关掉缓冲
            "X-Session-Id": sid,            # 让前端不用自己生成 session_id
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
