"""
main.py
=======
订单客服智能体 — 意图识别模块入口（v2：create_agent 范式）。

用法:
    # 1. 单轮分类
    python main.py --query "我的订单123456到哪了"

    # 2. 交互式 REPL
    python main.py --interactive

    # 3. 流式输出
    python main.py --query "我要退款" --stream

环境变量:
    AGNES_API_KEY     （必需，.env 已写入）
    AGNES_BASE_URL    （可选，默认 https://apihub.agnes-ai.com/v1）
    AGNES_MODEL       （可选，默认 agnes-2.5-flash）
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)

from langchain_core.messages import HumanMessage

from agent.config import LLM_MODEL
from agent.result_extract import flatten_agent_result
from agent.workflow import get_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("intent-agent")


def _print_result(result: dict) -> None:
    """格式化打印一次意图识别结果"""
    print("\n" + "=" * 60)
    print(f"🎯 识别意图:     {result.get('current_intent', 'N/A')}")
    print(f"📊 置信度:       {result.get('intent_confidence', 0):.2%}")
    print(f"💭 分类理由:     {result.get('intent_reasoning', '')}")
    slots = result.get("extracted_slots", {})
    if slots:
        slot_text = {k: v for k, v in slots.items()}
        print(f"🔑 抽取槽位:     {json.dumps(slot_text, ensure_ascii=False)}")
    if result.get("needs_clarification"):
        print(f"❓ 需要追问:     {result.get('clarification_question')}")
    if result.get("fallback_reason"):
        print(f"⚠️  Fallback:    {result.get('fallback_reason')}")
    print(f"🚏 路由到:       {result.get('routed_node')}")
    print("=" * 60)


async def classify_once(query: str) -> dict:
    """单轮意图识别"""
    agent = get_agent()
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    return flatten_agent_result(result)


async def interactive_loop() -> None:
    """交互式 REPL"""
    print(f"订单客服意图识别 Demo | 模型: {LLM_MODEL} (create_agent 范式)")
    print("输入用户问句,Enter 提交;输入 q / quit / 退出 结束\n")

    while True:
        try:
            query = input("👤 用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 👋")
            return
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit", "退出", "再见"}:
            print("再见 👋")
            return
        try:
            result = await classify_once(query)
            _print_result(result)
        except Exception as e:
            logger.exception("分类失败: %s", e)
            print(f"❌ 出错了: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="订单客服意图识别 — LangChain create_agent (AGNES LLM)"
    )
    parser.add_argument(
        "--query", "-q", type=str,
        help="单条用户问句(与 --interactive 互斥)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="进入交互式 REPL",
    )
    parser.add_argument(
        "--stream", "-s", action="store_true",
        help="流式输出(仅对 --query 生效)",
    )
    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_loop())
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    if args.stream:
        async def _stream() -> None:
            agent = get_agent()
            async for event in agent.astream(
                {"messages": [HumanMessage(content=args.query)]},
            ):
                print(f"\n📡 节点事件: {list(event.keys())}")
                for node_name, node_state in event.items():
                    preview = {
                        k: str(v)[:200]
                        for k, v in (node_state or {}).items()
                    }
                    print(f"  └─ {node_name}: {json.dumps(preview, ensure_ascii=False)[:300]}")
        asyncio.run(_stream())
    else:
        result = asyncio.run(classify_once(args.query))
        _print_result(result)


if __name__ == "__main__":
    main()