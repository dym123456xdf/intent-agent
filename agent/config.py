"""
agent.config
============
集中管理 LLM / 模型 / 阈值等配置。所有配置走环境变量 + 默认值。

为什么不用 .env 直读？
- 项目目录没有 .env，直接用环境变量更轻量
- 环境变量在 ~/.zshrc 已经 export 过 MINIMAX_API_KEY / MINIMAX_BASE_URL
- 测试时方便临时覆盖：MINIMAX_MODEL=MiniMax-M2-highspeed pytest
"""
import os
from pathlib import Path

# 项目根目录: ~/PycharmProjects/intent-agent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# —— LLM 配置 ——
# 兼容 OpenAI 协议：MiniMax / DeepSeek / 通义千问 / OpenAI 都走 ChatOpenAI
LLM_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
LLM_API_KEY = os.getenv("MINIMAX_API_KEY", "")
LLM_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M3")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))  # 意图分类要稳定 → 0
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))  # 秒

# —— 意图识别阈值 ——
# confidence < INTENT_CONFIDENCE_THRESHOLD → 走 fallback
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.6"))

# —— 数据路径 ——
INTENTS_FILE = PROJECT_ROOT / "data" / "intents.json"