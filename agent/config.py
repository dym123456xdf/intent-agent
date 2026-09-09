"""
agent.config
============
集中管理 LLM / 模型 / 阈值等配置。所有配置从 .env 读 + 环境变量兜底。

为什么走 .env 而不是直接 os.getenv？
- 项目根目录已有 .env，单一来源管理更可控
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录: ~/PycharmProjects/intent-agent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载 .env（override=False：保留进程已有 env var 的优先级）
_DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)

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
