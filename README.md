# Intent Agent — 订单客服意图识别

> 基于 LangGraph 状态机的订单系统在线客服意图识别引擎，覆盖 LangGraph StateGraph / Tool Calling / 状态机 / 异常降级等核心 Agent 能力。

## 业务背景

订单系统在线客服每天面对海量用户问句：「我的订单到哪了」「我要退款」「怎么开发票」「双十一有什么活动」……

如果让大模型直接生成回复，会出现：
- **意图识别漂移** —— 用户问订单状态，模型跑去讲优惠活动
- **幻觉订单号** —— 用户没给订单号，模型自己编一个
- **多轮对话失忆** —— 用户上一轮说"订单123456"，这一轮问"它发货了吗"，模型问"哪个订单？"

**本项目是这套客服系统的第一层 —— 意图识别层**。识别完之后，下游的 Tool Calling 层（业务查询 / 退款 / 物流 / 支付）才能精准执行。

## 核心能力

### 1. 八大类意图分类

| 编码 | 名称 | 典型场景 |
|---|---|---|
| `ORDER_QUERY` | 订单查询 | 「帮我查订单123456详情」 |
| `LOGISTICS_DELIVERY` | 物流配送 | 「我的快递到哪了」 |
| `REFUND_AFTER_SALE` | 退款售后 | 「我要退款」 |
| `PAYMENT_ISSUE` | 支付问题 | 「支付失败怎么办」 |
| `ACCOUNT_MEMBER` | 账户会员 | 「我的积分怎么用」 |
| `PROMOTION_COUPON` | 优惠活动 | 「双十一有什么优惠」 |
| `COMPLAINT_SUGGESTION` | 投诉建议 | 「我要投诉客服态度」 |
| `CHITCHAT_GREETING` | 闲聊寒暄 | 「你好」 |

每类意图在 `data/intents.json` 维护：
- 编码 + 中文名 + 详细描述
- 典型示例（注入 prompt 做 few-shot）
- 必需槽位（订单号 / 商品号等）

### 2. 槽位抽取 (Slot Filling)

从用户语句中识别 `order_id` / `product_id` / `user_id` 等关键参数，结构化输出供下游 Tool 调用。

### 3. 多轮对话支持

通过 `Annotated[list, add_messages]` reducer + LangGraph 的 `messages` placeholder，支持多轮上下文注入，LLM 可参考历史对话做分类。

### 4. 异常降级

LLM 调用异常 / schema 校验失败 / 超时 → 自动回退到 `FALLBACK_UNKNOWN` 路由，避免用户问句被吞掉。

### 5. 置信度门控 + 追问策略

- 置信度 < 阈值 (默认 0.6) → 走 fallback 兜底
- LLM 主动判定信息缺失 → 自动生成追问问题 (`needs_clarification=true`)

## 技术栈

| 维度 | 选型 | 理由 |
|---|---|---|
| **编排框架** | LangGraph 1.2.11 StateGraph | 状态机模式适合多意图 + 路由 + 降级链路 |
| **LLM** | MiniMax-M3（兼容 OpenAI 协议） | 公司主用 MiniMax 模型族；OpenAI 协议可平替 DeepSeek / 通义 / OpenAI |
| **结构化输出** | Pydantic v2 + PydanticOutputParser | 强类型 + schema 校验，LLM 输出幻觉可被拦截 |
| **Python** | 3.13.15 | conda env `/opt/anaconda3/envs/langgraph/bin/python` |
| **测试** | pytest 9.x + pytest-asyncio | 离线（路由逻辑）+ E2E（真 LLM 调用）双层覆盖 |

## 项目结构

```
intent-agent/
├── main.py                       # CLI 入口（单轮 / 交互 / 流式）
├── README.md                     # 本文档
├── agent/
│   ├── __init__.py
│   ├── config.py                 # LLM/阈值/路径配置
│   ├── models.py                 # Pydantic schema（IntentClassification 等）
│   ├── state.py                  # LangGraph AgentState TypedDict
│   ├── prompts.py                # 意图分类 prompt + JSON 约束
│   ├── llm.py                    # ChatOpenAI MiniMax 客户端工厂
│   ├── nodes.py                  # classify_intent / intent_router 节点
│   ├── workflow.py               # StateGraph 编排
│   └── routes.py                 # 进阶路由函数（预留）
├── data/
│   └── intents.json              # 8 大意图定义（可扩展）
└── tests/
    └── test_intent.py            # 20 个测试用例（离线 14 + E2E 5 + 降级 2）
```

## LangGraph 拓扑

```
START
  ↓
classify_intent       ← 调 LLM 做意图分类
  ↓
intent_router         ← 纯逻辑路由（置信度门控 + 追问优先级 + intent→node 映射）
  ├─→ fallback_handler          (低置信度 / 未知意图)
  ├─→ clarification_handler     (信息缺失)
  ├─→ order_query_handler       (ORDER_QUERY)
  ├─→ refund_handler            (REFUND_AFTER_SALE)
  ├─→ payment_handler           (PAYMENT_ISSUE)
  ├─→ logistics_handler         (LOGISTICS_DELIVERY)
  ├─→ account_handler           (ACCOUNT_MEMBER)
  ├─→ promotion_handler         (PROMOTION_COUPON)
  ├─→ complaint_handler         (COMPLAINT_SUGGESTION)
  └─→ chitchat_handler          (CHITCHAT_GREETING)
```

每个 `*_handler` 当前是占位 stub，本仓库只交付**意图识别层**。下游 Tool Calling / 多轮对话管理 / 流式输出将在后续迭代中补齐。

## 快速开始

### 1. 环境

```bash
# 用 langgraph conda env（已有 langgraph 1.2.11 / langchain 1.3.18 / langchain-openai 1.6.0）
/opt/anaconda3/envs/langgraph/bin/python --version

# 安装 pytest + pytest-asyncio（如未装）
/opt/anaconda3/envs/langgraph/bin/pip install pytest pytest-asyncio
```

### 2. 配置环境变量（已写入 ~/.zshrc）

```bash
export MINIMAX_API_KEY="sk-cp-..."
export MINIMAX_BASE_URL="https://api.minimaxi.com/v1"
export MINIMAX_MODEL="MiniMax-M3"  # 可选，默认就是这个
```

### 3. CLI 用法

```bash
# 单轮分类
python main.py --query "我的订单123456到哪了"

# 交互式 REPL
python main.py --interactive

# 流式输出（每个节点事件）
python main.py --query "我要退款" --stream
```

### 4. 跑测试

```bash
# 全部测试（20 个，离线 15 + E2E 5）
/opt/anaconda3/envs/langgraph/bin/python -m pytest tests/ -v

# 只要离线测试（不需要 LLM）
/opt/anaconda3/envs/langgraph/bin/python -m pytest tests/ -v -m "not skipif"
```

## 设计取舍说明

### 为什么意图分类走 LangGraph 而不是单条 Chain？

意图识别是 LangGraph 的**标准入门应用**：

1. **可观测性**：每个节点的状态写入都可被 LangSmith 追踪
2. **可扩展性**：以后想加"意图二次校验" / "敏感词过滤" / "VIP 优先"，加节点即可
3. **多入口复用**：同一份 StateGraph 既能服务 API 调用，也能服务异步批处理

### 为什么路由拆成独立节点而不是在 classify_intent 里直接返回节点名？

1. **解耦**：分类是 LLM 任务（耗时、易变）；路由是纯逻辑（毫秒、确定）。混在一起会让单测难写
2. **可替换**：未来想加 LLM replanner 或基于上下文的复杂路由，只换 intent_router 这一个节点
3. **trace 清晰**：LangSmith 上能看到独立的"分类 → 路由"两步决策

### 为什么用 Pydantic 而不用 TypedDict？

- LangGraph **State** 用 TypedDict（langgraph 的设计）
- LLM **结构化输出** 用 Pydantic（langchain-output-parser 的设计）
- 二者职责不同，不冲突

### 为什么置信度阈值默认 0.6？

- 实测 MiniMax-M3 在典型 query 上 confidence ≥ 0.85
- 0.6 是"宁可多走 fallback 也别乱路由"的保守阈值
- 业务上线后可按真实 query 分布调优（参考 `INTENT_CONFIDENCE_THRESHOLD` env var）

## 已验证的关键场景

| 场景 | 用户输入 | 识别结果 | 路由 |
|---|---|---|---|
| 订单详情 | 「帮我查一下订单123456的详情」 | ORDER_QUERY (95%+) | order_query_handler |
| 物流追踪 | 「我的订单123456到哪了」 | LOGISTICS_DELIVERY (95%+) | logistics_handler |
| 退款申请 | 「我要申请退款，订单号是888999」 | REFUND_AFTER_SALE (98%+) | refund_handler |
| 闲聊 | 「你好」 | CHITCHAT_GREETING (98%+) | chitchat_handler |
| 信息缺失 | 「我的订单到哪了」 | LOGISTICS_DELIVERY + needs_clarification | clarification_handler |
| LLM 异常 | （mock 异常） | FALLBACK_UNKNOWN | fallback_handler |

## 后续 Roadmap

- [ ] Tool Calling 层（订单查询 / 退款 / 物流 API 实际调用）
- [ ] 多轮对话状态管理（user_id / session 持久化）
- [ ] LangSmith trace 接入（生产可观测性）
- [ ] LangServe 部署（HTTP / SSE 流式）
- [ ] RAG 增强（订单系统知识库召回）

## 项目背景与定位

本仓库聚焦于订单系统在线客服智能体的**意图识别层**。完整的智能客服系统通常需要：

1. **意图识别**（本仓库）—— 理解用户在问什么
2. **Tool Calling** —— 调业务 API（订单查询 / 退款 / 物流）
3. **多轮对话管理** —— 用户身份 / session 持久化
4. **流式输出** —— SSE / WebSocket 实时响应
5. **可观测性** —— LangSmith trace / 指标埋点

当前实现为意图识别第一阶段，后续将按上述 5 个方向迭代。