# LLM Agent 范式综述（2026 主流）

> 本文件梳理 2026 年 LLM Agent 领域的主流范式分类，并对照本项目 `intent-agent` 的实现，帮助在工程上选择合适的工作流组织方式。
>
> 适用范围：基于大语言模型、需要调用工具 / 多步决策 / 长时任务的 Agent 系统。
>
> 综述依据：吴恩达 DeepLearning.AI《Agentic AI》课程（2025）、Future AGI《LLM Agent Architectures in 2026》、EasyClaw《AI Agent Architecture in 2026》及多份 2025~2026 综述。

---

## 一、为什么需要"范式"

LLM 本身是单次「prompt → completion」的函数。要让 LLM 完成"查订单 → 调退款 API → 检查物流 → 回复用户"这种多步任务，必须有一套**编排协议**决定：

- 何时调工具（tool calling）
- 何时停下来反思（reflection）
- 何时把控制权交回用户 / 外部系统
- 如何在出错时恢复

"范式"就是这些编排协议的抽象模板。2026 年的主流叙事按**三个不同维度**分类范式，理解维度差异比记住具体名字更重要：

| 维度 | 关注什么 | 典型分类 |
|------|---------|---------|
| **设计模式** | Agent 内部具备的能力 | Reflection / Tool Use / Planning / Multi-Agent（吴恩达 4 大） |
| **推理模式** | Planner 怎么拆解目标 | ReAct / Plan-and-Execute / Reflexion / Tree-of-Thoughts / Multi-Agent Orchestration |
| **编排模式** | 多 Agent / 跨 Agent 怎么协作 | 单 Agent ReAct / Supervisor-Worker / Hierarchical / Peer-to-Peer Mesh |

下文按"推理模式 → 设计模式 → 编排模式 → 速查表 → 趋势 → 本项目定位"组织。

---

## 二、推理模式（Planner 层 — Agent 怎么"想"）

> 这一层决定 Agent 内部每一步的"思考-行动"循环。来源：Future AGI《LLM Agent Architectures in 2026》。

### 1. ReAct（Reason + Act）— 推理-行动循环

**核心思想**：LLM 在每一步都先"想"（Thought），再"做"（Action），再"看"（Observation），循环往复直到完成。

```
Thought 1 → Action 1 → Observation 1 → Thought 2 → Action 2 → Observation 2 → ... → Final Answer
```

**关键组件**：
- Thought：链上推理（CoT），可显式输出
- Action：调用工具（API / 检索 / 计算器）
- Observation：工具返回结果，反馈给 LLM 作为下一轮上下文

**优点**：
- 简单直观，prompt 模板短
- 适合**短链路**任务（≤ 5 步工具调用）
- 可解释性强：每一步 thought 都可观测

**缺点**：
- 上下文会指数膨胀（每步 thought + observation 都塞进 prompt）
- 没有"全局规划"，长任务容易走偏
- 没有"反思 / 修正"机制，错了就继续错

**代表工作**：Yao et al. 2022《ReAct: Synergizing Reasoning and Acting in Language Models》（arXiv:2210.03629，ICLR 2023 oral）；LangChain 早期 `AgentExecutor` 默认模式；本项目 `nodes.classify_intent` 单步 LLM 调用本质就是退化版 ReAct（只有 Thought + Action，没有 Observation 反馈环）。

**典型场景**：单轮客服问答、SQL 生成、简单数据抽取、FAQ 检索。

---

### 2. Plan-and-Execute — 先规划后执行

**核心思想**：把任务拆成两阶段——**先让 LLM 一次性规划出完整步骤列表，再逐步执行**。中间步骤用确定性代码执行（不调 LLM），只在必要时回 LLM 调整。

```
Planner (LLM) → [step1, step2, step3, ...] → Executor (code/tool) → result → Replanner? → ...
```

**关键组件**：
- Planner：LLM 产出 DAG / 序列计划
- Executor：每个 step 是确定性函数（调 API / 查 DB / 跑代码）
- Replanner（可选）：每 N 步让 LLM 重新审视计划

**优点**：
- **token 省**：中间步骤不调 LLM，长任务成本可控
- **可控性强**：业务关键路径用代码写死，LLM 只做"决策"
- **可观测性优**：每个 step 的输入输出都是结构化记录
- **适合生产**：把 LLM 当"大脑"而非"手脚"

**缺点**：
- 计划阶段一次性产出 → 错了一步可能整条链路废
- Replanner 频率难调：调太频繁 = 退化成 ReAct；调太低 = 错后无救
- 不适合"边做边学"的探索类任务

**代表工作**：LangGraph 的 Plan-and-Execute cookbook；本项目 `workflow.py` 就是该范式的精简版——`classify_intent` = Planner（一次性输出意图 + 槽位），`intent_router` + 后续 handler = Executor（纯逻辑路由 / 业务调用）。

**典型场景**：客服工单处理、RPA 自动化、数据分析 pipeline、CI 流水线、研究助手（搜 → 读 → 写）。

---

### 3. Reflexion — 反思-修正循环（带记忆）

**核心思想**：Agent 不只产出结果，还产出"对自己结果的批评"，把批评**写进 episodic memory**，下次基于记忆重试。常配 ReAct 使用。

```
Attempt → Score → Verbal Critique → Store → Attempt (with critique in context) → ... → Final
```

**关键组件**：
- Actor：执行任务（通常是 ReAct 循环）
- Self-Evaluator：对结果打分（成功 / 失败 / 改进点）
- Verbal Critic：把失败转成自然语言反思（"我应该先查询 X 再调 Y"）
- Episodic Memory Buffer：把反思存起来，下次任务时检索相关反思注入 context

**与 Reflection 的关键差异**：Reflection 是"对自己当前输出做批评修订"（无记忆），Reflexion 是"跨尝试把反思持久化到记忆中"。

**优点**：
- 输出质量显著高于单轮（代码任务 +10~30% pass@1）
- 跨任务可复用反思（不是每任务从零开始）

**缺点**：
- **成本翻倍**：每轮至少 2 次 LLM 调用 + 记忆检索
- Critic LLM 自己也可能错（幻觉批评）
- 终止条件难设：可能 100 轮还"不满意"

**代表工作**：Shinn et al. 2023《Reflexion: Language Agents with Verbal Reinforcement Learning》（arXiv:2303.11366，NeurIPS 2023）。

**典型场景**：多轮调试类任务（代码生成、SWE-Bench）、长时博弈、跨会话学习。

---

### 4. Tree-of-Thoughts（ToT）— 树状搜索 + 回溯

**核心思想**：把 ReAct 的"线性思考链"扩展为**树状搜索**——每一步展开多个候选分支，对中间状态打分，剪掉低分路径，对高分路径继续展开。失败可回溯。

```
                Thought A (score 0.8)
               /                         \
        Thought A1 (0.7)               Thought A2 (0.9)
        /         \                       /          \
       ...         ...                ...           ...
       ↓           ↓                   ↓            ↓
    prune 0.3   keep 0.7           prune 0.2      expand
```

**关键组件**：
- Thought Generator：每个状态生成 K 个候选下一步
- State Evaluator：对每个候选状态打分（LLM 打分 / 启发式）
- Search Algorithm：BFS / DFS / Beam Search 选最优路径
- Backtrack：发现死胡同时回溯上一层

**优点**：
- 能处理**需要试错 + 推理**的任务（数学证明、逻辑谜题、24 点游戏）
- 比 ReAct 鲁棒（线性链一错全错，树可绕路）

**缺点**：
- **成本极高**：每步展开 K 个分支 = token × K
- 需要能"打分"的中间状态（很多任务打分函数难写）
- 实时性差（不适合实时对话）

**代表工作**：Yao et al. 2023《Tree of Thoughts: Deliberate Problem Solving with Large Language Models》（arXiv:2305.10601，NeurIPS 2023）。

**典型场景**：数学竞赛题、逻辑推理、24 点游戏、密码破译、约束规划。

---

### 5. Multi-Agent Orchestration（推理视角）— 多 Agent 协作推理

**核心思想**：把任务拆给**多个专精 Agent**（每个有自己的 system prompt / 工具集 / 上下文），通过**消息总线**协作。每个子 Agent 内部仍是 ReAct / Plan-and-Execute / Reflexion。

这一层与下方"编排模式"是同一对象的不同视角：编排模式关注**拓扑结构**，推理视角关注**每个 Agent 内部的推理方式**。本节已在编排模式中展开，此处不重复。

---

## 三、设计模式（Agent 内部具备的能力 — 吴恩达分类）

> 来源：吴恩达 DeepLearning.AI《Agentic AI》课程。设计模式不是"选一种用"，而是**叠加组合**——一个真实 Agent 通常同时具备其中 2~4 种能力。

### 1. Reflection（反思）

**核心**：Agent 对**自己的当前输出**做评估与修订，单轮 / 多轮内闭环，不需要外部记忆。

```
Draft → Critique → Revise → Critique → Revise → Final
```

与 Reflexion 的区别：Reflection 不跨任务持久化反思。

**代表工作**：Self-Refine（Madaan et al. 2023, arXiv:2303.17651）；Constitutional AI（Anthropic 2022.12, arXiv:2212.08073, 用一组"宪法原则"做 critique 标准）。

**典型应用**：代码生成后自检、文案润色、输出格式校验。

---

### 2. Tool Use（工具使用）

**核心**：LLM 通过 function calling / tool schema 调用外部工具（API、DB、计算器、检索）。详见下文"Tool-Use First 范式"。

**典型应用**：结构化数据查询、API 调用、IDE 编程助手。

---

### 3. Planning（规划）

**核心**：把复杂任务拆成步骤序列 + 子目标。ReAct / Plan-and-Execute / ToT 都是 Planning 的具体实现方式。

**典型应用**：多步研究任务、自动化流程、CI 流水线。

---

### 4. Multi-Agent Collaboration（多 Agent 协作）

**核心**：多个专精 Agent 通过显式协议协作完成任务。详见下文"编排模式"。

**典型应用**：软件研发全流程、市场分析、多文档综述。

---

## 四、Tool-Use First — 工具调用范式（横切关注点）

**核心思想**：LLM 推理能力已强到**直接选工具**——不需要显式 Thought / 推理循环。模型原生支持 function calling schema（OpenAI / Anthropic / DeepSeek / Qwen / MiniMax 都内置），由模型自主决定何时调、调哪个、参数是什么。

```
User Query → LLM (with tool schemas) → tool_call(JSON) → execute → tool_result → LLM → ... → answer
```

**关键组件**：
- 工具 schema：OpenAPI / JSON Schema 描述
- Runtime：解析 tool_call → 执行 → 回填结果
- **MCP（Model Context Protocol）**：Anthropic 2024.11.25 发布，作为"AI 工具的 USB-C 接口"标准化跨厂商工具注册与调用。2025 年起 OpenAI、Google、Microsoft、IDE 厂商相继接入，目前是 Tool-Use 范式的生态事实标准。

**优点**：
- **延迟低**：少一轮 Thought
- **生态成熟**：LangChain / LlamaIndex / LangGraph 都深度集成
- **MCP 标准化**：工具可跨模型复用

**缺点**：
- 强依赖模型的 function calling 能力（小模型可能选错工具）
- 工具 schema 写得烂 = 模型调不对
- 本身不提供"反思"机制（需叠加 Reflection 设计模式）

**代表工作**：OpenAI Function Calling（2023 GPT-4 引入，2024 Assistants API 全面支持）、Anthropic Tool Use、MCP（Model Context Protocol）、Vercel AI SDK。本项目 `agent.llm.get_llm()` 走的就是这条路（MiniMax API 兼容 OpenAI function calling）。

**典型场景**：结构化数据查询、API 调用、IDE 编程助手（Cursor / Cline）、企业内部知识库问答。

> Tool-Use 在 2026 主流分类中既可单列为范式，也可作为"设计模式 Tool Use"的工程化形态。它本身不是一个完整的 Agent 形态，而是其他范式的**能力底座**。

---

## 五、编排模式（多 Agent / 跨 Agent 怎么协作）

> 来源：EasyClaw《AI Agent Architecture in 2026》、Future AGI Multi-Agent 系列。适用于"任务需要多个专精角色"或"跨团队 / 跨系统的复杂流程"。

### 1. 单 Agent ReAct（最简）

**形态**：一个 LLM + 一个推理循环 + 一组工具，没有编排层。

```
User → [LLM + ReAct Loop] → Tools → Response
```

- **适合**：聚焦、范围明确的任务（研究摘要、数据抽取、单域 Q&A）
- **延迟**：最低
- **失败风险**：context 饱和、长任务卡死

---

### 2. Supervisor-Worker（主管-工人）— 2026 生产部署最广泛

**形态**：一个 Supervisor Agent 拆任务、分发给多个专精 Worker Agent，聚合结果。

```
User → Supervisor Agent
        ├── Worker A (Research)
        ├── Worker B (Writing)
        └── Worker C (Review)
```

- **适合**：电商内容流水线、客服分流、研究 → 写作 → 审核流水线
- **LangGraph 与 OpenAI Agents SDK 都原生支持**（图边 + handoff 机制）
- **Supervisor 持有共享 state**，workers 读写该 state

---

### 3. Hierarchical（层级化）— 企业级

**形态**：Supervisor 还有 Supervisor，多层嵌套，适合**跨部门 / 大规模企业流程**。

```
Orchestrator
├── Team Supervisor A → [Worker, Worker, Worker]
└── Team Supervisor B → [Worker, Worker, Worker]
```

- **适合**：法律文档处理、大规模 DevOps 自动化、跨部门企业流程
- **核心挑战**：可观测性 — 调试 4 层深度的失败需要结构化 tracing

---

### 4. Peer-to-Peer Mesh（对等网状）— 2026 新兴

**形态**：没有中心 Supervisor，Agents 通过共享消息总线 / blackboard 自主发现、协商、分工。

```
Agent A ↔ Agent B ↔ Agent C
  ↕         ↕         ↕
Agent D   Agent E
```

- **适合**：仿真环境、前置任务结构未知的研究 pipeline、动态联盟
- **成熟度**：受 AG2 / AutoGen group chat 推动，**生产可用仅限受控领域**，不建议直接面向 C 端用户

---

### 5. Plan-then-Execute（编排视角）

**形态**：先用一个 Planner Agent 产出完整步骤图，再分配给 Executor Agents 执行。可与 Supervisor-Worker 组合。

- **适合**：可前置分解的长任务（软件研发 CI、数据 ETL、多文档综述）

---

## 六、范式选择速查表

| 任务特征 | 推荐范式 | 理由 |
|---------|---------|------|
| 单轮问答 / 检索 | ReAct 或 Tool-Use | 简单直接 |
| 短工具链（≤ 5 步） | ReAct + Tool-Use | 灵活 + 可解释 |
| 长链路业务流程 | Plan-and-Execute | 省 token + 可控 |
| 高质量输出（代码 / 论文） | Reflection / Reflexion | 多次打磨提质量 |
| 需要试错回溯的任务 | Tree-of-Thoughts | 搜索 + 回溯 |
| 多角色协作 / 团队任务 | Supervisor-Worker / Multi-Agent | 职责清晰 |
| 跨工具 / 跨系统集成 | Tool-Use + MCP | 标准化生态 |
| 跨部门企业流程 | Hierarchical | 嵌套编排 |
| 仿真 / 探索研究 | Peer-to-Peer Mesh | 动态联盟 |

---

## 七、2026 年趋势观察

> 注：以下趋势综合 2025~2026 多家框架对比综述与社区共识，部分判断仍处演变中，仅供参考。

1. **推理模型（Reasoning Model）模糊范式边界**：o-series（OpenAI o1 2024.09、o3 2024.12）、DeepSeek-R1（2025.01 开源）、Claude with extended thinking 在模型内部就做了反思 + 多步推理，外面看就是"单次调用"。传统显式多轮 Reflection 的优势在被压缩——但 Plan-and-Execute / Hierarchical 仍是长任务首选，因为外部可控性 ≠ 模型内部推理能完全替代。

2. **MCP 标准化工具生态**：Anthropic 2024.11.25 发布 Model Context Protocol 后，2025 年起 OpenAI、Google、Microsoft、IDE 厂商相继接入。Tool-Use 范式的工程摩擦在快速下降，是 Tool-Use 段最重要的基础设施变化。

3. **LangGraph 在生产场景占优，CrewAI/AutoGen 在快速原型场景占优**：
   - LangGraph 的"显式状态机 + 条件边"在**生产级可控性**上领先（可观测、可调试、可恢复），适合复杂业务流程
   - CrewAI 的"角色+任务"心智模型最直观，**快速原型 / 单次脚本**场景仍是最快路径
   - AutoGen 的"对话协商"适合**研究 / 探索**类任务，但生产部署摩擦最大
   - OpenAI 2025 年推出 Agents SDK（原 Swarm 实验项目 2026 初正式归档后转为正式 SDK），定位介于 CrewAI 与 LangGraph 之间
   - Microsoft 2025 年发布 Agent Framework（统一 Semantic Kernel + AutoGen 路线）

4. **多 Agent 协作的争议仍未定论**：2024~2025 一度"万 Agent 皆可"，2026 年工程共识是——**单 Agent + 好工具链 + 显式状态机**往往比 5 个弱 Agent 协作更强；但**职责清晰的 Supervisor-Worker 多 Agent**（如 PM/Dev/QA 软件研发流）仍是有效模式。"小任务单 Agent，跨职能任务多 Agent"。

5. **Computer Use / GUI Agent 是最大变量**：2024.10 Anthropic 首发 Claude Computer Use（公测），2025.01 OpenAI Operator（基于 CUA 模型），2025.10 Google Gemini 2.5 Computer Use、OpenAI ChatGPT Atlas 浏览器代理。把"工具调用"从 API 扩展到 GUI，是 2026 年最重要的范式增量——传统 SaaS / 旧系统无需 API 也能被 Agent 操作。

6. **显式工作流 vs. Agentic 的成本之争**：2025 年起社区开始反思"过度 Agentic"的隐性成本（调试难、延迟高、不可预测），部分企业回归**显式 DAG 工作流**（LangGraph、Dify 等）。"能不用 Agent 就不用 Agent，需要 Agent 时用最简的范式"成为新工程原则。EasyClaw 2026 报告："40% 的 agent 项目失败是因为过度设计"。

7. **可观测性成为基础设施**：OpenTelemetry 兼容的 tracing 在 2026 成为 agent runtime 的标配（不是事后加的）。LangSmith、Future AGI traceAI、Mem0/Letta/Zep 等专用记忆层成熟为独立产品。

---

## 八、本项目（intent-agent）的范式定位

本项目 `intent-agent` 是 **Plan-and-Execute + Tool-Use 的精简版**：

- **Planner 节点**：`classify_intent` —— 一次 LLM 调用，产出意图 + 置信度 + 槽位 + 是否需要追问（结构化输出，Pydantic 校验）。
- **纯逻辑 Executor**：`intent_router` —— 不调 LLM，按规则（needs_clarification / confidence / intent code）路由到对应 handler。
- **下游 Handler**：每个业务 handler 占位（`order_query_handler` / `refund_handler` 等），实际项目里会调业务 API / DB。

**优点**：意图分类这一"决策"环节用 LLM（需要语言理解），后续路由 + 执行用代码（确定 + 可测 + 省 token）。

**可演进方向**：
- 业务 handler 接上 ReAct（每个 handler 内部允许多步 tool call）
- 引入 Reflection / Reflexion 节点校验回复质量
- handler 之间用 LangGraph 子图嵌套 → 升级到 Supervisor-Worker / Hierarchical 编排模式
- 接入 MCP，让 handler 工具可被外部复用
- Tree-of-Thoughts 用在意图歧义消解（多条合理路径竞争时）

---

## 九、参考资源

### 核心论文

- Yao et al. 2022 — *ReAct: Synergizing Reasoning and Acting in Language Models* (arXiv:2210.03629, ICLR 2023)
- Shinn et al. 2023 — *Reflexion: Language Agents with Verbal Reinforcement Learning* (arXiv:2303.11366, NeurIPS 2023)
- Yao et al. 2023 — *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (arXiv:2305.10601, NeurIPS 2023)
- Madaan et al. 2023 — *Self-Refine: Iterative Refinement with Self-Feedback* (arXiv:2303.17651, NeurIPS 2023)
- Bai et al. 2022 — *Constitutional AI: Harmlessness from AI Feedback* (arXiv:2212.08073, Anthropic)

### 课程与综述

- Andrew Ng — *Agentic AI* 课程（DeepLearning.AI，2025）
- Future AGI — *LLM Agent Architectures in 2026* (futureagi.com)
- EasyClaw — *AI Agent Architecture in 2026: Patterns, Frameworks & Production Deployment*
- LangChain Plan-and-Execute Cookbook

### 框架与生态

- LangGraph Multi-Agent 文档
- Anthropic MCP — https://modelcontextprotocol.io/
- OpenAI Agents SDK（2026 从 Swarm 正式化）
- Microsoft Agent Framework（2025）

---

> 最后更新：2026-09 · 与 `intent-agent` 项目当前实现对齐 · 基于 2025~2026 多源综述整理