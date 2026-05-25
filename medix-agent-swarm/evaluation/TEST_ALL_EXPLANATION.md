# `examples/test_all.py` 测试套件说明

`examples/test_all.py` 是 MediX Agent Swarm 原项目中的综合工程测试套件，用来验证系统核心模块能否按设计协同工作。它更接近“集成测试 + 端到端功能测试”，不是独立医学问答评测基准。

## 代码定位

- 文件路径：`examples/test_all.py`
- 测试数量：26 个测试项
- 执行入口：`python examples/test_all.py`
- 输出形式：终端汇总，并可生成 `TEST_REPORT.md`

## 覆盖范围

| 阶段 | 覆盖模块 | 验证重点 |
|---|---|---|
| Phase 1 | Agent Loop | 简单问题响应、工具调用链路 |
| Phase 2 | Agent Swarm | SharedContext、能力匹配、任务认领、单 Agent / Swarm 路由、SessionSummary |
| Phase 3 | Memory | 短期记忆、Mem0 长期记忆、多轮上下文 |
| Phase 4 | Medical Skills | 生活方式建议、ICD-10 疾病分类、临床指南检索 |
| Phase 5 | DeepResearch | 证据综合、ResearchAgent 集成、端到端研究流程 |
| Phase 7 | Unified Memory | 单 Agent / Swarm 统一记忆、单例模式、去重保存 |
| Phase 8 | Harness Engineering | 约束验证、自动修复、熵管理、非侵入式集成 |

## 它实际验证了什么

1. Agent 是否能完成 Think-Act-Observe 式的工具调用循环。
2. Swarm 是否能根据任务复杂度在单 Agent 和多 Agent 协作之间路由。
3. SharedContext 是否能支撑 Agent 间的事件发布、状态共享和贡献汇总。
4. 各 Agent 是否能注册并自主选择医学 Skills。
5. Milvus 本地知识库是否能支撑生活方式、疾病编码、临床指南等检索型 Skills。
6. 记忆系统是否能保存会话历史，并在多轮追问中提供上下文。
7. Harness 约束系统是否能对工具调用、输出质量、风险提示和上下文熵进行约束。

## 它不等同于什么

- 不等同于医学安全评测：没有覆盖大规模临床病例、真实诊疗标准或专家标注。
- 不等同于模型能力排行榜：没有对多个模型做横向比较。
- 不等同于离线 Eval benchmark：用例写在代码中，主要验证工程链路，不是独立数据集驱动。
- 不等同于生产验收：部分测试依赖外部 API、Mem0、DuckDuckGo 或本地向量库状态。

## 和新增 Eval 的关系

`test_all.py` 证明“项目内部模块能运行、能协作”；新增的 `evaluation/` 目录证明“项目可以被复现地展示和评测”。

| 项目 | `examples/test_all.py` | `evaluation/run_eval.py` |
|---|---|---|
| 目标 | 验证工程模块正确性 | 生成可展示的功能评测报告 |
| 用例来源 | 写在测试代码中 | 来自 `evaluation/datasets/medix_eval_cases.json` |
| 关注点 | 模块、集成、回归 | 简历展示、能力覆盖、可复跑结果 |
| 输出 | 终端结果 / `TEST_REPORT.md` | JSON 结果 / Markdown 评测报告 |
| 适合表述 | “综合测试套件” | “核心功能回归评测集” |

## 简历或面试中的推荐表述

可以这样描述：

> 原项目已有 26 项综合工程测试，覆盖 Agent Loop、Swarm 协作、Memory、Skills、DeepResearch 与 Harness 约束系统。我在此基础上补充了独立的核心功能回归评测集，将测试用例从代码中抽离为 JSON 数据集，并输出可复现的评测报告，便于展示系统在 LLM 接入、工具路由、本地 RAG、急症风险识别和多轮上下文上的端到端表现。

## 运行方式

```powershell
cd "C:\Users\nings\Downloads\宁思源简历_V3_2026-03-20-15_33_30\Medix-agent-swarm\medix-agent-swarm"
.\.venv\Scripts\python.exe -X utf8 examples\test_all.py
```

如只想验证新增 Eval 数据集结构：

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluation\run_eval.py --dry-run
```
