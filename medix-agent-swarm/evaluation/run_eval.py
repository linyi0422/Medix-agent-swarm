#!/usr/bin/env python3
"""
Run a small, reproducible evaluation for the MediX agent swarm demo.

The evaluation focuses on the project paths that matter for a portfolio demo:
DeepSeek connectivity, single-agent routing, safety-oriented triage, local
Milvus RAG, and short-term context use.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))


@dataclass
class EvalCase:
    case_id: str
    title: str
    expected_capability: str
    runner: Callable[[], Awaitable[dict[str, Any]]]
    expected_skills: list[str] = field(default_factory=list)
    must_contain_any: list[str] = field(default_factory=list)
    timeout_seconds: int = 90


class SkillRecorder:
    def __init__(self) -> None:
        self.active_case_id: str | None = None
        self.by_case: dict[str, list[str]] = {}
        self._original = None

    def install(self) -> None:
        from core.skill_registry import SkillRegistry

        self._original = SkillRegistry.execute
        recorder = self

        async def tracked_execute(registry, name: str, **kwargs):
            if recorder.active_case_id:
                recorder.by_case.setdefault(recorder.active_case_id, []).append(name)
            return await recorder._original(registry, name, **kwargs)

        SkillRegistry.execute = tracked_execute

    def set_case(self, case_id: str | None) -> None:
        self.active_case_id = case_id

    def get_skills(self, case_id: str) -> list[str]:
        return self.by_case.get(case_id, [])


def validate_environment() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Set it before running evaluation.")
    if api_key.strip() in {"你的新 DeepSeek Key", "your-api-key", "your-llm-api-key"}:
        raise SystemExit("DEEPSEEK_API_KEY is still a placeholder.")


def prepare_isolated_milvus_db(output_dir: Path) -> None:
    source = PACKAGE_ROOT / "knowledge" / "data" / "milvus_lite.db"
    target = output_dir / "_runtime_milvus_lite.db"

    if not source.exists():
        return

    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(source, target, ignore=shutil.ignore_patterns("LOCK"))
    os.environ["MEDIX_MILVUS_DB_PATH"] = str(target)


async def run_swarm_case(coordinator, question: str, session_id: str) -> dict[str, Any]:
    return await coordinator.process(question, session_id=session_id)


async def run_clinical_guideline_case() -> dict[str, Any]:
    skill_path = PACKAGE_ROOT / ".claude" / "skills" / "clinical-guideline" / "script"
    sys.path.insert(0, str(skill_path))
    from guideline import clinical_guideline

    result = await clinical_guideline("hypertension", max_results=1)
    return {
        "answer": result.get("answer", ""),
        "source": result.get("source"),
        "guideline_title": result.get("guideline_title"),
        "year": result.get("year"),
        "swarm_enabled": False,
        "direct_skill": "clinical_guideline",
    }


async def run_multiturn_case(coordinator) -> dict[str, Any]:
    session_id = "eval-multiturn-context"
    first = await run_swarm_case(coordinator, "我嘴唇一直干裂，可能是什么原因？", session_id=session_id)
    second = await run_swarm_case(coordinator, "那我需要去医院吗？", session_id=session_id)
    return {
        "answer": second.get("answer", ""),
        "first_answer_preview": (first.get("answer") or "")[:260],
        "swarm_enabled": second.get("swarm_enabled"),
        "session_id": session_id,
        "raw": second,
    }


def build_cases(coordinator) -> list[EvalCase]:
    return [
        EvalCase(
            case_id="basic_health_advice",
            title="基础健康咨询",
            expected_capability="DeepSeek 接入 + 单 Agent 健康建议",
            runner=lambda: run_swarm_case(coordinator, "多喝水对健康有什么好处？", "eval-basic-health"),
            must_contain_any=["水", "饮水", "健康"],
        ),
        EvalCase(
            case_id="symptom_lip_cracking",
            title="症状咨询：嘴唇干裂",
            expected_capability="症状分析 + 生活方式建议 + 就医提醒",
            runner=lambda: run_swarm_case(coordinator, "我嘴唇一直干裂，可能是什么原因？", "eval-lip-cracking"),
            expected_skills=["analyze_symptoms", "assess_risk", "search_knowledge"],
            must_contain_any=["唇", "干裂", "口唇"],
        ),
        EvalCase(
            case_id="local_rag_guideline",
            title="本地 RAG：高血压临床指南",
            expected_capability="Milvus 本地知识库 + clinical_guideline Skill",
            runner=run_clinical_guideline_case,
            expected_skills=["clinical_guideline"],
            must_contain_any=["高血压", "指南", "140/90"],
            timeout_seconds=60,
        ),
        EvalCase(
            case_id="emergency_triage",
            title="紧急风险识别：胸痛呼吸困难",
            expected_capability="急症识别 + 安全分诊",
            runner=lambda: run_swarm_case(coordinator, "我胸痛、呼吸困难、出冷汗，严重吗？", "eval-emergency"),
            expected_skills=["assess_risk"],
            must_contain_any=["急", "就医", "120", "急诊"],
        ),
        EvalCase(
            case_id="multiturn_context",
            title="多轮上下文追问",
            expected_capability="短期记忆 + 上下文理解",
            runner=lambda: run_multiturn_case(coordinator),
            must_contain_any=["医院", "就医", "皮肤科", "口腔科"],
            timeout_seconds=150,
        ),
    ]


async def run_case(case: EvalCase, recorder: SkillRecorder) -> dict[str, Any]:
    start = time.perf_counter()
    recorder.set_case(case.case_id)
    try:
        result = await asyncio.wait_for(case.runner(), timeout=case.timeout_seconds)
        status = "pass" if result.get("answer") else "fail"
        error = ""
    except Exception as exc:
        result = {}
        status = "fail"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        recorder.set_case(None)

    elapsed = round(time.perf_counter() - start, 2)
    answer = result.get("answer", "") or ""
    skills = recorder.get_skills(case.case_id)
    direct_skill = result.get("direct_skill")
    if direct_skill and direct_skill not in skills:
        skills.append(direct_skill)

    expected_skill_hit = (
        True
        if not case.expected_skills
        else any(skill in skills for skill in case.expected_skills)
    )
    keyword_hit = (
        True
        if not case.must_contain_any
        else any(keyword in answer for keyword in case.must_contain_any)
    )
    pass_checks = status == "pass" and expected_skill_hit and keyword_hit

    return {
        "case_id": case.case_id,
        "title": case.title,
        "expected_capability": case.expected_capability,
        "status": "pass" if pass_checks else "fail",
        "has_answer": bool(answer),
        "expected_skills": case.expected_skills,
        "skills_called": skills,
        "expected_skill_hit": expected_skill_hit,
        "keyword_hit": keyword_hit,
        "swarm_enabled": result.get("swarm_enabled"),
        "agents_involved": result.get("agents_involved"),
        "elapsed_seconds": elapsed,
        "answer_preview": answer[:700],
        "suggestions": result.get("suggestions", []),
        "error": error,
    }


def render_markdown(results: list[dict[str, Any]], generated_at: str, cold_start_seconds: float) -> str:
    passed = sum(1 for item in results if item["status"] == "pass")
    total = len(results)

    lines = [
        "# MediX Agent Swarm 功能展示与评测报告",
        "",
        f"生成时间：{generated_at}",
        f"评测规模：{total} 个核心场景",
        f"通过情况：{passed}/{total}",
        f"冷启动初始化耗时：{cold_start_seconds}s",
        "",
        "## 评测范围",
        "",
        "- DeepSeek OpenAI-compatible API 接入",
        "- 单 Agent 健康咨询",
        "- 症状咨询与急症风险提醒",
        "- 本地 Milvus 知识库 RAG",
        "- 短期多轮上下文",
        "",
        "## 结果总览",
        "",
        "| Case | 预期能力 | 结果 | 模式 | Skill 调用 | 耗时 |",
        "|---|---|---:|---|---|---:|",
    ]

    for item in results:
        mode = "Swarm" if item.get("swarm_enabled") else "Single/Direct"
        skills = ", ".join(item.get("skills_called") or []) or "-"
        status = "PASS" if item["status"] == "pass" else "FAIL"
        lines.append(
            f"| {item['title']} | {item['expected_capability']} | {status} | {mode} | {skills} | {item['elapsed_seconds']}s |"
        )

    lines.extend(
        [
            "",
            "## 逐项观察",
            "",
        ]
    )

    for item in results:
        lines.extend(
            [
                f"### {item['title']}",
                "",
                f"- 结果：{'通过' if item['status'] == 'pass' else '未通过'}",
                f"- 是否有回答：{item['has_answer']}",
                f"- 预期 Skill 命中：{item['expected_skill_hit']}",
                f"- 关键词校验：{item['keyword_hit']}",
                f"- 耗时：{item['elapsed_seconds']}s",
                f"- 调用 Skill：{', '.join(item.get('skills_called') or []) or '无显式 Skill 调用'}",
            ]
        )
        if item.get("error"):
            lines.append(f"- 错误：{item['error']}")
        if item.get("answer_preview"):
            preview = item["answer_preview"].replace("\n", " ").strip()
            lines.append(f"- 回答片段：{preview}")
        lines.append("")

    lines.extend(
        [
            "## 当前结论",
            "",
            "项目已能展示从 LLM 接入、Agent 路由到本地 RAG 检索的核心闭环。对简历展示而言，最有价值的亮点不是医疗结论本身，而是把开源 Agent 项目跑通、修复运行问题，并建立了可复跑的评测基线。",
            "",
            "## 已知限制",
            "",
            "- Mem0 长期记忆未配置 Key，本次评测只验证短期上下文。",
            "- 复杂 Swarm 问题可能触发 DuckDuckGo 外网搜索，存在限流和 90 秒超时风险。",
            "- 医疗内容只适合产品/工程演示，不应作为真实诊疗建议。",
            "- 当前评测是小样本 smoke/e2e baseline，不等价于医学安全评测。",
            "",
            "## 复跑方式",
            "",
            "```powershell",
            'cd "C:\\Users\\nings\\Downloads\\宁思源简历_V3_2026-03-20-15_33_30\\Medix-agent-swarm\\medix-agent-swarm"',
            '$env:DEEPSEEK_API_KEY="sk-你的真实key"',
            '$env:PYTHONPATH="C:\\Users\\nings\\Downloads\\宁思源简历_V3_2026-03-20-15_33_30\\Medix-agent-swarm;C:\\Users\\nings\\Downloads\\宁思源简历_V3_2026-03-20-15_33_30\\Medix-agent-swarm\\medix-agent-swarm"',
            '$env:USERPROFILE="C:\\Users\\nings\\Downloads\\宁思源简历_V3_2026-03-20-15_33_30\\Medix-agent-swarm\\medix-agent-swarm"',
            ".\\.venv\\Scripts\\python.exe -X utf8 evaluation\\run_eval.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run MediX agent swarm evaluation.")
    parser.add_argument(
        "--output-dir",
        default=str(PACKAGE_ROOT / "eval_results"),
        help="Directory for JSON and Markdown evaluation outputs.",
    )
    args = parser.parse_args()

    validate_environment()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_isolated_milvus_db(output_dir)

    recorder = SkillRecorder()
    recorder.install()

    print("Initializing shared SwarmCoordinator...")
    cold_start = time.perf_counter()
    from swarm import SwarmCoordinator

    coordinator = SwarmCoordinator(enable_swarm=True)
    cold_start_seconds = round(time.perf_counter() - cold_start, 2)
    print(f"Shared SwarmCoordinator initialized in {cold_start_seconds}s")

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []
    for case in build_cases(coordinator):
        print(f"Running {case.case_id}: {case.title}")
        results.append(await run_case(case, recorder))

    json_path = output_dir / "medix_eval_results.json"
    report_path = output_dir / "MEDIX_EVAL_REPORT.md"

    json_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "summary": {
                    "total": len(results),
                    "passed": sum(1 for item in results if item["status"] == "pass"),
                    "failed": sum(1 for item in results if item["status"] != "pass"),
                    "cold_start_seconds": cold_start_seconds,
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(results, generated_at, cold_start_seconds), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
