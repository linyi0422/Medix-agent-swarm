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
DEFAULT_DATASET_PATH = PACKAGE_ROOT / "evaluation" / "datasets" / "medix_eval_cases.json"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))


@dataclass
class EvalCase:
    case_id: str
    title: str
    case_type: str
    expected_capability: str
    runner: Callable[[], Awaitable[dict[str, Any]]]
    expected_skills: list[str] = field(default_factory=list)
    must_contain_any: list[str] = field(default_factory=list)
    evaluation_focus: list[str] = field(default_factory=list)
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


async def run_direct_skill_case(skill_name: str, skill_args: dict[str, Any]) -> dict[str, Any]:
    if skill_name != "clinical_guideline":
        raise ValueError(f"Unsupported direct skill case: {skill_name}")

    skill_path = PACKAGE_ROOT / ".claude" / "skills" / "clinical-guideline" / "script"
    sys.path.insert(0, str(skill_path))
    from guideline import clinical_guideline

    result = await clinical_guideline(
        skill_args.get("disease", "hypertension"),
        max_results=skill_args.get("max_results", 1),
    )
    return {
        "answer": result.get("answer", ""),
        "source": result.get("source"),
        "guideline_title": result.get("guideline_title"),
        "year": result.get("year"),
        "swarm_enabled": False,
        "direct_skill": skill_name,
    }


async def run_multiturn_case(coordinator, prompts: list[str], session_id: str) -> dict[str, Any]:
    if len(prompts) < 2:
        raise ValueError("Multiturn cases require at least two prompts.")

    previous_answers = []
    result: dict[str, Any] = {}
    for prompt in prompts:
        result = await run_swarm_case(coordinator, prompt, session_id=session_id)
        previous_answers.append(result.get("answer") or "")

    return {
        "answer": result.get("answer", ""),
        "first_answer_preview": previous_answers[0][:260],
        "swarm_enabled": result.get("swarm_enabled"),
        "session_id": session_id,
        "raw": result,
    }


def load_eval_dataset(dataset_path: Path) -> dict[str, Any]:
    if not dataset_path.exists():
        raise SystemExit(f"Evaluation dataset not found: {dataset_path}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if "cases" not in dataset or not isinstance(dataset["cases"], list):
        raise SystemExit("Evaluation dataset must contain a cases list.")
    return dataset


def build_case(coordinator, raw_case: dict[str, Any]) -> EvalCase:
    case_type = raw_case["case_type"]
    prompts = raw_case.get("prompts") or []
    session_id = raw_case.get("session_id", raw_case["case_id"])

    if case_type == "swarm":
        if len(prompts) != 1:
            raise ValueError(f"{raw_case['case_id']} swarm cases require exactly one prompt.")
        runner = lambda: run_swarm_case(coordinator, prompts[0], session_id)
    elif case_type == "multiturn":
        runner = lambda: run_multiturn_case(coordinator, prompts, session_id)
    elif case_type == "direct_skill":
        runner = lambda: run_direct_skill_case(
            raw_case["direct_skill"],
            raw_case.get("skill_args", {}),
        )
    else:
        raise ValueError(f"Unsupported case_type: {case_type}")

    return EvalCase(
        case_id=raw_case["case_id"],
        title=raw_case["title"],
        case_type=case_type,
        expected_capability=raw_case["expected_capability"],
        runner=runner,
        expected_skills=raw_case.get("expected_skills", []),
        must_contain_any=raw_case.get("must_contain_any", []),
        evaluation_focus=raw_case.get("evaluation_focus", []),
        timeout_seconds=raw_case.get("timeout_seconds", 90),
    )


def build_cases(coordinator, dataset: dict[str, Any]) -> list[EvalCase]:
    return [build_case(coordinator, raw_case) for raw_case in dataset["cases"]]


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
        "case_type": case.case_type,
        "status": "pass" if pass_checks else "fail",
        "has_answer": bool(answer),
        "expected_skills": case.expected_skills,
        "skills_called": skills,
        "expected_skill_hit": expected_skill_hit,
        "keyword_hit": keyword_hit,
        "evaluation_focus": case.evaluation_focus,
        "swarm_enabled": result.get("swarm_enabled"),
        "agents_involved": result.get("agents_involved"),
        "elapsed_seconds": elapsed,
        "answer_preview": answer[:700],
        "suggestions": result.get("suggestions", []),
        "error": error,
    }


def render_markdown(
    results: list[dict[str, Any]],
    generated_at: str,
    cold_start_seconds: float,
    dataset: dict[str, Any],
) -> str:
    passed = sum(1 for item in results if item["status"] == "pass")
    total = len(results)

    lines = [
        "# MediX Agent Swarm 功能展示与评测报告",
        "",
        f"生成时间：{generated_at}",
        f"评测规模：{total} 个核心场景",
        f"评测集：{dataset.get('dataset_name')} / {dataset.get('version')}",
        f"通过情况：{passed}/{total}",
        f"冷启动初始化耗时：{cold_start_seconds}s",
        "",
        "## 评测集构建",
        "",
        f"- 构建方式：{dataset.get('construction_method', {}).get('source')}",
        f"- 覆盖维度：{', '.join(dataset.get('coverage_dimensions', []))}",
        f"- 默认通过标准：{'; '.join(dataset.get('default_pass_criteria', []))}",
        "- 排除规则：不使用真实患者隐私数据，不声明真实临床诊断结论，不依赖外网搜索作为通过条件。",
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
                f"- 用例类型：{item['case_type']}",
                f"- 覆盖维度：{', '.join(item.get('evaluation_focus') or []) or '-'}",
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
            "- 当前评测是小样本核心功能回归评测，不等价于医学安全评测。",
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
            "只重建评测集：",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -X utf8 evaluation\\build_eval_dataset.py",
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
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the evaluation dataset JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the dataset without calling the LLM.",
    )
    args = parser.parse_args()

    dataset = load_eval_dataset(Path(args.dataset))
    if args.dry_run:
        print(f"Dataset: {dataset.get('dataset_name')} / {dataset.get('version')}")
        print(f"Cases: {len(dataset['cases'])}")
        for case in dataset["cases"]:
            print(f"- {case['case_id']}: {case['title']} ({case['case_type']})")
        return

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
    for case in build_cases(coordinator, dataset):
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
                "dataset": {
                    "dataset_name": dataset.get("dataset_name"),
                    "version": dataset.get("version"),
                    "coverage_dimensions": dataset.get("coverage_dimensions", []),
                    "default_pass_criteria": dataset.get("default_pass_criteria", []),
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        render_markdown(results, generated_at, cold_start_seconds, dataset),
        encoding="utf-8",
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
