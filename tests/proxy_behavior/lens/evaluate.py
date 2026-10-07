import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel

from litellm.proxy.lens.analysis import analyze_sample
from litellm.proxy.lens.inference import _SYSTEM
from litellm.proxy.lens.models import (
    Activity,
    Check,
    Claim,
    Coverage,
    Execution,
    ExecutionContent,
    Finding,
    InFlight,
    Job,
    LensSettings,
    ModelRequest,
    ModelResult,
    Review,
    Sample,
    TracePart,
)

logger: Final = logging.getLogger(__name__)


class Case(BaseModel):
    name: str
    split: str
    task: str
    answer: str
    steps: tuple[tuple[str, str, str, str, str], ...]
    expected: frozenset[str]
    context: str
    missing_root: bool = False
    incomplete: bool = False


class Dataset(BaseModel):
    checks: tuple[Check, ...]
    cases: tuple[Case, ...]
    feedback: tuple[Finding, ...] = ()


def fixtures(case: Case) -> tuple[Execution, tuple[TracePart, ...]]:
    execution: Final = Execution(
        id=case.name,
        source="traces",
        trace_id=case.name,
        team_id="",
        name="recorded task",
        start_time="",
        span_count=len(case.steps) + int(not case.missing_root),
        root_seen=not case.missing_root,
    )
    root: Final = TracePart(
        execution_id=case.name,
        span_id="000",
        name="task",
        kind="agent",
        content=f"Input: {case.task}\nOutput: {case.answer}\nStatus: OK",
    )
    parts: Final = tuple(
        TracePart(
            execution_id=case.name,
            span_id=f"{i:03}",
            parent_span_id="000",
            name=name,
            kind=kind,
            content=f"Input: {inp}\nOutput: {out}\nStatus: {status}",
        )
        for i, (name, kind, inp, out, status) in enumerate(case.steps, 1)
    )
    return execution, parts if case.missing_root else (root, *parts)


async def evaluate(
    cases: tuple[Case, ...],
    checks: tuple[Check, ...],
    client: httpx.AsyncClient,
    model_name: str,
    concurrency: int,
    feedback: tuple[Finding, ...] = (),
) -> dict[str, object]:
    records: Final = MappingProxyType({case.name: fixtures(case) for case in cases})
    settings: Final = LensSettings(
        name="Quality evaluation",
        model=model_name,
        checks=checks,
        context="Assess each run against its own recorded user request. Root output is the delivered answer. No agent roles or tools are mandatory unless the task requires them.",
        concurrency=concurrency,
        enabled=False,
    )
    now: Final = datetime.now(timezone.utc)
    claim: Final = Claim(
        lens_id="evaluation",
        findings=feedback,
        job=Job(id="evaluation", created_at=now, start=now, end=now, settings=settings, revision=1),
    )

    async def read(identity: str, cursor: str, offset: int) -> ExecutionContent:
        execution, parts = records[identity]
        selected: Final = tuple(p for p in parts if p.span_id > cursor)[:40]
        return ExecutionContent(
            execution=execution,
            parts=tuple(
                p.model_copy(
                    update=MappingProxyType(
                        {
                            "content": p.content[offset : offset + 8000],
                            "truncated": len(p.content) > offset + 8000,
                        }
                    )
                )
                for p in selected
            ),
            next_cursor=selected[-1].span_id if len(selected) == 40 else None,
            partial=not execution.root_seen or next(c.incomplete for c in cases if c.name == identity),
        )

    costs: Final = SimpleQueue[float | None]()
    decisions: Final = SimpleQueue[tuple[str, str]]()
    started: Final = time.monotonic()

    async def model(request: ModelRequest) -> ModelResult:
        response: Final = await client.post(
            "/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": request.prompt}],
                "max_tokens": 4096,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        raw_cost: Final = response.headers.get("x-litellm-response-cost")
        cost: Final = float(raw_cost) if raw_cost else None
        costs.put(cost)
        answer: Final = response.json()["choices"][0]["message"]["content"]
        if request.purpose == "investigate":
            payload, _ = json.JSONDecoder().raw_decode(request.prompt)
            decisions.put((payload["candidate"]["title"], answer))
        return ModelResult(content=answer, cost=cost or 0)

    async def progress(
        stage: str | None,
        coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        activity: Activity | None = None,
        /,
    ) -> None:
        if activity is not None:
            logger.info("%s", activity.model_dump_json())
        elif coverage is not None:
            logger.info("%s", json.dumps({"stage": stage, **coverage.model_dump()}))

    result: Final = await analyze_sample(
        claim,
        Sample(executions=tuple(r[0] for r in records.values()), eligible=len(records), selected=len(records)),
        read,
        model,
        progress,
    )
    assessed: Final = MappingProxyType({a.execution_id: frozenset(a.issue_checks) for a in result.assessments})
    final_checks: Final = MappingProxyType(
        {
            case.name: frozenset(
                f.check_id
                for f in result.findings
                if f.kind == "issue" and any(e.execution_id == case.name and e.role == "support" for e in f.evidence)
            )
            for case in cases
        }
    )
    comparisons: Final = tuple(
        {
            "case": c.name,
            "split": c.split,
            "expected": sorted(c.expected),
            "found": sorted(assessed.get(c.name, frozenset())),
            "missed": sorted(c.expected - assessed.get(c.name, frozenset())),
            "unexpected": sorted(assessed.get(c.name, frozenset()) - c.expected),
            "final_found": sorted(final_checks[c.name]),
            "final_missed": sorted(c.expected - final_checks[c.name]),
            "final_unexpected": sorted(final_checks[c.name] - c.expected),
        }
        for c in cases
    )
    measured: Final = tuple(costs.get_nowait() for _ in range(costs.qsize()))
    return {
        "cases": comparisons,
        "runtime_seconds": time.monotonic() - started,
        "model_calls": len(measured),
        "reported_cost_usd": sum(value for value in measured if value is not None)
        if all(value is not None for value in measured)
        else None,
        "missed_checks": sum(len(c["missed"]) for c in comparisons),
        "unexpected_checks": sum(len(c["unexpected"]) for c in comparisons),
        "investigation_responses": tuple(decisions.get_nowait() for _ in range(decisions.qsize())),
        "result": result.model_dump(mode="json"),
    }


async def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Run paid, real-model Lens quality evaluations")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("quality_cases.json"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="all")
    parser.add_argument("--background", type=int, default=0, help="Additional clean runs for rare-problem batch tests")
    parser.add_argument("--concurrency", type=int, default=8)
    args: Final = parser.parse_args()
    dataset: Final = Dataset.model_validate_json(args.dataset.read_text())
    selected: Final = tuple(c for c in dataset.cases if args.split == "all" or c.split == args.split)
    background: Final = tuple(
        Case(
            name=f"background-{i}",
            split="background",
            task=f"Add {i} and 7.",
            answer=str(i + 7),
            steps=(),
            expected=frozenset(),
            context="Direct arithmetic answers do not need tools or an editor.",
        )
        for i in range(args.background)
    )
    async with httpx.AsyncClient(
        base_url=args.api_base.rstrip("/"),
        headers={"Authorization": "Bearer " + os.environ["LITELLM_API_KEY"]},
        timeout=180,
    ) as client:
        report: Final = await evaluate(
            (*selected, *background), dataset.checks, client, args.model, args.concurrency, dataset.feedback
        )
    args.output.write_text(
        json.dumps({"model": args.model, "background_runs": args.background, **report}, indent=2) + "\n"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
