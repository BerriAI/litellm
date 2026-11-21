from __future__ import annotations

import math
import shlex
from collections.abc import Iterator
from contextlib import closing
from io import StringIO
from pathlib import Path
from typing import Final

import pytest
from rich.console import Console

from litellm.harness.evaluation import CheckResult, EvalProgress, EvalRun, EvalStatus
from litellm.harness.types import Event, Result, ToolCall, ToolResult, Usage
from litellm.lens.analysis import (
    print_analysis,
    quality_extremes,
    ranked_cases,
    target_metrics,
    tool_report,
)
from litellm.lens.core import RunManifest, RunStore, TargetInfo, TaskRow


def _save(
    store: RunStore,
    target: str,
    task: str,
    status: EvalStatus,
    duration: float,
    cost: float | None,
    events: tuple[Event, ...] = (),
) -> None:
    run_id: Final = f"{target}-{task}"
    store.save_progress(EvalProgress(store.run_id, run_id, task, target, 1))
    for event in events:
        store.save_progress(EvalProgress(store.run_id, run_id, task, target, 1, event))
    store.save_result(
        EvalRun(
            run_id,
            task,
            target,
            1,
            status,
            (
                Result(
                    "answer",
                    None,
                    [],
                    list(events),
                    Usage(calls=1),
                    cost,
                    "done",
                    run_id,
                )
                if cost is not None
                else None
            ),
            (CheckResult(status is EvalStatus.PASS, "fixture check") if status is not EvalStatus.ERROR else None),
            duration,
            "fixture failure" if status is not EvalStatus.PASS else "fixture pass",
        )
    )


@pytest.fixture
def analyzed_run(tmp_path: Path) -> Iterator[RunStore]:
    manifest: Final = RunManifest(
        run_id="eval_analysis",
        created_at=1_700_000_000,
        tasks_path="tasks.csv",
        config_path="evals.py",
        fingerprint="fixture",
        trials=1,
        total=16,
        fixture=True,
        targets=tuple(
            TargetInfo(
                label=label,
                harness="custom",
                model=f"fixture/{label}",
                telemetry_complete=label in ("measured", "free"),
            )
            for label in ("measured", "unknown", "free", "silent")
        ),
        tasks=tuple(TaskRow(id=task, prompt="Fixture prompt", expected="answer") for task in ("a", "b", "c", "d")),
    )
    store: Final = RunStore(tmp_path, manifest)
    _save(
        store,
        "measured",
        "a",
        EvalStatus.PASS,
        1,
        1,
        (
            ToolCall(
                "one",
                "lookup_ticket",
                "lookup_ticket",
                {"key": "value", "options": {"b": 2, "a": 1}},
            ),
            ToolResult("one", "found"),
            ToolCall(
                "two",
                "lookup_ticket",
                "lookup_ticket",
                {"options": {"a": 1, "b": 2}, "key": "value"},
            ),
            ToolResult("two", "found"),
        ),
    )
    _save(
        store,
        "measured",
        "b",
        EvalStatus.FAIL,
        3,
        3,
        (
            ToolCall(
                "one",
                "lookup_ticket",
                "lookup_ticket",
                {"key": "value", "options": {"a": 1, "b": 2}},
            ),
            ToolResult("one", "failed", is_error=True),
            ToolResult("one", "same failure reported twice", is_error=True),
        ),
    )
    _save(
        store,
        "measured",
        "c",
        EvalStatus.ERROR,
        5,
        2,
        (ToolResult("orphan", "unmatched", is_error=True),),
    )
    _save(
        store,
        "unknown",
        "a",
        EvalStatus.PASS,
        2,
        0,
        (ToolCall("one", "search", "search", {"query": "value"}),),
    )
    _save(store, "free", "a", EvalStatus.PASS, 0.5, 0)
    _save(store, "silent", "a", EvalStatus.PASS, 2, 0)
    try:
        yield store
    finally:
        store.close()


def test_should_include_errors_in_quality_and_preserve_unknown_and_zero_costs(
    analyzed_run: RunStore,
) -> None:
    metrics: Final = {metric.target: metric for metric in target_metrics(analyzed_run)}
    measured: Final = metrics["measured"]
    assert measured.pass_rate is not None
    assert math.isclose(measured.pass_rate, 1 / 3)
    assert measured.mean_agent_cost == 2
    assert measured.mean_duration == 3
    assert measured.p95_duration == 5
    assert measured.cost_per_pass == 6
    assert measured.errors == 1
    assert metrics["unknown"].mean_agent_cost is None
    assert metrics["unknown"].missing_cost_count == 1
    assert metrics["unknown"].cost_per_pass is None
    assert metrics["free"].mean_agent_cost == 0
    assert metrics["free"].cost_per_pass == 0


def test_should_aggregate_tools_over_all_attempts_and_repeat_only_within_an_attempt(
    analyzed_run: RunStore,
) -> None:
    report: Final = tool_report(analyzed_run)
    lookup: Final = next(
        metric for metric in report.metrics if metric.target == "measured" and metric.tool == "lookup_ticket"
    )
    assert lookup.attempts == 3
    assert lookup.calls == 3
    assert lookup.calls_per_attempt == 1
    assert lookup.errors == 1
    assert lookup.repeated_attempts == 1
    assert lookup.extra_calls == 1
    assert math.isclose(lookup.extra_call_rate, 1 / 3)
    assert len(report.repeats) == 1
    assert report.repeats[0].case.task_id == "a"
    assert report.repeats[0].calls == 2
    assert report.unmatched_errors == 1
    assert report.unobserved_targets == ("silent",)
    partial: Final = next(metric for metric in report.metrics if metric.target == "unknown")
    assert partial.calls == 1
    assert not partial.telemetry_complete
    zero: Final = next(metric for metric in report.metrics if metric.target == "free")
    assert zero.calls == 0 and zero.calls_per_attempt == 0
    assert zero.telemetry_complete


def test_should_leave_cost_means_unknown_when_only_some_attempt_costs_are_known(
    analyzed_run: RunStore,
) -> None:
    _save(analyzed_run, "measured", "d", EvalStatus.ERROR, 7, None)
    measured: Final = next(metric for metric in target_metrics(analyzed_run) if metric.target == "measured")
    assert measured.attempts == 4
    assert measured.known_cost_count == 3
    assert measured.missing_cost_count == 1
    assert measured.mean_agent_cost is None
    assert measured.cost_per_pass is None
    assert measured.mean_duration == 4


def test_should_rank_explicit_outcomes_and_keep_all_tied_targets(
    analyzed_run: RunStore,
) -> None:
    best, worst = quality_extremes(target_metrics(analyzed_run))
    assert {metric.target for metric in best} == {"unknown", "free", "silent"}
    assert tuple(metric.target for metric in worst) == ("measured",)
    cases: Final = ranked_cases(analyzed_run)
    assert cases[0].target == "free"
    assert cases[0].status == "pass"
    assert cases[-1].status == "error"
    assert cases[-1].task_id == "c"


@pytest.mark.parametrize("width", (80, 132))
def test_should_render_observed_tools_charts_and_exact_failure_links(analyzed_run: RunStore, width: int) -> None:
    output: Final = StringIO()
    print_analysis(Console(file=output, width=width, color_system=None), analyzed_run)
    rendered: Final = output.getvalue()
    assert "Cost vs quality" in rendered
    assert "Cost vs latency" in rendered
    assert "lookup_ticket" in rendered and "search" in rendered
    assert "lower bounds" in rendered
    assert "tool usage unknown" in rendered
    assert "Repeated lookup" in rendered
    assert "/inspect eval_analysis --case a --target measured --trial 1" in " ".join(rendered.split())
    assert "Fixture metrics are simulated" in rendered
    assert "Chart omitted unknown costs" in rendered
    assert "Model-step counts are unavailable" in rendered


def test_should_respect_analysis_modules_without_changing_saved_results(
    analyzed_run: RunStore,
) -> None:
    before: Final = analyzed_run.rows()
    output: Final = StringIO()
    print_analysis(
        Console(file=output, width=132, color_system=None),
        analyzed_run,
        tools=False,
        redundancy=False,
        charts=False,
    )
    rendered: Final = output.getvalue()
    assert "$/ATTEMPT" in rendered
    assert "Cost vs quality" not in rendered
    assert "Observed tool calls" not in rendered
    assert analyzed_run.rows() == before


def test_should_quote_case_and_target_names_in_copyable_inspect_commands(
    analyzed_run: RunStore, tmp_path: Path
) -> None:
    case_id: Final = "case with spaces and 'quotes'"
    target_label: Final = 'target with spaces and "quotes"'
    manifest: Final = analyzed_run.manifest.model_copy(
        update={
            "targets": (analyzed_run.manifest.targets[0].model_copy(update={"label": target_label}),),
            "tasks": (TaskRow(id=case_id, prompt="Fixture prompt", expected="answer"),),
            "total": 1,
        }
    )
    directory: Final = tmp_path / "spaces"
    directory.mkdir()
    output: Final = StringIO()
    with closing(RunStore(directory, manifest)) as store:
        _save(store, target_label, case_id, EvalStatus.FAIL, 1, 0)
        print_analysis(
            Console(file=output, width=500, color_system=None),
            store,
            tools=False,
            redundancy=False,
            charts=False,
        )
        command: Final = next(
            line[line.index("/inspect") :] for line in output.getvalue().splitlines() if "/inspect" in line
        )
        assert shlex.split(command) == [
            "/inspect",
            store.run_id,
            "--case",
            case_id,
            "--target",
            target_label,
            "--trial",
            "1",
        ]
