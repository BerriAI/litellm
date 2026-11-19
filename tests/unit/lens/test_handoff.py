from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

import pytest

from litellm.harness.evaluation import CheckResult, EvalProgress, EvalRun, EvalStatus
from litellm.harness.types import Result, ToolCall, ToolResult, Usage
from litellm.lens.core import RunManifest, RunStore, TargetInfo, TaskRow
from litellm.lens.handoff import HandoffPreview, export_handoff


@pytest.fixture
def saved_run(tmp_path: Path) -> Iterator[RunStore]:
    manifest: Final = RunManifest(
        run_id="eval_fixture",
        created_at=1_700_000_000,
        tasks_path="tasks.csv",
        config_path="evals.py",
        fingerprint="fixture",
        trials=1,
        total=1,
        fixture=True,
        targets=(
            TargetInfo(
                label="model-a",
                harness="custom",
                model="fixture/model-a",
                telemetry_complete=False,
            ),
        ),
        tasks=(
            TaskRow(
                id="case-a",
                prompt="private task prompt",
                expected="private expected answer",
            ),
        ),
    )
    store: Final = RunStore(tmp_path, manifest)
    call: Final = ToolCall("lookup-1", "lookup", "lookup", {"query": "private lookup arguments"})
    output: Final = ToolResult("lookup-1", "private lookup output", is_error=True)
    store.save_progress(EvalProgress("eval_fixture", "attempt-a", "case-a", "model-a", 1))
    store.save_progress(EvalProgress("eval_fixture", "attempt-a", "case-a", "model-a", 1, call))
    store.save_progress(EvalProgress("eval_fixture", "attempt-a", "case-a", "model-a", 1, output))
    store.save_result(
        EvalRun(
            "attempt-a",
            "case-a",
            "model-a",
            1,
            EvalStatus.FAIL,
            Result(
                "private model answer",
                None,
                [],
                [call, output],
                Usage(),
                0,
                "done",
                "fixture-session",
            ),
            CheckResult(False, "private grading detail"),
            1.0,
            "private grading detail",
        )
    )
    try:
        yield store
    finally:
        store.close()


def test_metadata_handoff_excludes_contents_and_preserves_unknown_cost(
    saved_run: RunStore,
) -> None:
    preview: Final = export_handoff(saved_run)
    encoded: Final = preview.json_path.read_text(encoding="utf-8")
    report: Final = preview.html_path.read_text(encoding="utf-8")
    assert "private" not in encoded
    assert "private" not in report
    payload: Final = json.loads(encoded)
    root: Final = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    attributes: Final = {attribute["key"]: attribute["value"] for attribute in root["attributes"]}
    assert attributes["lens.source"] == {"stringValue": "evaluation"}
    assert attributes["lens.eval.agent_cost_known"] == {"boolValue": False}
    assert "lens.eval.agent_cost" not in attributes
    assert "unknown" in report
    assert "Nothing has been uploaded" in report
    assert preview.case_url("case-a", "model-a").startswith(preview.report_url + "#case-")


def test_repeated_handoff_keeps_trace_ids_and_saved_cases(saved_run: RunStore) -> None:
    before: Final = saved_run.rows()
    first: Final = export_handoff(saved_run)
    first_json: Final = first.json_path.read_bytes()
    second: Final = export_handoff(saved_run)
    assert second.json_path.read_bytes() == first_json
    assert saved_run.rows() == before
    payload: Final = json.loads(first_json)
    spans: Final = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len({span["traceId"] for span in spans}) == 1
    assert len({span["spanId"] for span in spans}) == len(spans)
    assert all(span["parentSpanId"] == spans[0]["spanId"] for span in spans[1:])
    assert all(len(span["traceId"]) == 32 and len(span["spanId"]) == 16 for span in spans)


def test_case_link_prefers_failure_and_preserves_explicit_target(
    saved_run: RunStore,
) -> None:
    failed: Final = saved_run.rows()[0].model_copy(update={"target": "model-b"})
    passed: Final = failed.model_copy(update={"target": "model-a", "status": "pass"})
    preview: Final = HandoffPreview(
        saved_run.directory / "preview.json",
        saved_run.directory / "preview.html",
        (passed, failed),
    )
    assert preview.case_url("case-a") == preview.case_url("case-a", "model-b")
    assert preview.case_url("case-a", "model-a") != preview.case_url("case-a")


def test_content_export_is_opt_in_and_html_escaped(saved_run: RunStore) -> None:
    script: Final = '<script>alert("content")</script>'
    saved_run.save_progress(
        EvalProgress(
            "eval_fixture",
            "attempt-a",
            "case-a",
            "model-a",
            1,
            ToolResult("lookup-1", script),
        )
    )
    metadata: Final = export_handoff(saved_run)
    assert "private model answer" not in metadata.html_path.read_text(encoding="utf-8")
    content: Final = export_handoff(saved_run, include_content=True)
    report: Final = content.html_path.read_text(encoding="utf-8")
    assert "private model answer" in report
    assert "private lookup arguments" in content.json_path.read_text(encoding="utf-8")
    assert script not in report
    assert "&lt;script&gt;" in report
    assert metadata.json_path != content.json_path
    with pytest.raises(ValueError, match="No saved attempt"):
        content.case_url("missing-case")


def test_report_omits_unknown_cost_points_and_labels_fixture(
    saved_run: RunStore,
) -> None:
    preview: Final = export_handoff(saved_run)
    report: Final = preview.html_path.read_text(encoding="utf-8")
    assert 'aria-label="Cost vs. quality"' in report
    assert 'aria-label="Cost vs. latency"' in report
    assert 'class="plot-point"' not in report
    assert "1 target(s) omitted: model-a (1 unknown costs)" in report
    assert "SIMULATED FIXTURE" in report
    assert "$0 billed" in report
    assert "private" not in report


def test_report_graphs_aggregate_attempts_include_errors_and_plot_zero(
    tmp_path: Path,
) -> None:
    manifest: Final = RunManifest(
        run_id="eval_graphs",
        created_at=1_700_000_000,
        tasks_path="tasks.csv",
        config_path="evals.py",
        fingerprint="fixture",
        trials=1,
        total=6,
        fixture=True,
        targets=tuple(
            TargetInfo(
                label=name,
                harness="custom",
                model=name,
                telemetry_complete=name != "unknown-c",
            )
            for name in ("model-a", "model-b", "unknown-c")
        ),
        tasks=tuple(TaskRow(id=name, prompt="private prompt", expected="answer") for name in ("case-1", "case-2")),
    )
    store: Final = RunStore(tmp_path, manifest)
    attempts: Final = (
        ("model-a", "case-1", EvalStatus.PASS, 0.1, 1.0),
        ("model-a", "case-2", EvalStatus.FAIL, 0.3, 3.0),
        ("model-b", "case-1", EvalStatus.PASS, 0.0, 2.0),
        ("model-b", "case-2", EvalStatus.ERROR, 0.0, 4.0),
        ("unknown-c", "case-1", EvalStatus.PASS, 0.0, 2.0),
        ("unknown-c", "case-2", EvalStatus.PASS, 0.0, 2.0),
    )
    try:
        for target, task_id, status, cost, duration in attempts:
            store.save_result(
                EvalRun(
                    f"{target}-{task_id}",
                    task_id,
                    target,
                    1,
                    status,
                    Result(
                        "private answer",
                        None,
                        [],
                        [],
                        Usage(calls=1),
                        cost,
                        "done",
                        "fixture",
                    ),
                    CheckResult(status is EvalStatus.PASS, "private detail"),
                    duration,
                    "private detail",
                )
            )
        preview: Final = export_handoff(store)
        report: Final = preview.html_path.read_text(encoding="utf-8")
        charts: Final = tuple(
            ElementTree.fromstring(fragment) for fragment in re.findall(r"<svg\b.*?</svg>", report, re.DOTALL)
        )
        assert len(charts) == 2
        for chart in charts:
            points: Final = {point.attrib["data-target"]: point.attrib for point in chart.findall(".//g")}
            assert points.keys() == {"model-a", "model-b"}
            assert float(points["model-a"]["data-cost"]) == pytest.approx(0.2)
            assert float(points["model-a"]["data-pass-rate"]) == pytest.approx(0.5)
            assert float(points["model-a"]["data-mean-duration"]) == pytest.approx(2.0)
            assert float(points["model-b"]["data-cost"]) == 0.0
            assert float(points["model-b"]["data-pass-rate"]) == pytest.approx(0.5)
            assert float(points["model-b"]["data-mean-duration"]) == pytest.approx(3.0)
        assert "unknown-c (2 unknown costs)" in report
        assert "Highest pass rate</strong> unknown-c 100.0% (2/2)" in report
        assert "Lowest pass rate</strong> model-a 50.0% (1/2)" in report
        assert "model-b 50.0% (1/2)" in report
        assert "private" not in report
    finally:
        store.close()
