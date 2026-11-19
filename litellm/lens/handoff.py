from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

from litellm.lens.analysis import (
    TargetMetrics,
    quality_extremes,
    ranked_cases,
    target_metrics,
)
from litellm.lens.core import CaseRow, EventRow, RunStore

_COLORS: Final = ("#b395ff", "#5c91ff", "#8ad9b3", "#edc07c", "#ed8b9f")


class _ChartKind(Enum):
    QUALITY = "quality"
    LATENCY = "latency"


def _identity(*parts: str | int) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def _anchor(case: CaseRow) -> str:
    return "case-" + _identity(case.task_id, case.target, case.trial)[:24]


@dataclass(frozen=True, slots=True)
class HandoffPreview:
    json_path: Path
    html_path: Path
    cases: tuple[CaseRow, ...]

    @property
    def report_url(self) -> str:
        return self.html_path.resolve().as_uri()

    def case_url(self, task_id: str, target: str | None = None, trial: int = 1) -> str:
        matching: Final = tuple(case for case in self.cases if case.task_id == task_id and case.trial == trial)
        selected: Final = tuple(case for case in matching if target is None or case.target == target)
        if not selected:
            raise ValueError("No saved attempt matches that case, target and trial")
        case: Final = next((case for case in selected if case.status != "pass"), selected[0])
        return f"{self.report_url}#{_anchor(case)}"


def _attribute(key: str, value: str | float | bool) -> Mapping[str, object]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": value}}


def _task_prompt(store: RunStore, case: CaseRow) -> str:
    return next(task.prompt for task in store.manifest.tasks if task.id == case.task_id)


def _case_attributes(store: RunStore, case: CaseRow, include_content: bool) -> tuple[Mapping[str, object], ...]:
    identity: Final = (
        _attribute("lens.source", "evaluation"),
        _attribute("lens.eval.run_id", store.run_id),
        _attribute("lens.eval.case_id", case.task_id),
        _attribute("lens.eval.target_id", case.target),
        _attribute("lens.eval.trial", case.trial),
        _attribute("lens.eval.status", case.status),
        _attribute("lens.eval.preview", True),
        _attribute("lens.eval.agent_cost_known", case.agent_cost is not None),
        _attribute("lens.eval.judge_cost_known", case.judge_cost is not None),
    )
    metrics: Final = tuple(
        _attribute(key, value)
        for key, value in (
            ("lens.eval.agent_cost", case.agent_cost),
            ("lens.eval.judge_cost", case.judge_cost),
            ("lens.eval.model_calls", case.model_calls),
            ("lens.eval.tool_calls", case.tool_calls),
        )
        if value is not None
    )
    content: Final = (
        (
            _attribute("lens.eval.prompt", _task_prompt(store, case)),
            _attribute("lens.eval.output", case.text or ""),
            _attribute("lens.eval.detail", case.detail),
        )
        if include_content
        else ()
    )
    return identity + metrics + content


def _event_attributes(event: EventRow, include_content: bool) -> tuple[Mapping[str, object], ...]:
    metadata: Final = (
        _attribute("lens.event.kind", event.kind),
        _attribute("lens.event.sequence", event.sequence),
        _attribute("lens.telemetry.timing", "instant event; span duration unavailable"),
    )
    name: Final = (_attribute("tool.name", event.name),) if event.name is not None else ()
    arguments: Final = (
        (_attribute("tool.arguments", event.arguments),) if include_content and event.arguments is not None else ()
    )
    output: Final = (_attribute("tool.output", event.output),) if include_content and event.output is not None else ()
    return metadata + name + arguments + output


def _event_span(suite_id: str, case: CaseRow, event: EventRow, include_content: bool) -> Mapping[str, object]:
    return {
        "traceId": _identity(suite_id, case.task_id, case.target, case.trial)[:32],
        "spanId": _identity(suite_id, case.task_id, case.target, case.trial, event.sequence)[:16],
        "parentSpanId": _identity(suite_id, case.task_id, case.target, case.trial, "attempt")[:16],
        "name": event.name or event.kind,
        "kind": 1,
        "startTimeUnixNano": str(int(event.at * 1_000_000_000)),
        "endTimeUnixNano": str(int(event.at * 1_000_000_000)),
        "attributes": _event_attributes(event, include_content),
        "status": {"code": 2 if event.is_error else 1},
    }


def _case_spans(store: RunStore, case: CaseRow, include_content: bool) -> tuple[Mapping[str, object], ...]:
    events: Final = store.events(case.run_id)
    started: Final = case.started_at
    root: Final = {
        "traceId": _identity(store.run_id, case.task_id, case.target, case.trial)[:32],
        "spanId": _identity(store.run_id, case.task_id, case.target, case.trial, "attempt")[:16],
        "name": "eval.attempt",
        "kind": 1,
        "startTimeUnixNano": str(int(started * 1_000_000_000)),
        "endTimeUnixNano": str(int((started + case.duration) * 1_000_000_000)),
        "attributes": _case_attributes(store, case, include_content),
        "status": {"code": 2 if case.status == "error" else 1},
    }
    return (root,) + tuple(_event_span(store.run_id, case, event, include_content) for event in events)


def _resource_spans(store: RunStore, case: CaseRow, include_content: bool) -> Mapping[str, object]:
    target: Final = next(target for target in store.manifest.targets if target.label == case.target)
    return {
        "resource": {
            "attributes": (
                _attribute("service.name", "lens-cli-eval-preview"),
                _attribute("deployment.environment.name", "evaluation"),
                _attribute("lens.eval.fixture", store.manifest.fixture),
                _attribute("lens.eval.harness", target.harness),
                _attribute("gen_ai.request.model", target.model),
            )
        },
        "scopeSpans": (
            {
                "scope": {"name": "litellm.lens.preview", "version": "0.1"},
                "spans": _case_spans(store, case, include_content),
            },
        ),
    }


def _money(values: Sequence[float | None]) -> str:
    known: Final = tuple(value for value in values if value is not None)
    if not known:
        return "unknown"
    suffix: Final = " (partial)" if len(known) < len(values) else ""
    return f"${sum(known):.4f}{suffix}"


def _rate(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "unknown"


def _seconds(value: float | None) -> str:
    return f"{value:.2f}s" if value is not None else "unknown"


def _summary(metric: TargetMetrics, cases: tuple[CaseRow, ...]) -> str:
    selected: Final = tuple(case for case in cases if case.target == metric.target)
    return (
        f"<tr><th>{html.escape(metric.target)}</th><td>{_rate(metric.pass_rate)}</td>"
        f"<td>{metric.passed} / {metric.failed} / {metric.errors}</td>"
        f"<td>{_money((metric.mean_agent_cost,))}</td><td>{_seconds(metric.mean_duration)}</td>"
        f"<td>{_money(tuple(case.agent_cost for case in selected))}</td>"
        f"<td>{_money(tuple(case.judge_cost for case in selected))}</td></tr>"
    )


def _plot_value(metric: TargetMetrics, kind: _ChartKind) -> float | None:
    if kind is _ChartKind.LATENCY:
        return metric.mean_duration
    return metric.pass_rate * 100 if metric.pass_rate is not None else None


def _dot(metric: TargetMetrics, index: int, kind: _ChartKind, x_max: float, y_max: float) -> str:
    cost: Final = metric.mean_agent_cost
    value: Final = _plot_value(metric, kind)
    if cost is None or value is None:
        return ""
    x: Final = 66 + cost / x_max * 440
    y: Final = 282 - value / y_max * 224
    color: Final = _COLORS[index % len(_COLORS)]
    name: Final = metric.target if len(metric.target) < 24 else metric.target[:21] + "..."
    label_x: Final = x + 11 if x < 380 else x - 11
    anchor: Final = "start" if x < 380 else "end"
    return (
        f'<g class="plot-point" data-target="{html.escape(metric.target)}" data-cost="{cost}" '
        f'data-pass-rate="{metric.pass_rate}" data-mean-duration="{metric.mean_duration}">'
        f"<title>{html.escape(metric.target)} · {_rate(metric.pass_rate)} pass · ${cost:.6g}/attempt · "
        f"{_seconds(metric.mean_duration)} mean latency · {metric.attempts} attempts</title>"
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}" stroke="#100d16" stroke-width="2"/>'
        f'<text class="target-label" x="{label_x:.2f}" y="{y - 12:.2f}" fill="{color}" text-anchor="{anchor}">'
        f"{html.escape(name)}</text></g>"
    )


def _tick(value: float, kind: _ChartKind) -> str:
    return f"{value:.0f}%" if kind is _ChartKind.QUALITY else f"{value:.3g}s"


def _grid(x_max: float, y_max: float, kind: _ChartKind) -> str:
    horizontal: Final = "".join(
        f'<line x1="66" x2="506" y1="{282 - fraction * 224:.2f}" y2="{282 - fraction * 224:.2f}" '
        f'stroke="#30283e"/><text x="54" y="{286 - fraction * 224:.2f}" text-anchor="end">'
        f"{_tick(fraction * y_max, kind)}</text>"
        for fraction in (0, 0.25, 0.5, 0.75, 1)
    )
    vertical: Final = "".join(
        f'<text x="{66 + fraction * 440:.2f}" y="306" text-anchor="middle">${fraction * x_max:.3g}</text>'
        for fraction in (0, 0.5, 1)
    )
    return horizontal + vertical


def _legend(metrics: tuple[TargetMetrics, ...]) -> str:
    entries: Final = "".join(
        f'<li><i style="background:{_COLORS[index % len(_COLORS)]}"></i>'
        f"{html.escape(metric.target)} <span>{metric.attempts} attempts</span></li>"
        for index, metric in enumerate(metrics)
    )
    return f'<ul class="legend">{entries}</ul>'


def _omission(metric: TargetMetrics) -> str:
    if not metric.attempts:
        return f"{html.escape(metric.target)} (no completed attempts)"
    return f"{html.escape(metric.target)} ({metric.missing_cost_count} unknown costs)"


def _scatter(metrics: tuple[TargetMetrics, ...], kind: _ChartKind) -> str:
    costs: Final = tuple(metric.mean_agent_cost for metric in metrics if metric.mean_agent_cost is not None)
    durations: Final = tuple(metric.mean_duration for metric in metrics if metric.mean_duration is not None)
    x_max: Final = max(costs, default=0) * 1.2 or 0.000001
    y_max: Final = 100.0 if kind is _ChartKind.QUALITY else max(durations, default=0) * 1.2 or 1.0
    title: Final = "Cost vs. quality" if kind is _ChartKind.QUALITY else "Cost vs. latency"
    y_label: Final = "Pass rate" if kind is _ChartKind.QUALITY else "Mean latency (s)"
    dots: Final = "".join(_dot(metric, index, kind, x_max, y_max) for index, metric in enumerate(metrics))
    omitted: Final = tuple(metric for metric in metrics if metric.mean_agent_cost is None)
    note: Final = (
        f"{len(omitted)} target(s) omitted: {', '.join(_omission(metric) for metric in omitted)}."
        if omitted
        else "All completed attempts have known agent cost."
    )
    return (
        f'<figure><h2>{title}</h2><svg viewBox="0 0 560 350" role="img" aria-label="{title}">'
        f"<title>{title}. One point per target. {note}</title>"
        f'<text x="66" y="25">{y_label}</text>{_grid(x_max, y_max, kind)}'
        f'<line x1="66" y1="282" x2="506" y2="282" stroke="#756887"/>{dots}'
        '<text x="286" y="339" text-anchor="middle">Mean agent cost / attempt (USD)</text>'
        f"</svg><figcaption>{note}</figcaption></figure>"
    )


def _target_evidence(metrics: tuple[TargetMetrics, ...]) -> str:
    return " · ".join(
        f"{html.escape(metric.target)} {_rate(metric.pass_rate)} ({metric.passed}/{metric.attempts}), "
        f"{_money((metric.mean_agent_cost,))}/attempt, {_seconds(metric.mean_duration)} mean"
        for metric in metrics
    )


def _case_evidence(label: str, case: CaseRow) -> str:
    return (
        f'<p><strong>{label}</strong> <a href="#{_anchor(case)}">{html.escape(case.task_id)}</a> · '
        f"{html.escape(case.target)} · trial {case.trial} · {html.escape(case.status)} · "
        f"{case.duration:.2f}s · {_money((case.agent_cost,))}</p>"
    )


def _findings(store: RunStore, metrics: tuple[TargetMetrics, ...]) -> str:
    best, worst = quality_extremes(metrics)
    ranked: Final = ranked_cases(store)
    if not ranked:
        return "<p>No completed attempts yet.</p>"
    return (
        f"<h2>Best and worst results</h2><p><strong>Highest pass rate</strong> {_target_evidence(best)}</p>"
        f"<p><strong>Lowest pass rate</strong> {_target_evidence(worst)}</p>"
        "<p>Quality ties are shown together. Pass rate includes errors in the denominator.</p>"
        f"{_case_evidence('Best recorded attempt', ranked[0])}"
        f"{_case_evidence('Worst recorded attempt', ranked[-1])}"
        "<p>Attempt examples are ordered pass, fail, error, then fastest to slowest within each result.</p>"
    )


def _event_html(event: EventRow, include_content: bool) -> str:
    detail: Final = (
        f"<pre>{html.escape(event.arguments or '')}\n{html.escape(event.output or '')}</pre>" if include_content else ""
    )
    return (
        f"<li>{html.escape(event.kind)} {html.escape(event.name or '')}"
        f"{' · error' if event.is_error else ''}{detail}</li>"
    )


def _case_html(store: RunStore, case: CaseRow, include_content: bool) -> str:
    events: Final = "".join(_event_html(event, include_content) for event in store.events(case.run_id))
    detail: Final = (
        f"<pre>Prompt\n{html.escape(_task_prompt(store, case))}\n\nAnswer\n{html.escape(case.text or '')}"
        f"\n\nCheck\n{html.escape(case.detail)}</pre>"
        if include_content
        else ""
    )
    return (
        f'<details id="{_anchor(case)}"><summary><a href="#{_anchor(case)}">'
        f"{html.escape(case.task_id)}</a> · {html.escape(case.target)} · trial {case.trial} · "
        f'<span class="{html.escape(case.status)}">{html.escape(case.status)}</span></summary>'
        f"<p>Agent {_money((case.agent_cost,))} · Judge {_money((case.judge_cost,))} · "
        f"{case.duration:.2f}s</p>{detail}<ol>{events}</ol></details>"
    )


def _report_html(store: RunStore, cases: tuple[CaseRow, ...], include_content: bool) -> str:
    metrics: Final = target_metrics(store)
    summaries: Final = "".join(_summary(metric, cases) for metric in metrics)
    charts: Final = _scatter(metrics, _ChartKind.QUALITY) + _scatter(metrics, _ChartKind.LATENCY)
    findings: Final = _findings(store, metrics)
    legend: Final = _legend(metrics)
    attempts: Final = "".join(_case_html(store, case, include_content) for case in cases)
    privacy: Final = (
        "Content included by request"
        if include_content
        else "Metadata only. Prompts, answers and tool contents excluded."
    )
    fixture: Final = (
        "SIMULATED FIXTURE. All cost and latency figures are simulated. $0 billed. No model requests."
        if store.manifest.fixture
        else "Saved local run."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Lens local handoff preview</title><style>
body{{background:#100d16;color:#ece6f5;font:15px ui-monospace,monospace;max-width:1100px;margin:40px auto;padding:0 24px}}
h1,a{{color:#b395ff}}a{{text-decoration:none}}p{{color:#b6afc5}}table{{border-collapse:collapse;width:100%;margin:24px 0}}
th,td{{text-align:left;padding:12px;border-bottom:1px solid #3a304b}}details{{padding:16px 0;border-bottom:1px solid #3a304b}}
summary{{cursor:pointer}}details:target{{outline:1px solid #b395ff;background:#20172e}}details:target>summary{{padding:8px}}
li{{margin:12px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:#c5bfd2}}.fail,.error{{color:#ffad99}}.pass{{color:#9ce3bb}}
h2{{font-size:17px;font-weight:500;color:#ece6f5;margin-top:30px}}strong{{font-weight:500;color:#ece6f5}}
.charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px;border-block:1px solid #3a304b;padding:8px 0 24px}}
figure{{margin:0;min-width:0}}svg{{width:100%;height:auto;font:12px ui-monospace,monospace}}svg text:not(.target-label){{fill:#aaa0b9}}
figcaption{{color:#9d92ae;font-size:12px;line-height:1.6}}.table-scroll{{overflow-x:auto}}
.legend{{display:flex;gap:24px;flex-wrap:wrap;list-style:none;padding:0;font-size:13px}}.legend span{{color:#9d92ae}}
.legend i{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px}}
@media(max-width:750px){{.charts{{grid-template-columns:1fr}}body{{margin:24px auto;padding:0 16px}}}}
</style></head><body><h1>LENS / local handoff preview</h1>
<p>{html.escape(store.run_id)} · {len(cases)} completed attempts · {fixture}</p>
<p>{privacy} Nothing has been uploaded to Lens. Live publish and dashboard filtering are not connected.</p>
<p>Charts use completed attempts. Mean cost includes every attempt only when all costs are known. Judge cost is separate.</p>
{legend}<div class="charts">{charts}</div>{findings}<div class="table-scroll"><table><thead><tr><th>Target</th><th>Pass rate</th>
<th>Pass / fail / error</th><th>Mean agent / attempt</th><th>Mean latency</th><th>Agent total</th><th>Judge total</th></tr></thead>
<tbody>{summaries}</tbody></table></div><h2>Saved cases</h2>{attempts}</body></html>"""


def _write(path: Path, content: str) -> None:
    temporary: Final = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def export_handoff(store: RunStore, *, include_content: bool = False) -> HandoffPreview:
    cases: Final = store.rows()
    variant: Final = "content" if include_content else "metadata"
    json_path: Final = store.directory / f"handoff.{variant}.json"
    html_path: Final = store.directory / f"handoff.{variant}.html"
    payload: Final = {"resourceSpans": tuple(_resource_spans(store, case, include_content) for case in cases)}
    _write(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _write(html_path, _report_html(store, cases, include_content))
    return HandoffPreview(json_path, html_path, cases)
