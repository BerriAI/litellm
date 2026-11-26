"""Tasks, checks and scorecards for agent evaluations."""

from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias, get_args

from pydantic import BaseModel

from litellm.harness.options import HarnessOptions
from litellm.harness.sandbox.base import Sandbox
from litellm.harness.types import Event, FileChange, Harness, Result, StopReason, ToolCall, Usage, require_harness

_STOP_REASONS: Final = frozenset(get_args(StopReason))


class EvalOutput(Protocol):
    """Shared output fields for predicates over built-in and custom agents.

    Use this annotation instead of Result when a predicate also accepts custom agent output.
    """

    @property
    def text(self) -> str: ...

    @property
    def files(self) -> Sequence[FileChange]: ...

    @property
    def events(self) -> Sequence[Event] | None: ...

    @property
    def usage(self) -> Usage | None: ...

    @property
    def cost(self) -> float | None: ...

    @property
    def stop_reason(self) -> StopReason: ...

    @property
    def output(self) -> BaseModel | None: ...

    @property
    def session_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class AgentInput:
    """One task attempt with its prepared, isolated sandbox, trial and run metadata."""

    task_id: str
    prompt: str
    sandbox: Sandbox
    trial: int
    metadata: Mapping[str, object]


def _validate_agent_output(text: object, cost: object, usage: object, stop_reason: object) -> None:
    if not isinstance(text, str):
        raise TypeError("agent output text must be a string")
    if cost is not None:
        if isinstance(cost, bool) or not isinstance(cost, int | float):
            raise ValueError("agent output cost must be finite and nonnegative")
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("agent output cost must be finite and nonnegative")
    if usage is not None and not isinstance(usage, Usage):
        raise TypeError("agent output usage must be a Usage value")
    if not isinstance(stop_reason, str) or stop_reason not in _STOP_REASONS:
        raise ValueError("agent output stop_reason is unsupported")


@dataclass(frozen=True, slots=True)
class AgentOutput:
    """Custom agent output with optional reported metrics.

    Leave cost, usage and events as None when unreported. Set events=() to report zero tool calls.
    """

    text: str
    cost: float | None = field(default=None, kw_only=True)
    usage: Usage | None = field(default=None, kw_only=True)
    events: tuple[Event, ...] | None = field(default=None, kw_only=True)
    files: tuple[FileChange, ...] = field(default=(), kw_only=True)
    stop_reason: StopReason = field(default="done", kw_only=True)
    output: BaseModel | None = field(default=None, kw_only=True)
    session_id: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        _validate_agent_output(self.text, self.cost, self.usage, self.stop_reason)


class EvalAgent(ABC):
    """Subclass and implement run to evaluate your own agent beside EvalTarget instances.

    The evaluator can invoke the same instance concurrently for independent task attempts.
    """

    def __init__(self, name: str, *, model: str | None = None) -> None:
        _require_text(name, "agent name")
        if model is not None:
            _require_text(model, "agent model")
        self._name = name
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str | None:
        return self._model

    @abstractmethod
    async def run(self, task: AgentInput) -> AgentOutput:
        """Run one task using its prepared sandbox and return output with any known metrics."""
        ...


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_argument(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("command arguments must be strings")


def _require_timeout(timeout: float) -> None:
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("check timeout must be a positive finite number")


@dataclass(frozen=True, slots=True)
class CommandCheck:
    command: Sequence[str]
    timeout: float = 30

    def __post_init__(self) -> None:
        if isinstance(self.command, str | bytes) or not self.command:
            raise ValueError("command must be a non-empty sequence of arguments")
        for argument in self.command:
            _require_argument(argument)
        _require_text(self.command[0], "command program")
        _require_timeout(self.timeout)
        object.__setattr__(self, "command", tuple(self.command))


@dataclass(frozen=True, slots=True)
class PredicateCheck:
    predicate: Callable[[EvalOutput], bool | Awaitable[bool]]

    def __post_init__(self) -> None:
        if not callable(self.predicate):
            raise TypeError("predicate must be callable")


@dataclass(frozen=True, slots=True)
class JudgeCheck:
    rubric: str
    model: str
    timeout: float = 30

    def __post_init__(self) -> None:
        _require_text(self.rubric, "rubric")
        _require_text(self.model, "judge model")
        _require_timeout(self.timeout)


EvalCheck: TypeAlias = CommandCheck | PredicateCheck | JudgeCheck


def _require_check(value: object) -> None:
    if not isinstance(value, CommandCheck | PredicateCheck | JudgeCheck):
        raise TypeError("check must be a CommandCheck, PredicateCheck or JudgeCheck")


@dataclass(frozen=True, slots=True)
class EvalTask:
    id: str
    prompt: str
    check: EvalCheck = field(kw_only=True)
    workspace: str | os.PathLike[str] | None = None
    files: Mapping[str, str | bytes] = field(default_factory=lambda: MappingProxyType({}))
    setup: Sequence[Sequence[str]] = ()

    def __post_init__(self) -> None:
        _require_text(self.id, "task id")
        _require_text(self.prompt, "prompt")
        _require_check(self.check)
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
        object.__setattr__(self, "setup", tuple(tuple(CommandCheck(command).command) for command in self.setup))
        if self.workspace is not None:
            object.__setattr__(self, "workspace", Path(self.workspace))


@dataclass(frozen=True, slots=True)
class EvalTarget:
    harness: Harness
    model: str
    label: str | None = None
    options: HarnessOptions | None = None
    tools: Sequence[Callable[..., object]] = ()

    def __post_init__(self) -> None:
        require_harness(self.harness)
        _require_text(self.model, "target model")
        if self.label is not None:
            _require_text(self.label, "target label")
        object.__setattr__(self, "tools", tuple(self.tools))

    @property
    def name(self) -> str:
        return self.label if self.label is not None else f"{self.harness.value}:{self.model}"


class EvalStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CheckResult:
    passed: bool
    detail: str
    judge_usage: Usage | None = field(default_factory=Usage)
    judge_cost: float | None = 0.0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvalRun:
    run_id: str
    task_id: str
    target: str
    trial: int
    status: EvalStatus
    result: Result | AgentOutput | None
    check: CheckResult | None
    duration: float
    detail: str

    @property
    def stop_reason(self) -> StopReason | None:
        return self.result.stop_reason if self.result is not None else None

    @property
    def cost(self) -> float | None:
        return self.result.cost if self.result is not None else None

    @property
    def usage(self) -> Usage | None:
        return self.result.usage if self.result is not None else None

    @property
    def tool_calls(self) -> int | None:
        if self.result is None or self.result.events is None:
            return None
        return sum(isinstance(event, ToolCall) for event in self.result.events)


@dataclass(frozen=True, slots=True)
class EvalSummary:
    target: str
    total: int
    passed: int
    errors: int
    cost: float | None
    usage: Usage | None
    tool_calls: int | None
    mean_duration: float
    metrics_complete: bool
    judge_cost: float | None
    judge_usage: Usage | None

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def _usage_sum(usages: Sequence[Usage | None]) -> Usage | None:
    if any(usage is None for usage in usages):
        return None
    known: Final = tuple(usage for usage in usages if usage is not None)
    return Usage(
        input_tokens=sum(usage.input_tokens for usage in known),
        output_tokens=sum(usage.output_tokens for usage in known),
        calls=sum(usage.calls for usage in known),
    )


def _cost_sum(costs: Sequence[float | None]) -> float | None:
    return None if any(cost is None for cost in costs) else sum(cost for cost in costs if cost is not None)


def _summary(target: str, runs: tuple[EvalRun, ...]) -> EvalSummary:
    selected: Final = tuple(run for run in runs if run.target == target)
    results: Final = tuple(run.result for run in selected if run.result is not None)
    checks: Final = tuple(run.check for run in selected if run.check is not None)
    tool_counts: Final = tuple(run.tool_calls for run in selected if run.result is not None)
    return EvalSummary(
        target=target,
        total=len(selected),
        passed=sum(run.status is EvalStatus.PASS for run in selected),
        errors=sum(run.status is EvalStatus.ERROR for run in selected),
        cost=_cost_sum(tuple(result.cost for result in results)),
        usage=_usage_sum(tuple(result.usage for result in results)),
        tool_calls=None
        if any(count is None for count in tool_counts)
        else sum(count for count in tool_counts if count is not None),
        mean_duration=sum(run.duration for run in selected) / len(selected),
        metrics_complete=all(
            run.cost is not None and run.usage is not None and run.tool_calls is not None for run in selected
        )
        and all(check.judge_cost is not None and check.judge_usage is not None for check in checks),
        judge_cost=_cost_sum(tuple(check.judge_cost for check in checks)),
        judge_usage=_usage_sum(tuple(check.judge_usage for check in checks)),
    )


def _usage_json(usage: Usage | None) -> Mapping[str, object] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "calls": usage.calls,
    }


def _run_json(run: EvalRun) -> Mapping[str, object]:
    return {
        "run_id": run.run_id,
        "task_id": run.task_id,
        "target": run.target,
        "trial": run.trial,
        "status": run.status.value,
        "stop_reason": run.stop_reason,
        "cost": run.cost,
        "usage": _usage_json(run.usage),
        "tool_calls": run.tool_calls,
        "duration": run.duration,
        "detail": run.detail,
        "text": run.result.text if run.result is not None else None,
        "files": (
            tuple({"path": change.path, "kind": change.kind, "diff": change.diff} for change in run.result.files)
            if run.result is not None
            else ()
        ),
        "check": (
            {
                "passed": run.check.passed,
                "detail": run.check.detail,
                "error": run.check.error,
                "judge_cost": run.check.judge_cost,
                "judge_usage": _usage_json(run.check.judge_usage),
            }
            if run.check is not None
            else None
        ),
    }


def _summary_json(summary: EvalSummary) -> Mapping[str, object]:
    return {
        "target": summary.target,
        "total": summary.total,
        "passed": summary.passed,
        "errors": summary.errors,
        "pass_rate": summary.pass_rate,
        "cost": summary.cost,
        "usage": _usage_json(summary.usage),
        "tool_calls": summary.tool_calls,
        "mean_duration": summary.mean_duration,
        "metrics_complete": summary.metrics_complete,
        "judge_cost": summary.judge_cost,
        "judge_usage": _usage_json(summary.judge_usage),
    }


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _summary_row(summary: EvalSummary) -> str:
    incomplete: Final = "*" if not summary.metrics_complete else ""
    cost: Final = f"{summary.cost:.4f}{incomplete}" if summary.cost is not None else "unknown"
    judge_cost: Final = f"{summary.judge_cost:.4f}{incomplete}" if summary.judge_cost is not None else "unknown"
    input_tokens: Final = summary.usage.input_tokens if summary.usage is not None else "unknown"
    output_tokens: Final = summary.usage.output_tokens if summary.usage is not None else "unknown"
    model_calls: Final = summary.usage.calls if summary.usage is not None else "unknown"
    tool_calls: Final = summary.tool_calls if summary.tool_calls is not None else "unknown"
    return (
        f"| {_cell(summary.target)} | {summary.passed}/{summary.total} | {summary.pass_rate:.1%} | {summary.errors} "
        f"| {cost} | {input_tokens} | {output_tokens} "
        f"| {model_calls} | {tool_calls} | {summary.mean_duration:.2f} | {judge_cost} |"
    )


def _run_row(run: EvalRun) -> str:
    cost: Final = f"{run.cost:.4f}" if run.cost is not None else "unknown"
    return (
        f"| {_cell(run.task_id)} | {_cell(run.target)} | {run.trial} | {run.status.value} "
        f"| {run.stop_reason or '-'} | {cost} | {run.duration:.2f} | {_cell(run.detail)} |"
    )


@dataclass(frozen=True, slots=True)
class EvalScorecard:
    suite_id: str
    runs: tuple[EvalRun, ...]

    @property
    def summaries(self) -> tuple[EvalSummary, ...]:
        targets: Final = tuple(dict.fromkeys(run.target for run in self.runs))
        return tuple(_summary(target, self.runs) for target in targets)

    def to_json(self) -> str:
        return json.dumps(
            {
                "suite_id": self.suite_id,
                "summaries": tuple(_summary_json(summary) for summary in self.summaries),
                "runs": tuple(_run_json(run) for run in self.runs),
            },
            indent=2,
            allow_nan=False,
        )

    def to_markdown(self) -> str:
        summary_header: Final = (
            "| Target | Passed | Pass rate | Errors | Agent cost (USD) | Input tokens | Output tokens "
            "| Model calls | Tool calls | Mean duration (s) | Judge cost (USD) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|"
        )
        run_header: Final = (
            "| Task | Target | Trial | Result | Stop reason | Agent cost (USD) | Duration (s) | Detail |\n"
            "|---|---|---|---|---|---|---|---|"
        )
        note: Final = (
            "\n\n* Known subtotals only; some metrics are unavailable."
            if any(not summary.metrics_complete for summary in self.summaries)
            else ""
        )
        return (
            summary_header
            + "\n"
            + "\n".join(_summary_row(summary) for summary in self.summaries)
            + "\n\n"
            + run_header
            + "\n"
            + "\n".join(_run_row(run) for run in self.runs)
            + note
        )
