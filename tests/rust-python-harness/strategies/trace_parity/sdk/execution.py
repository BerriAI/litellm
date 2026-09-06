from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Final, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from ....shared.parity.replay import replay_server
from ....shared.reporting.models import Surface
from ....shared.tracing.native import native_trace_events
from ....shared.tracing.profiler import FunctionTraceEvent, profile_python
from ....shared.tracing.steps import Engine, PipelineProjection, PipelineStep, pipeline_projection
from ..models import RouteSpec, TraceExecutionFailure, TraceMode, TraceScenario
from ..reporting import TraceComparisonArtifact


class SdkCall(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


class _WorkerCommand(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    asynchronous: bool
    kwargs: dict[str, JsonValue]


class _WorkerTraceEvent(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: int
    parent_id: int | None
    function: str
    module_path: str | None = None
    file: str | None = None
    line: int | None = None

    def event(self) -> FunctionTraceEvent:
        return FunctionTraceEvent(self.id, self.parent_id, self.function, self.module_path, self.file, self.line)


class _WorkerSuccess(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    status: Literal["ok"] = "ok"
    python: tuple[_WorkerTraceEvent, ...]
    native: tuple[_WorkerTraceEvent, ...]


class _WorkerFailure(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    status: Literal["error"] = "error"
    message: str


_WORKER_PREFIX: Final = "LITELLM_TRACE_RESULT "


def _invoke(function: SdkCall, kwargs: dict[str, object], *, asynchronous: bool) -> object:
    async def invoke_async() -> object:
        return await cast(Awaitable[object], function(**kwargs))

    if asynchronous:
        return asyncio.run(invoke_async())
    return function(**kwargs)


def _entrypoint(spec: RouteSpec, engine: Engine, *, asynchronous: bool) -> SdkCall | TraceExecutionFailure:
    import litellm
    from litellm.anthropic_interface import messages as sdk_messages
    from litellm.rust_bridge import get_native_bridge

    if engine == "rust":
        bridge: Final = cast(object | None, get_native_bridge())
        if bridge is None:
            return TraceExecutionFailure("rust", "native Rust bridge is required for trace parity")
        trace_bridge: Final[object | None] = getattr(bridge, "_trace", None)
        if trace_bridge is None:
            return TraceExecutionFailure("rust", "native Rust bridge must include the trace-parity feature")
        entrypoint: Final = spec.rust_entrypoints[int(asynchronous)]
        function: Final[object | None] = getattr(trace_bridge, entrypoint, None)
        if function is None:
            return TraceExecutionFailure("rust", f"native Rust trace bridge does not expose {entrypoint}")
        return cast(SdkCall, function)
    owner: Final = sdk_messages if spec.route == "messages" else litellm
    return cast(SdkCall, getattr(owner, spec.python_entrypoints[int(asynchronous)]))


def _collect(
    function: SdkCall, kwargs: dict[str, object], engine: Engine, *, asynchronous: bool
) -> tuple[FunctionTraceEvent, ...]:
    if engine == "rust":
        return native_trace_events(_invoke(function, kwargs, asynchronous=asynchronous))
    import litellm

    with profile_python(Path(litellm.__file__).parent, threads=True) as profiler:
        _invoke(function, kwargs, asynchronous=asynchronous)
    return tuple(profiler.events)


def collect_trace(
    spec: RouteSpec, engine: Engine, *, asynchronous: bool
) -> tuple[FunctionTraceEvent, ...] | TraceExecutionFailure:
    function: Final = _entrypoint(spec, engine, asynchronous=asynchronous)
    if isinstance(function, TraceExecutionFailure):
        return function
    try:
        with replay_server() as provider:
            fixture: Final = spec.fixture(engine, provider.url)
            for response in fixture.provider_responses:
                provider.enqueue_response(response)
            kwargs: Final = {
                **fixture.kwargs,
                "api_key": "test-key",
                "api_base": provider.url,
                **({"timeout_seconds": 5} if engine == "rust" else {"timeout": 5}),
            }
            events: Final = _collect(function, kwargs, engine, asynchronous=asynchronous)
            provider.take_requests(len(fixture.provider_responses))
    except Exception as error:
        return TraceExecutionFailure(engine, f"{type(error).__name__}: {error}")
    if not events:
        return TraceExecutionFailure(engine, "trace is empty")
    return events


def _worker_event(event: FunctionTraceEvent) -> _WorkerTraceEvent:
    return _WorkerTraceEvent(
        id=event.id,
        parent_id=event.parent_id,
        function=event.function,
        module_path=event.module_path,
        file=event.file,
        line=event.line,
    )


def _public_worker(command: _WorkerCommand) -> _WorkerSuccess | _WorkerFailure:
    try:
        import litellm
        from litellm.rust_bridge import get_native_bridge

        native_events: tuple[FunctionTraceEvent, ...] = ()
        trace_bridge: object | None = None
        capture_id: int | None = None
        if os.environ.get("LITELLM_RUST") == "1":
            bridge: Final = cast(object | None, get_native_bridge())
            trace_bridge = getattr(bridge, "_trace", None)
            if trace_bridge is None:
                return _WorkerFailure(message="native Rust bridge must include the trace-parity feature")
            capture_id = cast(int, getattr(trace_bridge, "start_capture")())
        entrypoint: Final = cast(SdkCall, getattr(litellm, "aocr" if command.asynchronous else "ocr"))
        with profile_python(Path(litellm.__file__).parent, threads=True) as profiler:
            _invoke(entrypoint, cast(dict[str, object], command.kwargs), asynchronous=command.asynchronous)
        if trace_bridge is not None and capture_id is not None:
            native_events = native_trace_events(
                {"response": None, "trace": getattr(trace_bridge, "finish_capture")(capture_id)}
            )
        return _WorkerSuccess(
            python=tuple(_worker_event(event) for event in profiler.events),
            native=tuple(_worker_event(event) for event in native_events),
        )
    except Exception as error:
        return _WorkerFailure(message=f"{type(error).__name__}: {error}")


def _run_public_worker(
    engine: Engine, kwargs: dict[str, object], *, asynchronous: bool
) -> tuple[tuple[FunctionTraceEvent, ...], tuple[FunctionTraceEvent, ...]] | TraceExecutionFailure:
    command: Final = _WorkerCommand.model_validate({"asynchronous": asynchronous, "kwargs": kwargs})
    environment: Final = {**os.environ, "LITELLM_RUST": "1" if engine == "rust" else "0"}
    completed: Final = subprocess.run(
        (sys.executable, "-m", __name__, "--public-worker"),
        input=command.model_dump_json(),
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=60,
    )
    result_line: Final = next(
        (
            line.removeprefix(_WORKER_PREFIX)
            for line in completed.stdout.splitlines()
            if line.startswith(_WORKER_PREFIX)
        ),
        None,
    )
    if result_line is None:
        output: Final = "\n".join((*completed.stdout.splitlines()[-10:], *completed.stderr.splitlines()[-10:]))
        return TraceExecutionFailure(
            engine, f"public SDK trace worker failed with exit code {completed.returncode}: {output}"
        )
    try:
        payload: Final = _WorkerSuccess.model_validate_json(result_line)
    except ValidationError:
        failure: Final = _WorkerFailure.model_validate_json(result_line)
        return TraceExecutionFailure(engine, failure.message)
    return tuple(event.event() for event in payload.python), tuple(event.event() for event in payload.native)


def _collect_public_trace(
    scenario: TraceScenario, engine: Engine, *, asynchronous: bool
) -> tuple[tuple[FunctionTraceEvent, ...], tuple[FunctionTraceEvent, ...]] | TraceExecutionFailure:
    try:
        with replay_server() as provider:
            fixture: Final = scenario.fixture("python", provider.url)
            for response in fixture.provider_responses:
                provider.enqueue_response(response)
            kwargs: Final = {**fixture.kwargs, "api_key": "test-key", "api_base": provider.url, "timeout": 5}
            traces: Final = _run_public_worker(engine, kwargs, asynchronous=asynchronous)
            if isinstance(traces, TraceExecutionFailure):
                return traces
            provider.take_requests(len(fixture.provider_responses))
            return traces
    except Exception as error:
        return TraceExecutionFailure(engine, f"{type(error).__name__}: {error}")


def _combined_rust_projection(
    python_events: tuple[FunctionTraceEvent, ...],
    native_events: tuple[FunctionTraceEvent, ...],
    scenario: TraceScenario,
    mode: TraceMode,
) -> PipelineProjection:
    wrapper: Final = pipeline_projection("python", python_events, scenario.rust_wrapper_mappings)
    native: Final = pipeline_projection("rust", native_events, scenario.mappings_for(mode))
    wrapper_steps: Final = tuple(
        PipelineStep(index, None if step.parent_id is None else index - 1, step.span, step.raw)
        for index, step in enumerate(wrapper.steps)
    )
    native_offset: Final = len(wrapper_steps)
    native_parent: Final = wrapper_steps[-1].id if wrapper_steps else None
    native_steps: Final = tuple(
        PipelineStep(
            step.id + native_offset,
            native_parent if step.parent_id is None else step.parent_id + native_offset,
            step.span,
            step.raw,
        )
        for step in native.steps
    )
    return PipelineProjection((*wrapper_steps, *native_steps), wrapper.unmatched)


def _execute_public_trace(
    route: RouteSpec, scenario: TraceScenario, mode: TraceMode, surface: Surface
) -> TraceComparisonArtifact:
    asynchronous: Final = mode == "async"
    python_trace: Final = _collect_public_trace(scenario, "python", asynchronous=asynchronous)
    rust_trace: Final = _collect_public_trace(scenario, "rust", asynchronous=asynchronous)
    python_error: Final = _failure_message(python_trace)
    rust_error: Final = _failure_message(rust_trace)
    mappings: Final = scenario.mappings_for(mode)
    try:
        python: Final = (
            PipelineProjection()
            if isinstance(python_trace, TraceExecutionFailure)
            else pipeline_projection("python", python_trace[0], mappings)
        )
        rust: Final = (
            PipelineProjection()
            if isinstance(rust_trace, TraceExecutionFailure)
            else _combined_rust_projection(rust_trace[0], rust_trace[1], scenario, mode)
        )
    except ValueError as error:
        return TraceComparisonArtifact.from_traces(
            surface=surface,
            sdk_function=route.route,
            scenario=scenario.name,
            mode=mode,
            mappings=mappings,
            contract=scenario.contract,
            python=(),
            rust=(),
            python_unmatched=0,
            python_error=f"harness: {error}",
            rust_error=rust_error,
        )
    return TraceComparisonArtifact.from_traces(
        surface=surface,
        sdk_function=route.route,
        scenario=scenario.name,
        mode=mode,
        mappings=mappings,
        contract=scenario.contract,
        python=python.steps,
        rust=rust.steps,
        python_unmatched=python.unmatched,
        python_error=python_error,
        rust_error=rust_error,
    )


def _failure_message(result: tuple[object, ...] | TraceExecutionFailure) -> str | None:
    if isinstance(result, tuple):
        return None
    return f"{result.engine}: {result.message}"


def execute_trace(
    route: RouteSpec, scenario: TraceScenario, mode: TraceMode, surface: Surface
) -> TraceComparisonArtifact:
    if scenario.boundary == "public_sdk":
        return _execute_public_trace(route, scenario, mode, surface)
    asynchronous: Final = mode == "async"
    mappings: Final = scenario.mappings_for(mode)
    scenario_route: Final = RouteSpec(
        route=route.route,
        python_entrypoints=route.python_entrypoints,
        rust_entrypoints=route.rust_entrypoints,
        fixture=scenario.fixture,
    )
    python_trace: Final = collect_trace(scenario_route, "python", asynchronous=asynchronous)
    rust_trace: Final = collect_trace(scenario_route, "rust", asynchronous=asynchronous)
    python_error: Final = _failure_message(python_trace)
    rust_error: Final = _failure_message(rust_trace)
    python_events: Final = python_trace if isinstance(python_trace, tuple) else ()
    rust_events: Final = rust_trace if isinstance(rust_trace, tuple) else ()
    try:
        python: Final = pipeline_projection("python", python_events, mappings)
        rust: Final = pipeline_projection("rust", rust_events, mappings)
    except ValueError as error:
        return TraceComparisonArtifact.from_traces(
            surface=surface,
            sdk_function=route.route,
            scenario=scenario.name,
            mode=mode,
            mappings=mappings,
            contract=scenario.contract,
            python=(),
            rust=(),
            python_unmatched=0,
            python_error=f"harness: {error}",
        )
    return TraceComparisonArtifact.from_traces(
        surface=surface,
        sdk_function=route.route,
        scenario=scenario.name,
        mode=mode,
        mappings=mappings,
        contract=scenario.contract,
        python=python.steps,
        rust=rust.steps,
        python_unmatched=python.unmatched,
        python_error=python_error,
        rust_error=rust_error,
    )


def _worker_main() -> int:
    result: Final = _worker_result()
    print(f"{_WORKER_PREFIX}{result.model_dump_json()}", flush=True)
    return int(isinstance(result, _WorkerFailure))


def _worker_result() -> _WorkerSuccess | _WorkerFailure:
    try:
        command: Final = _WorkerCommand.model_validate_json(sys.stdin.read())
        return _public_worker(command)
    except Exception as error:
        return _WorkerFailure(message=f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--public-worker"]:
        raise SystemExit("usage: execution.py --public-worker")
    raise SystemExit(_worker_main())
