import asyncio
from typing import Final, Literal

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry, Histogram

from litellm._internal_context import request_parent_otel_span
from litellm.litellm_core_utils import callback_timing


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "failure", "cancelled"))
async def test_callback_timing_observes_once_when_hook_exits(
    monkeypatch: pytest.MonkeyPatch,
    outcome: Literal["success", "failure", "cancelled"],
) -> None:
    registry: Final = CollectorRegistry()
    histogram: Final = Histogram(
        "litellm_callback_execution_seconds",
        "Callback hook duration",
        ("callback_name", "hook", "execution_path"),
        registry=registry,
    )
    labels: Final = {
        "callback_name": "callbacks.sink",
        "hook": "async_log_success_event",
        "execution_path": "background",
    }
    monkeypatch.setattr(callback_timing, "_callback_timing_enabled", lambda: True)
    monkeypatch.setattr(callback_timing, "_load_histogram", lambda: histogram)
    inside: Final = asyncio.Event()
    release: Final = asyncio.Event()
    failure: Final = RuntimeError("callback failed")

    async def run_hook() -> str:
        with callback_timing.measure_callback_execution(labels["callback_name"], labels["hook"], "background"):
            inside.set()
            await release.wait()
            if outcome == "failure":
                raise failure
            return "callback result"

    task: Final = asyncio.create_task(run_hook())
    try:
        await asyncio.wait_for(inside.wait(), timeout=5)
        assert registry.get_sample_value("litellm_callback_execution_seconds_count", labels) is None
        if outcome == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "failure":
            release.set()
            with pytest.raises(RuntimeError) as caught:
                await task
            assert caught.value is failure
        else:
            release.set()
            assert await task == "callback result"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert registry.get_sample_value("litellm_callback_execution_seconds_count", labels) == 1


def test_callback_stage_span_anchors_to_request_and_does_not_export_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter: Final = InMemorySpanExporter()
    provider: Final = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer: Final = provider.get_tracer(__name__)
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    error: Final = RuntimeError("private payload fragment")

    try:
        with callback_timing.callback_stage_span("orphan"):
            pass
        with tracer.start_span("request") as parent, request_parent_otel_span(parent):
            with callback_timing.callback_stage_span("redact"):
                with pytest.raises(RuntimeError) as caught, callback_timing.callback_stage_span("nested"):
                    raise error
        assert caught.value is error

        spans: Final = {span.name: span for span in exporter.get_finished_spans()}
        assert set(spans) == {"request", "litellm.callback.stage.redact", "litellm.callback.stage.nested"}
        redact: Final = spans["litellm.callback.stage.redact"]
        nested: Final = spans["litellm.callback.stage.nested"]
        assert redact.parent == parent.get_span_context()
        assert nested.parent == redact.context
        assert not nested.events and not nested.attributes and not nested.status.description
    finally:
        provider.shutdown()
