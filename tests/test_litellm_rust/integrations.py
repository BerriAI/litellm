import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Final, Literal

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.opentelemetry import LITELLM_REQUEST_SPAN_NAME, OpenTelemetry, OpenTelemetryConfig
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy.guardrails.guardrail_hooks.azure.text_moderation import (
    AzureContentSafetyTextModerationGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs
from tests.test_litellm_rust.callback_recorder import drain_logging
from tests.test_litellm_rust.contracts import (
    MESSAGES,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
    OCR_RESPONSE,
    call_native_aocr,
    call_native_ocr,
)
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

RouteName = Literal["ocr-sync", "ocr-async", "messages", "messages-stream"]
GuardrailObservation = tuple[Literal["request", "response"], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class Route:
    name: RouteName
    call_type: str
    provider_model: str
    provider_response: dict[str, object]
    response_text: str
    provider: str
    expected_cost: float
    logging_only_scan: GuardrailObservation
    fires_async_hooks: bool

    async def invoke(self, server: RecordingServer, **kwargs: object) -> object:
        match self.name:
            case "ocr-sync":
                return await asyncio.to_thread(call_native_ocr, server, **kwargs)
            case "ocr-async":
                return await call_native_aocr(server, **kwargs)
            case "messages":
                return await _call_messages(server, **kwargs)
            case "messages-stream":
                stream: Final = await _call_messages(server, stream=True, **kwargs)
                return [chunk async for chunk in stream]


async def _call_messages(server: RecordingServer, **kwargs: object) -> object:
    return await litellm.anthropic.messages.acreate(
        model=MESSAGES_MODEL,
        messages=MESSAGES,
        max_tokens=64,
        api_key="test-key",
        api_base=server.base_url,
        **kwargs,
    )


MESSAGES_COST: Final = 5 * 3e-06 + 4 * 1.5e-05
OCR_COST: Final = 0.004

OCR_SYNC: Final = Route(
    name="ocr-sync",
    call_type="ocr",
    provider_model="mistral-ocr-latest",
    provider_response=OCR_RESPONSE,
    response_text="native OCR response",
    provider="mistral",
    expected_cost=OCR_COST,
    logging_only_scan=("response", ("native OCR response",)),
    fires_async_hooks=False,
)
OCR_ASYNC: Final = Route(
    name="ocr-async",
    call_type="aocr",
    provider_model="mistral-ocr-latest",
    provider_response=OCR_RESPONSE,
    response_text="native OCR response",
    provider="mistral",
    expected_cost=OCR_COST,
    logging_only_scan=("response", ("native OCR response",)),
    fires_async_hooks=True,
)
MESSAGES_ROUTE: Final = Route(
    name="messages",
    call_type="anthropic_messages",
    provider_model="claude-sonnet-4-5-20250929",
    provider_response=MESSAGES_RESPONSE,
    response_text="Hello from native Messages",
    provider="anthropic",
    expected_cost=MESSAGES_COST,
    logging_only_scan=("request", ("Hello",)),
    fires_async_hooks=True,
)
MESSAGES_STREAM: Final = Route(
    name="messages-stream",
    call_type="anthropic_messages",
    provider_model="claude-sonnet-4-5-20250929",
    provider_response=MESSAGES_RESPONSE,
    response_text="Hello from native Messages",
    provider="anthropic",
    expected_cost=MESSAGES_COST,
    logging_only_scan=("request", ("Hello",)),
    fires_async_hooks=True,
)
ALL_ROUTES: Final = (OCR_SYNC, OCR_ASYNC, MESSAGES_ROUTE, MESSAGES_STREAM)
ASYNC_ROUTES: Final = tuple(route for route in ALL_ROUTES if route.fires_async_hooks)
NON_STREAM_ASYNC_ROUTES: Final = (OCR_ASYNC, MESSAGES_ROUTE)
AZURE_MODERATION_ALLOW_RESPONSE: Final = {
    "blocklistsMatch": [],
    "categoriesAnalysis": [
        {"category": "Hate", "severity": 0},
        {"category": "Sexual", "severity": 0},
        {"category": "SelfHarm", "severity": 0},
        {"category": "Violence", "severity": 0},
    ],
}
AZURE_MODERATION_BLOCK_RESPONSE: Final = {
    **AZURE_MODERATION_ALLOW_RESPONSE,
    "categoriesAnalysis": [{"category": "Violence", "severity": 6}],
}


def route_id(route: Route) -> str:
    return route.name


def azure_text_moderation(server: RecordingServer) -> AzureContentSafetyTextModerationGuardrail:
    return AzureContentSafetyTextModerationGuardrail(
        guardrail_name="azure-text-review",
        api_key="test-azure-key",
        api_base=server.base_url,
        event_hook=GuardrailEventHooks.post_call,
    )


@pytest.fixture
def provider(recording_server: RecordingServer, route: Route) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=route.provider_response)
    return recording_server


@dataclass(frozen=True, slots=True)
class OtelHarness:
    logger: OpenTelemetry
    exporter: InMemorySpanExporter

    def spans(self, name: str = LITELLM_REQUEST_SPAN_NAME) -> tuple[ReadableSpan, ...]:
        return tuple(span for span in self.exporter.get_finished_spans() if span.name == name)

    async def wait_for_spans(
        self, name: str = LITELLM_REQUEST_SPAN_NAME, count: int = 1, timeout: float = 10
    ) -> tuple[ReadableSpan, ...]:
        deadline: Final = time.monotonic() + timeout
        while len(self.spans(name)) < count:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {count} {name} spans; saw {self.spans(name)}")
            await asyncio.sleep(0.01)
        await drain_logging()
        return self.spans(name)


@pytest.fixture
def otel(isolate_rust_state: ExitStack) -> OtelHarness:
    exporter: Final = InMemorySpanExporter()
    tracer_provider: Final = TracerProvider()
    isolate_rust_state.callback(tracer_provider.shutdown)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    logger: Final = OpenTelemetry(config=OpenTelemetryConfig(exporter=exporter), tracer_provider=tracer_provider)
    return OtelHarness(logger=logger, exporter=exporter)


@pytest.fixture
def prometheus() -> PrometheusLogger:
    return PrometheusLogger()


def metric_value(name: str, **labels: str) -> float:
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name and all(sample.labels.get(key) == value for key, value in labels.items()):
                return sample.value
    return 0.0


class RecordingGuardrail(CustomGuardrail):
    def __init__(self, guardrail_name: str = "rust-review", fail_with: Exception | None = None) -> None:
        super().__init__(
            guardrail_name=guardrail_name,
            event_hook=GuardrailEventHooks.logging_only,
            default_on=True,
        )
        self.observations: list[GuardrailObservation] = []
        self._fail_with = fail_with

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None) -> GenericGuardrailAPIInputs:
        self.observations.append((input_type, tuple(inputs.get("texts") or ())))
        if self._fail_with is not None:
            raise self._fail_with
        return inputs


class ReviewGuardrail(CustomGuardrail):
    def __init__(self, review: Callable[[object], Awaitable[object]]) -> None:
        super().__init__(guardrail_name="rust-review", event_hook=GuardrailEventHooks.post_call, default_on=True)
        self._review = review
        self.call_types: list[object] = []

    async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
        self.call_types.append(call_type)
        return await self._review(response)
