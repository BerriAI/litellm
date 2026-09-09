import asyncio
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.opentelemetry import LITELLM_REQUEST_SPAN_NAME, OpenTelemetry, OpenTelemetryConfig
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy.guardrails.guardrail_hooks.azure.text_moderation import (
    AzureContentSafetyTextModerationGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks
from tests.test_litellm_rust.callback_recorder import drain_logging
from tests.test_litellm_rust.contracts import (
    MESSAGES,
    MESSAGES_EVENTS,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
    OCR_RESPONSE,
    call_aocr,
    call_ocr,
)
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

RouteName = Literal["ocr-sync", "ocr-async", "messages", "messages-stream"]
DependencyProfile = Literal["required", "enterprise"]

OSS_LOGGER_NAMES: Final = frozenset(
    {
        "agentops",
        "anthropic_cache_control_hook",
        "argilla",
        "arize",
        "arize_phoenix",
        "aws_sqs",
        "azure_sentinel",
        "azure_storage",
        "bitbucket",
        "braintrust",
        "cloudzero",
        "datadog",
        "datadog_llm_observability",
        "datadog_metrics",
        "deepeval",
        "dotprompt",
        "dynamic_rate_limiter",
        "dynamic_rate_limiter_v3",
        "focus",
        "galileo",
        "gcs_bucket",
        "gcs_pubsub",
        "gitlab",
        "humanloop",
        "lago",
        "langfuse",
        "langfuse_otel",
        "langsmith",
        "langtrace",
        "levo",
        "literalai",
        "litellm_agent",
        "logfire",
        "mavvrik",
        "mlflow",
        "newrelic",
        "opentelemetry",
        "openmeter",
        "opik",
        "otel",
        "posthog",
        "prometheus",
        "s3_v2",
        "vantage",
        "vector_store_pre_call_hook",
        "weave_otel",
    }
)
ENTERPRISE_LOGGER_NAMES: Final = frozenset({"generic_api", "pagerduty", "resend_email", "sendgrid_email", "smtp_email"})
GUARDRAIL_NAMES: Final = frozenset(
    {
        "aim",
        "akto",
        "alice",
        "aporia",
        "azure/prompt_shield",
        "azure/text_moderations",
        "bedrock",
        "block_code_execution",
        "cato_networks",
        "cisco_ai_defense",
        "compresr",
        "crowdstrike_aidr",
        "custom_code",
        "deepkeep",
        "dynamoai",
        "enkryptai",
        "generic_guardrail_api",
        "grayswan",
        "guardrails_ai",
        "headroom",
        "hiddenlayer",
        "hide-secrets",
        "ibm_guardrails",
        "javelin",
        "lakera",
        "lakera_v2",
        "lasso",
        "litellm_content_filter",
        "llm_as_a_judge",
        "mcp_end_user_permission",
        "mcp_jwt_signer",
        "mcp_security",
        "microsoft_purview",
        "model_armor",
        "noma",
        "noma_v2",
        "onyx",
        "openai_moderation",
        "ovalix",
        "pangea",
        "panw_prisma_airs",
        "pillar",
        "presidio",
        "prompt_security",
        "promptguard",
        "qostodian_nexus",
        "qualifire",
        "repelloai",
        "rubrik",
        "semantic_guard",
        "singulr",
        "straiker",
        "tool_permission",
        "vigil_guard",
        "xecguard",
        "zscaler_ai_guard",
    }
)
DISCOVERED_ONLY_GUARDRAIL_NAMES: Final = frozenset({"tool_policy"})


@dataclass(frozen=True, slots=True)
class CoverageObligation:
    stable_name: str
    registration_names: tuple[str, ...]
    dependency_profile: DependencyProfile
    behavioral_cases: tuple[str, ...]


def _logger_obligations() -> Mapping[str, CoverageObligation]:
    from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry

    registry: Final = CustomLoggerRegistry.CALLBACK_CLASS_STR_TO_CLASS_TYPE
    expected_names: Final = OSS_LOGGER_NAMES | ENTERPRISE_LOGGER_NAMES
    grouped: Final[dict[type[object], tuple[str, ...]]] = {
        implementation: tuple(sorted(name for name in expected_names if registry.get(name) is implementation))
        for implementation in frozenset(registry.values())
    }
    obligations: Final = {
        names[0]: CoverageObligation(
            stable_name=names[0],
            registration_names=names,
            dependency_profile="enterprise" if set(names) & ENTERPRISE_LOGGER_NAMES else "required",
            behavioral_cases=(
                ("generic-api-success",)
                if "generic_api" in names
                else ("gcs-literalai-scheduling",)
                if "gcs_bucket" in names
                else ("gcs-literalai-scheduling",)
                if "literalai" in names
                else ("prometheus-string-registration",)
                if "prometheus" in names
                else ("otel-export",)
                if "opentelemetry" in names
                else ()
            ),
        )
        for names in grouped.values()
        if names
    }
    missing_optional: Final = ENTERPRISE_LOGGER_NAMES - registry.keys()
    unavailable: Final = {name: CoverageObligation(name, (name,), "enterprise", ()) for name in missing_optional}
    return MappingProxyType({**obligations, **unavailable})


LOGGER_OBLIGATIONS: Final = _logger_obligations()
GUARDRAIL_OBLIGATIONS: Final = MappingProxyType(
    {
        name: CoverageObligation(
            stable_name=name,
            registration_names=(name,),
            dependency_profile="required",
            behavioral_cases=(
                ("azure-text-moderation",)
                if name == "azure/text_moderations"
                else ("crowdstrike-redaction-native-chat",)
                if name == "crowdstrike_aidr"
                else ("rubrik-block-native-chat",)
                if name == "rubrik"
                else ("purview-audit-native-chat",)
                if name == "microsoft_purview"
                else ("content-filter-block",)
                if name == "litellm_content_filter"
                else ()
            ),
        )
        for name in GUARDRAIL_NAMES | DISCOVERED_ONLY_GUARDRAIL_NAMES
    }
)


@dataclass(frozen=True, slots=True)
class Route:
    name: RouteName
    call_type: str
    provider_model: str
    provider_response: dict[str, object]
    response_text: str
    provider: str
    expected_cost: float
    fires_async_hooks: bool

    async def invoke(self, server: RecordingServer, **kwargs: object) -> object:
        match self.name:
            case "ocr-sync":
                return await asyncio.to_thread(call_ocr, server, **kwargs)
            case "ocr-async":
                return await call_aocr(server, **kwargs)
            case "messages":
                return await _call_messages(server, **kwargs)
            case "messages-stream":
                stream: Final = await self.open_stream(server, **kwargs)
                return [chunk async for chunk in stream]

    async def open_stream(self, server: RecordingServer, **kwargs: object) -> AsyncIterator[object]:
        if self.name != "messages-stream":
            raise ValueError(f"{self.name} is not a streaming route")
        stream: Final = await _call_messages(server, stream=True, **kwargs)
        if not isinstance(stream, AsyncIterator):
            raise TypeError(f"Expected async stream, got {type(stream).__name__}")
        return stream


@dataclass(frozen=True, slots=True)
class CompositionCase:
    stable_id: str
    integrations: tuple[str, ...]
    route: Route
    scenario: Literal["success"]


@dataclass(frozen=True, slots=True)
class RunObservation:
    call_type: str
    model: str
    response_cost: float
    response_text: str
    provider_body: Mapping[str, object]
    native_dispatch: bool


@dataclass(frozen=True, slots=True)
class RecordedPost:
    url: str
    headers: Mapping[str, str]
    body: object


class RecordingAsyncClient:
    def __init__(
        self,
        responses: tuple[Mapping[str, object], ...] = (),
        blocked_url_fragment: str | None = None,
    ) -> None:
        self.posts: list[RecordedPost] = []
        self._responses = list(responses)
        self._blocked_url_fragment = blocked_url_fragment
        self.accepted = threading.Event()
        self.release = threading.Event()
        self.release.set()

    async def post(self, url: str, **kwargs: object) -> httpx.Response:
        data: Final = kwargs.get("data")
        json_body: Final = kwargs.get("json")
        body: Final = json_body if json_body is not None else data
        headers_value: Final = kwargs.get("headers")
        headers: Final = headers_value if isinstance(headers_value, Mapping) else {}
        self.posts.append(RecordedPost(url=url, headers=headers, body=body))
        should_block: Final = self._blocked_url_fragment is None or self._blocked_url_fragment in url
        if should_block:
            self.accepted.set()
            released: Final = await asyncio.to_thread(self.release.wait, 10)
            if not released:
                raise TimeoutError(f"Timed out releasing POST {url}")
        payload: Final = self._responses.pop(0) if self._responses else {}
        response_headers: Final = {"etag": '"test-scope"'} if "protectionScopes/compute" in url else {}
        return httpx.Response(200, json=payload, headers=response_headers, request=httpx.Request("POST", url))


class RecordingVertexInstance:
    async def _ensure_access_token_async(self, **kwargs: object) -> tuple[str, str]:
        return "test-access-token", "test-project"

    def _get_token_and_url(self, **kwargs: object) -> tuple[str, str]:
        return "test-access-token", ""


class AsyncBoundaryLogger(CustomLogger):
    def __init__(self, action: Callable[[], Awaitable[None]]) -> None:
        self._action = action

    async def async_log_success_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._action()


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
    fires_async_hooks=True,
)
ALL_ROUTES: Final = (OCR_SYNC, OCR_ASYNC, MESSAGES_ROUTE, MESSAGES_STREAM)
ASYNC_ROUTES: Final = tuple(route for route in ALL_ROUTES if route.fires_async_hooks)
NON_STREAM_ASYNC_ROUTES: Final = (OCR_ASYNC, MESSAGES_ROUTE)
EXPORT_COMPOSITIONS: Final = tuple(
    CompositionCase(
        stable_id=f"otel-prometheus-generic-api-{route.name}-success",
        integrations=("opentelemetry", "prometheus", "generic_api"),
        route=route,
        scenario="success",
    )
    for route in ASYNC_ROUTES
)
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
    recording_server.default_response = ResponseSpec(
        body=route.provider_response,
        events=MESSAGES_EVENTS if route.name == "messages-stream" else (),
    )
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


class ReviewGuardrail(CustomGuardrail):
    def __init__(self, review: Callable[[object], Awaitable[object]]) -> None:
        super().__init__(guardrail_name="rust-review", event_hook=GuardrailEventHooks.post_call, default_on=True)
        self._review = review
        self.call_types: list[object] = []

    async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
        self.call_types.append(call_type)
        return await self._review(response)


class MutatingFailingLogger(CustomLogger):
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        payload: Final = kwargs["standard_logging_object"]
        payload["metadata"]["composition_marker"] = "visible-before-failure"
        raise RuntimeError("synthetic callback failure")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        payload: Final = kwargs["standard_logging_object"]
        payload["metadata"]["composition_marker"] = "visible-before-failure"
        raise RuntimeError("synthetic callback failure")
