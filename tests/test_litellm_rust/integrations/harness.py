import asyncio
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final

import httpx
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from litellm.integrations.gcs_bucket.gcs_bucket_base import IAM_AUTH_KEY
from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger
from litellm.integrations.literal_ai import LiteralAILogger
from litellm.integrations.opentelemetry import LITELLM_REQUEST_SPAN_NAME, OpenTelemetry
from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview.purview_dlp import MicrosoftPurviewDLPGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from tests.test_litellm_rust.callback_recorder import drain_logging
from tests.test_litellm_rust.recording_server import RecordedRequest, RecordingServer


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
        self, responses: tuple[Mapping[str, object], ...] = (), blocked_url_fragment: str | None = None
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


@dataclass(frozen=True, slots=True)
class GCSLiteralAIHarness:
    gcs: GCSBucketLogger
    literal: LiteralAILogger
    storage: RecordingAsyncClient
    literal_sink: RecordingAsyncClient


@dataclass(frozen=True, slots=True)
class GenericAPIExportHarness:
    server: RecordingServer
    logger: GenericAPILogger

    @property
    def exports(self) -> tuple[RecordedRequest, ...]:
        return tuple(request for request in self.server.requests if request.path == "/logs")


@asynccontextmanager
async def gcs_literalai_harness() -> AsyncIterator[GCSLiteralAIHarness]:
    storage: Final = RecordingAsyncClient()
    literal_sink: Final = RecordingAsyncClient()
    gcs: Final = GCSBucketLogger(bucket_name="composition-bucket")
    gcs.async_httpx_client = storage
    gcs.vertex_instances[IAM_AUTH_KEY] = RecordingVertexInstance()
    literal: Final = LiteralAILogger(literalai_api_key="test-key")
    literal.async_httpx_client = literal_sink
    try:
        yield GCSLiteralAIHarness(gcs=gcs, literal=literal, storage=storage, literal_sink=literal_sink)
    finally:
        await gcs.aclose()


@dataclass(frozen=True, slots=True)
class PurviewHarness:
    graph: RecordingAsyncClient
    guardrail: MicrosoftPurviewDLPGuardrail


def purview_harness(name: str) -> PurviewHarness:
    graph: Final = RecordingAsyncClient(
        responses=({"access_token": "test-token", "expires_in": 3600}, {}, {}, {}),
        blocked_url_fragment="processContent",
    )
    graph.release.clear()
    guardrail: Final = MicrosoftPurviewDLPGuardrail(
        guardrail_name=name,
        tenant_id="test-tenant",
        client_id="test-client",
        client_secret="test-secret",
        event_hook=GuardrailEventHooks.logging_only,
        default_on=True,
    )
    guardrail.async_handler = graph
    return PurviewHarness(graph=graph, guardrail=guardrail)


async def wait_for_audits(graph: RecordingAsyncClient, count: int = 2, timeout: float = 3) -> tuple[RecordedPost, ...]:
    async with asyncio.timeout(timeout):
        while len(tuple(post for post in graph.posts if "processContent" in post.url)) < count:
            await asyncio.sleep(0.01)
    return tuple(post for post in graph.posts if "processContent" in post.url)


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
