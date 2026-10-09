import asyncio
import json
import struct
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from itertools import chain
from typing import Final
from zlib import crc32

import httpx
import pytest
import respx
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.bedrock.passthrough.transformation import BedrockPassthroughConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.vllm.passthrough.transformation import VLLMPassthroughConfig
from litellm.passthrough.main import (
    AsyncPassthroughStreamingResponse,
    PassthroughStreamingResponse,
    allm_passthrough_route,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

_API_BASE: Final = "http://vllm-upstream.test:8090"


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def test_hosted_vllm_resolves_vllm_passthrough_config() -> None:
    cfg: Final = ProviderConfigManager.get_provider_passthrough_config(
        model="hosted_vllm/my-deployment",
        provider=LlmProviders.HOSTED_VLLM,
    )
    assert isinstance(cfg, VLLMPassthroughConfig)


@pytest.mark.asyncio
async def test_allm_passthrough_route_hosted_vllm_sends_normalized_model(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(f"{_API_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client: Final = AsyncHTTPHandler()
    response: Final = await allm_passthrough_route(
        method="POST",
        endpoint="v1/chat/completions",
        model="hosted_vllm/my-deployment",
        api_base=_API_BASE,
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        client=client,
    )
    assert response.status_code == 200
    assert route.call_count == 1
    outbound: Final = json.loads(route.calls[0].request.content)
    assert outbound["model"] == "my-deployment"
    assert outbound["messages"] == [{"role": "user", "content": "Hello"}]


def _timing_logging() -> Logging:
    return Logging(
        model="anthropic.claude-sonnet-5-v1:0",
        messages=[],
        stream=False,
        call_type="allm_passthrough_route",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="synthetic-timing",
        function_id="synthetic-timing",
    )


def _bedrock_frame(kind: str, payload: dict[str, object]) -> bytes:
    headers: Final = {":event-type": kind, ":content-type": "application/json", ":message-type": "event"}
    parts: Final = tuple(
        bytes([len(key)]) + key.encode() + b"\x07" + struct.pack("!H", len(value)) + value.encode()
        for key, value in headers.items()
    )
    header_bytes: Final = b"".join(parts)
    body: Final = json.dumps(payload).encode()
    prelude: Final = struct.pack("!II", 16 + len(header_bytes) + len(body), len(header_bytes))
    message: Final = prelude + struct.pack("!I", crc32(prelude)) + header_bytes + body
    return message + struct.pack("!I", crc32(message))


class _TerminalObservation(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.finished: Final = asyncio.Event()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.count += 1
        self.finished.set()


@pytest.mark.asyncio
async def test_endpoint_inferred_stream_logs_first_chunk_once_after_consumption() -> None:
    first: Final = _bedrock_frame("messageStart", {"role": "assistant"})
    rest: Final = b"".join(
        (
            _bedrock_frame("contentBlockDelta", {"delta": {"text": "synthetic reply"}, "contentBlockIndex": 0}),
            _bedrock_frame("messageStop", {"stopReason": "end_turn"}),
            _bedrock_frame("metadata", {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}),
        )
    )
    logging: Final = _timing_logging()
    observer: Final = _TerminalObservation()
    reader: Final = InMemoryMetricReader()
    meter: Final = MeterProvider(metric_readers=[reader])
    otel: Final = OpenTelemetryV2(
        config=OpenTelemetryV2Config(exporter="in_memory", enable_metrics=True),
        meter_provider=meter,
    )
    logging.dynamic_async_success_callbacks = [otel, observer]

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            assert logging.completion_start_time is None, "Success logging ran before the first chunk"
            yield first
            yield rest

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Body(), headers={"content-type": "application/vnd.amazon.eventstream"})

    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    try:
        response: Final = await allm_passthrough_route(
            method="POST",
            endpoint="model/anthropic.claude-sonnet-5-v1:0/converse-stream",
            model="bedrock/anthropic.claude-sonnet-5-v1:0",
            api_base="http://bedrock.test",
            aws_access_key_id="synthetic",
            aws_secret_access_key="synthetic",
            aws_region_name="us-east-1",
            json={"messages": [{"role": "user", "content": [{"text": "synthetic"}]}]},
            client=client,
            litellm_logging_obj=logging,
        )
        assert isinstance(response, AsyncPassthroughStreamingResponse)
        await asyncio.sleep(0)
        await GLOBAL_LOGGING_WORKER.flush()
        assert logging.model_call_details["stream"] is True
        assert logging.completion_start_time is None
        assert await anext(response) == first
        observed: Final = logging.completion_start_time
        assert observed is not None
        assert await anext(response) == rest
        assert logging.completion_start_time == observed
        with pytest.raises(StopAsyncIteration):
            await anext(response)
        await asyncio.wait_for(observer.finished.wait(), timeout=2)
        assert observer.count == 1
        data: Final = reader.get_metrics_data()
        assert data is not None
        scopes: Final = tuple(chain.from_iterable(resource.scope_metrics for resource in data.resource_metrics))
        metrics: Final = tuple(chain.from_iterable(scope.metrics for scope in scopes))
        (timing,) = (metric for metric in metrics if metric.name == "gen_ai.server.time_to_first_token")
        (point,) = timing.data.data_points
        assert point.count == 1
        assert point.sum == pytest.approx(
            (observed - logging.model_call_details["api_call_start_time"]).total_seconds(), abs=1e-6
        )
    finally:
        await client.close()
        meter.shutdown()


@pytest.mark.parametrize("existing", [None, datetime(2026, 1, 1, 0, 0, 1)])
@pytest.mark.parametrize("use_send", [False, True])
def test_sync_passthrough_preserves_first_chunk_observation(existing: datetime | None, use_send: bool) -> None:
    logging: Final = _timing_logging()
    if existing is not None:
        logging.update_completion_start_time(existing)

    class Body(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b"first"
            yield b"second"

    response: Final = httpx.Response(200, stream=Body())
    stream: Final = PassthroughStreamingResponse(response, logging, BedrockPassthroughConfig())
    assert (stream.send(None) if use_send else next(stream)) == b"first"
    observed: Final = logging.completion_start_time
    assert observed is not None
    assert logging.model_call_details["completion_start_time"] == observed
    if existing is not None:
        assert observed == existing
    assert next(stream) == b"second"
    assert logging.completion_start_time == observed
    stream.close()


@pytest.mark.asyncio
async def test_async_asend_stamps_first_chunk_before_anext() -> None:
    logging: Final = _timing_logging()

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"first"
            yield b"second"

    async def opened_response() -> httpx.Response:
        return httpx.Response(200, stream=Body(), request=httpx.Request("POST", "http://bedrock.test"))

    stream: Final = await AsyncPassthroughStreamingResponse(opened_response(), logging, BedrockPassthroughConfig())
    assert await stream.asend(None) == b"first"
    observed: Final = logging.completion_start_time
    assert observed is not None
    assert await anext(stream) == b"second"
    assert logging.completion_start_time == observed
    await stream.aclose()


@pytest.mark.parametrize("consume", [False, True])
@pytest.mark.asyncio
async def test_empty_or_unconsumed_async_stream_has_no_first_chunk(consume: bool) -> None:
    logging: Final = _timing_logging()
    response: Final = httpx.Response(200, content=b"", request=httpx.Request("POST", "http://bedrock.test"))

    async def opened_response() -> httpx.Response:
        return response

    stream: Final = await AsyncPassthroughStreamingResponse(opened_response(), logging, BedrockPassthroughConfig())
    if consume:
        assert [chunk async for chunk in stream] == []
    await stream.aclose()
    assert logging.completion_start_time is None


@pytest.mark.parametrize("consume", [False, True])
def test_empty_or_unconsumed_stream_has_no_first_chunk(consume: bool) -> None:
    logging: Final = _timing_logging()
    response: Final = httpx.Response(200, content=b"")
    stream: Final = PassthroughStreamingResponse(response, logging, BedrockPassthroughConfig())
    if consume:
        assert list(stream) == []
    stream.close()
    assert logging.completion_start_time is None
