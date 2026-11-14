import json
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.otlp_sink import Span, SpanSinks, recorded_spans
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server


def _unavailable(request: Request, *, message: str = "primary region unavailable") -> Reply:
    if request.method == "GET":
        assert request.target == "/v1/models", request.target
        return Reply(body=b'{"object":"list","data":[]}')
    assert "/model/" in request.target or request.target == "/chat/completions", request.target
    return Reply(
        status=503,
        headers={"x-amzn-errortype": "ServiceUnavailableException"},
        body=json.dumps({"message": message}).encode(),
    )


def _completed(request: Request) -> Reply:
    assert "/model/" in request.target, request.target
    if request.target.endswith("/converse"):
        return Reply(
            body=b'{"output":{"message":{"role":"assistant","content":[{"text":"fallback ok"}]}},'
            b'"stopReason":"end_turn","usage":{"inputTokens":7,"outputTokens":2,"totalTokens":9},'
            b'"metrics":{"latencyMs":1}}'
        )
    return Reply(
        body=b'{"id":"msg-fallback","type":"message","role":"assistant",'
        b'"model":"claude-sonnet-5-5","content":[{"type":"text","text":"fallback ok"}],'
        b'"stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":2}}'
    )


@dataclass(frozen=True, slots=True)
class RetryRig:
    proxy: Gateway
    primary: Wire
    secondary: Wire
    fallback: Wire
    broken_stream: Wire
    sink: str


def _failed_stream(request: Request) -> Reply:
    if request.method == "GET":
        assert request.target == "/v1/models", request.target
        return Reply(body=b'{"object":"list","data":[]}')
    assert request.target == "/chat/completions", request.target
    assert json.loads(request.body)["stream"] is True
    return Reply(
        content_type="text/event-stream",
        body=b'data: {"id":"chatcmpl-partial","object":"chat.completion.chunk","created":0,'
        b'"model":"attempt-test","choices":[{"index":0,"delta":{"role":"assistant",'
        b'"content":"partial"},"finish_reason":null}]}\n\n'
        b'data: {"error":{"message":"stream attempt failed","type":"server_error",'
        b'"code":"internal_server_error"}}\n\n',
    )


@pytest.fixture(scope="module", params=("false", "true"), ids=("legacy", "v2"))
def retry_rig(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, audit_sinks: SpanSinks
) -> Iterator[RetryRig]:
    directory: Final = tmp_path_factory.mktemp("otel-retry")
    with (
        wire_server(_unavailable) as primary,
        wire_server(partial(_unavailable, message="secondary region unavailable")) as secondary,
        wire_server(_completed) as fallback,
        wire_server(_failed_stream) as broken_stream,
    ):
        config: Final = directory / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": name,
                            "litellm_params": {
                                "model": "bedrock/anthropic.claude-sonnet-5-5",
                                "api_base": wire.url,
                                "aws_region_name": region,
                                "aws_access_key_id": "synthetic-access-key",
                                "aws_secret_access_key": "synthetic-secret",
                                "order": order,
                            },
                        }
                        for name, wire, region, order in (
                            ("regional-retry", primary, "eu-west-1", 1),
                            ("regional-retry", fallback, "eu-central-1", 2),
                            ("recursive-primary", primary, "eu-west-1", 1),
                            ("recursive-secondary", secondary, "eu-north-1", 1),
                            ("recursive-fallback", fallback, "eu-central-1", 1),
                        )
                    ]
                    + [
                        {
                            "model_name": "stream-retry",
                            "litellm_params": {
                                "model": "openai/attempt-test",
                                "api_base": wire.url,
                                "api_key": "synthetic-openai-key",
                                "order": order,
                            },
                        }
                        for wire, order in ((primary, 1), (broken_stream, 2))
                    ],
                    "router_settings": {
                        "num_retries": 3,
                        "retry_after": 0,
                        "disable_cooldowns": True,
                        "fallbacks": [
                            {"recursive-primary": ["recursive-secondary"]},
                            {"recursive-secondary": ["recursive-fallback"]},
                        ],
                    },
                    "litellm_settings": {"callbacks": ["otel"]},
                    "callback_settings": {
                        "otel": {
                            "exporter": "http/json",
                            "endpoint": audit_sinks.operator,
                            "use_simple_processor": True,
                        }
                    },
                    "general_settings": {"master_key": "sk-integration-otel-retry", "store_model_in_db": True},
                }
            )
        )
        overrides: Final = {
            "LITELLM_OTEL_V2": str(request.param),
            "OTEL_EXPORTER": "http/json",
            "OTEL_ENDPOINT": audit_sinks.operator,
            "AWS_EC2_METADATA_DISABLED": "true",
        }
        with httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as client:
            base: Final = Gateway(client, "sk-integration-otel-retry", primary.url)
            with owned_proxy(base, directory, overrides, config=config, workers=2) as proxy:
                proxy.client.timeout = 120
                yield RetryRig(proxy, primary, secondary, fallback, broken_stream, audit_sinks.operator)


def _llm_spans(sink: str, start: int) -> tuple[Span, ...]:
    return tuple(
        span
        for span in recorded_spans(sink, start)[1]
        if "gen_ai.operation.name" in span["attributes"] or span["name"] == "litellm_request"
    )


@pytest.mark.parametrize("endpoint", ("/v1/messages", "/v1/chat/completions"))
@pytest.mark.parametrize("recursive", (False, True), ids=("ordered", "recursive"))
def test_every_failed_bedrock_attempt_has_its_own_destination(
    retry_rig: RetryRig, endpoint: str, recursive: bool
) -> None:
    start, _ = recorded_spans(retry_rig.sink)
    response: Final = retry_rig.proxy.request(
        "POST",
        endpoint,
        {
            "model": "recursive-primary" if recursive else "regional-retry",
            "messages": [{"role": "user", "content": "retry regions"}],
            "max_tokens": 16,
        },
    )
    assert response.status_code == 200, response.text
    assert "fallback ok" in response.text, response.text
    attempts: Final = tuple(request for request in retry_rig.primary.drain() if request.method == "POST")
    assert len(attempts) == 4, attempts
    assert len(retry_rig.secondary.drain()) == (4 if recursive else 0)
    assert len(retry_rig.fallback.drain()) == 1
    expected_failures: Final = 8 if recursive else 4
    spans: Final = eventually(
        lambda: _llm_spans(retry_rig.sink, start),
        lambda values: len(values) >= expected_failures + 1,
        seconds=15,
        return_last_on_timeout=True,
    )
    assert len(spans) == expected_failures + 1, spans
    ports: Final = sorted(int(str(span["attributes"].get("server.port"))) for span in spans)
    expected_ports: Final = (
        [urlsplit(retry_rig.primary.url).port] * 4
        + ([urlsplit(retry_rig.secondary.url).port] * 4 if recursive else [])
        + [urlsplit(retry_rig.fallback.url).port]
    )
    assert ports == sorted(expected_ports), spans
    assert all(span["attributes"].get("server.address") == "127.0.0.1" for span in spans), spans
    failures: Final = tuple(span for span in spans if span.get("status_code") == 2)
    assert len(failures) == expected_failures, spans
    for span in failures:
        region: Final = (
            "primary"
            if int(str(span["attributes"]["server.port"])) == urlsplit(retry_rig.primary.url).port
            else "secondary"
        )
        assert f"{region} region unavailable" in str(span["attributes"].get("error.message")), span


def test_failed_retry_stream_has_one_span_per_upstream_attempt(retry_rig: RetryRig) -> None:
    start, _ = recorded_spans(retry_rig.sink)
    response: Final = retry_rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": "stream-retry", "messages": [{"role": "user", "content": "stream retries"}], "stream": True},
    )
    assert response.status_code == 200, response.text
    assert "partial" in response.text and "stream attempt failed" in response.text, response.text
    assert len(tuple(request for request in retry_rig.primary.drain() if request.method == "POST")) == 4
    assert len(tuple(request for request in retry_rig.broken_stream.drain() if request.method == "POST")) == 1
    spans: Final = eventually(
        lambda: _llm_spans(retry_rig.sink, start),
        lambda values: len(values) >= 5,
        seconds=15,
    )
    assert len(spans) == 5, spans
    assert all(span.get("status_code") == 2 for span in spans), spans
    assert all(span.get("start_time_unix_nano", 0) > 0 for span in spans), spans
    assert len({span.get("start_time_unix_nano") for span in spans}) == 5, spans
    last_attempt: Final = max(spans, key=lambda span: span.get("start_time_unix_nano", 0))
    assert int(str(last_attempt["attributes"].get("server.port"))) == urlsplit(retry_rig.broken_stream.url).port, spans
    ports: Final = sorted(int(str(span["attributes"].get("server.port"))) for span in spans)
    assert ports == sorted([urlsplit(retry_rig.primary.url).port] * 4 + [urlsplit(retry_rig.broken_stream.url).port]), (
        spans
    )


def test_concurrent_retries_keep_every_attempt_in_its_request_trace(retry_rig: RetryRig) -> None:
    start, _ = recorded_spans(retry_rig.sink)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures: Final = tuple(
            pool.submit(
                retry_rig.proxy.request,
                "POST",
                "/v1/chat/completions",
                {"model": "regional-retry", "messages": [{"role": "user", "content": f"concurrent retries {index}"}]},
            )
            for index in range(3)
        )
        responses: Final = tuple(future.result() for future in futures)
    assert all(response.status_code == 200 for response in responses), tuple(response.text for response in responses)
    assert len(tuple(request for request in retry_rig.primary.drain() if request.method == "POST")) == 12
    assert len(retry_rig.fallback.drain()) == 3
    spans: Final = eventually(lambda: _llm_spans(retry_rig.sink, start), lambda values: len(values) >= 15, seconds=15)
    assert sorted(Counter(span["trace_id"] for span in spans).values()) == [5, 5, 5], spans
    assert sum(span.get("status_code") == 2 for span in spans) == 12, spans
