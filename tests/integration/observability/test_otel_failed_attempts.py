from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.otlp_sink import Span, SpanSinks, recorded_spans
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server


def _unavailable(request: Request) -> Reply:
    assert "/model/" in request.target, request.target
    return Reply(
        status=503,
        headers={"x-amzn-errortype": "ServiceUnavailableException"},
        body=b'{"message":"primary region unavailable"}',
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
        b'"model":"claude-sonnet-5","content":[{"type":"text","text":"fallback ok"}],'
        b'"stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":2}}'
    )


@dataclass(frozen=True, slots=True)
class RetryRig:
    proxy: Gateway
    primary: Wire
    fallback: Wire
    sink: str


@pytest.fixture(scope="module", params=("false", "true"), ids=("legacy", "v2"))
def retry_rig(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, audit_sinks: SpanSinks
) -> Iterator[RetryRig]:
    directory: Final = tmp_path_factory.mktemp("otel-retry")
    with wire_server(_unavailable) as primary, wire_server(_completed) as fallback:
        config: Final = directory / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": "regional-retry",
                            "litellm_params": {
                                "model": "bedrock/anthropic.claude-sonnet-5",
                                "api_base": wire.url,
                                "aws_region_name": region,
                                "aws_access_key_id": "synthetic-access-key",
                                "aws_secret_access_key": "synthetic-secret",
                                "order": order,
                            },
                        }
                        for wire, region, order in ((primary, "eu-west-1", 1), (fallback, "eu-central-1", 2))
                    ],
                    "router_settings": {"num_retries": 3, "retry_after": 0, "disable_cooldowns": True},
                    "litellm_settings": {"callbacks": ["otel"]},
                    "callback_settings": {
                        "otel": {"exporter": "http/json", "endpoint": audit_sinks.operator, "use_simple_processor": True}
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
                yield RetryRig(proxy, primary, fallback, audit_sinks.operator)


def _llm_spans(sink: str, start: int) -> tuple[Span, ...]:
    return tuple(
        span
        for span in recorded_spans(sink, start)[1]
        if "gen_ai.operation.name" in span["attributes"] or span["name"] == "litellm_request"
    )


@pytest.mark.parametrize("endpoint", ("/v1/messages", "/v1/chat/completions"))
def test_every_failed_bedrock_attempt_has_its_own_destination(retry_rig: RetryRig, endpoint: str) -> None:
    start, _ = recorded_spans(retry_rig.sink)
    response: Final = retry_rig.proxy.request(
        "POST",
        endpoint,
        {"model": "regional-retry", "messages": [{"role": "user", "content": "retry regions"}], "max_tokens": 16},
    )
    assert response.status_code == 200, response.text
    assert "fallback ok" in response.text, response.text
    attempts: Final = retry_rig.primary.drain()
    assert len(attempts) == 4, attempts
    assert len(retry_rig.fallback.drain()) == 1
    spans: Final = eventually(
        lambda: _llm_spans(retry_rig.sink, start),
        lambda values: len(values) >= 5,
        seconds=15,
        return_last_on_timeout=True,
    )
    assert len(spans) == 5, spans
    ports: Final = sorted(int(str(span["attributes"].get("server.port"))) for span in spans)
    assert ports == sorted([urlsplit(retry_rig.primary.url).port] * 4 + [urlsplit(retry_rig.fallback.url).port]), spans
    assert all(span["attributes"].get("server.address") == "127.0.0.1" for span in spans), spans
    failures: Final = tuple(span for span in spans if span.get("status_code") == 2)
    assert len(failures) == 4, spans
    assert all("primary region unavailable" in str(span["attributes"].get("error.message")) for span in failures), failures
    assert all(int(str(span["attributes"]["server.port"])) == urlsplit(retry_rig.primary.url).port for span in failures)
