import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest


def _span_attributes(request: Request) -> Iterator[dict[str, str]]:
    if request.headers.get("content-type") != "application/x-protobuf":
        return
    batch: Final = ExportTraceServiceRequest.FromString(request.body)
    for resource_spans in batch.resource_spans:
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                yield {attribute.key: attribute.value.string_value for attribute in span.attributes}


def _matching_llm_spans(
    requests: tuple[Request, ...],
    marker: str,
    marker_key: str,
) -> Iterator[dict[str, str]]:
    for request in requests:
        for attributes in _span_attributes(request):
            if attributes.get("openinference.span.kind") == "LLM" and attributes.get(marker_key) == marker:
                yield attributes


def _arize_config(tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    arize_config: Final = {
        **config,
        "litellm_settings": {
            **config["litellm_settings"],
            "callbacks": ["arize"],
        },
    }
    config_path: Final = tmp_path / "arize-otel-v2.yaml"
    config_path.write_text(yaml.safe_dump(arize_config))
    return config_path


def _arize_environment(destination: Wire) -> dict[str, str]:
    return {
        "LITELLM_OTEL_V2": "1",
        "ARIZE_HTTP_ENDPOINT": destination.url + "/v1/traces",
        "ARIZE_SPACE_ID": "integration-space",
        "ARIZE_API_KEY": "integration-arize-key",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "span_only",
        "LITELLM_OTEL_BAGGAGE_METADATA_KEYS": "requester_metadata.trace_marker",
    }


def _collector(_request: Request) -> Reply:
    return Reply(body=b"", content_type="application/x-protobuf")


@contextmanager
def _arize_proxy(
    gateway: Gateway,
    tmp_path: Path,
    upstream: Callable[[Request], Reply],
) -> Iterator[tuple[Gateway, Wire, Wire]]:
    with wire_server(upstream) as provider, wire_server(_collector) as destination:
        config_path: Final = _arize_config(tmp_path)
        environment: Final = _arize_environment(destination)
        with owned_proxy(gateway, tmp_path, environment, config=config_path) as candidate:
            yield candidate, provider, destination


def _matching_span(
    destination: Wire,
    marker: str,
    marker_key: str = "gen_ai.response.id",
) -> dict[str, str]:
    return eventually(
        lambda: tuple(_matching_llm_spans(destination.drain(), marker, marker_key)),
        lambda spans: len(spans) == 1,
        seconds=30,
    )[0]


def _assert_tool_call_span(span_attributes: dict[str, str], marker: str) -> None:
    tool_call: Final = "llm.output_messages.0.message.tool_calls.0.tool_call."
    keys: Final = (
        tool_call + "id",
        tool_call + "function.name",
        tool_call + "function.arguments",
        "metadata",
        "litellm.metadata.trace_marker",
    )
    observed: Final = {
        key: json.loads(span_attributes[key])
        if key in (tool_call + "function.arguments", "metadata") and key in span_attributes
        else span_attributes.get(key)
        for key in keys
    }
    expected: Final = {
        tool_call + "id": "call_" + marker,
        tool_call + "function.name": "lookup_weather",
        tool_call + "function.arguments": {"city": "Paris"},
        "metadata": {"trace_marker": marker},
        "litellm.metadata.trace_marker": marker,
    }
    expected_output: Final = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_" + marker,
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
    ]
    actual: Final = {"attributes": observed, "output.value": json.loads(span_attributes["output.value"])}
    expected_values: Final = {"attributes": expected, "output.value": expected_output}
    assert actual == expected_values, f"Arize OTel v2 span values for {marker}: {actual!r}"


def test_arize_otel_v2_llm_span_carries_openinference_tool_calls_and_metadata(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "arize-otel-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        body: Final = json.loads(request.body)
        expected_tools: Final = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        assert body.get("messages") == [{"role": "user", "content": "weather in Paris?"}], (
            f"Unexpected chat request messages: {body!r}"
        )
        assert body.get("tools") == expected_tools, f"Unexpected chat request tools: {body!r}"
        assert body.get("tool_choice") == {
            "type": "function",
            "function": {"name": "lookup_weather"},
        }, f"Unexpected chat request tool_choice: {body!r}"
        return Reply(
            body=json.dumps(
                {
                    "id": marker,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_" + marker,
                                        "type": "function",
                                        "function": {
                                            "name": "lookup_weather",
                                            "arguments": '{"city": "Paris"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                }
            ).encode()
        )

    with _arize_proxy(gateway, tmp_path, upstream) as (candidate, provider, destination):
        with candidate.scenario() as scenario:
            model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "weather in Paris?"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "description": "Get weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                },
                            },
                        }
                    ],
                    "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
                    "metadata": {"trace_marker": marker},
                    "cache": {"no-cache": True},
                },
            )
            assert response.status_code == 200, response.text
            _assert_tool_call_span(_matching_span(destination, marker), marker)


def test_arize_otel_v2_responses_span_carries_openinference_tool_calls_and_metadata(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "arize-otel-responses-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        assert request.target.endswith("/responses"), request.target
        body: Final = json.loads(request.body)
        expected_tools: Final = [
            {
                "type": "function",
                "name": "lookup_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]
        assert body.get("input") == "weather in Paris?", f"Unexpected Responses request input: {body!r}"
        assert body.get("tools") == expected_tools, f"Unexpected Responses request tools: {body!r}"
        assert body.get("tool_choice") == {
            "type": "function",
            "name": "lookup_weather",
        }, f"Unexpected Responses request tool_choice: {body!r}"
        return Reply(
            body=json.dumps(
                {
                    "id": marker,
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_" + marker,
                            "call_id": "call_" + marker,
                            "name": "lookup_weather",
                            "arguments": '{"city": "Paris"}',
                            "status": "completed",
                        }
                    ],
                    "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
                }
            ).encode()
        )

    with _arize_proxy(gateway, tmp_path, upstream) as (candidate, provider, destination):
        with candidate.scenario() as scenario:
            model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
            response: Final = candidate.request(
                "POST",
                "/v1/responses",
                {
                    "model": model,
                    "input": "weather in Paris?",
                    "tools": [
                        {
                            "type": "function",
                            "name": "lookup_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                    "tool_choice": {"type": "function", "name": "lookup_weather"},
                    "metadata": {"trace_marker": marker},
                    "cache": {"no-cache": True},
                },
            )
            assert response.status_code == 200, response.text
            _assert_tool_call_span(
                _matching_span(destination, marker, marker_key="litellm.metadata.trace_marker"), marker
            )
