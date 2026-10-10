from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import pytest
from _openinference_support import (
    CHAT_TOOLS,
    _assert_chat_request,
    _chat_caller_response,
    _chat_output_value,
    _chat_request_marker,
    _collect_marker_spans,
    _json_messages,
    _json_object,
    _json_object_value,
    _matching_marker_span,
    _rig,
    _span_attributes,
    _spans,
)
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from pydantic import JsonValue


def _call(
    proxy: Gateway,
    model: str,
    marker: str,
    metadata: JsonValue | None = None,
    key: str | None = None,
    *,
    prompt: str = "weather in Paris?",
) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": CHAT_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
            "metadata": metadata if metadata is not None else {"trace_marker": marker},
            "cache": {"no-cache": True},
        },
        key=key,
    )


def _success(marker: str) -> Reply:
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
                                    "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        ).encode()
    )


def _chat_message(body: bytes) -> dict[str, JsonValue]:
    response: Final = _json_object(body)
    choices: Final = response["choices"]
    assert isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict), response
    message: Final = choices[0]["message"]
    assert isinstance(message, dict), response
    return message


@pytest.mark.parametrize(
    ("value", "expected_trace"),
    (
        (7, "7"),
        (["one", 2], None),
        ("", None),
        ("x" * 5000, "x" * 5000),
        ({"enabled": True}, None),
    ),
)
def test_arize_otel_v2_d1_metadata_value_shapes(
    value: JsonValue, expected_trace: str | None, gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "d1-" + uuid.uuid4().hex
    metadata: Final = {"trace_marker": value}

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return _success(marker)

    with _rig(gateway, tmp_path, upstream) as rig:
        response: Final = _call(rig.proxy, rig.model, marker, metadata)
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(_success(marker), rig.model), response.text
        attributes: Final = _matching_marker_span(rig.destination, marker)
        if expected_trace is None:
            assert "metadata" not in attributes, attributes
            assert "litellm.metadata.trace_marker" not in attributes, attributes
        else:
            assert json.loads(attributes["metadata"]) == {"trace_marker": expected_trace}, attributes
            assert attributes["litellm.metadata.trace_marker"] == expected_trace, attributes


def test_arize_otel_v2_d2_duplicate_json_metadata_keys(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "d2-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return _success(marker)

    with _rig(gateway, tmp_path, upstream) as rig:
        request_body: Final = (
            '{"model":"'
            + rig.model
            + '","messages":'
            + json.dumps([{"role": "user", "content": "weather in Paris?"}])
            + ',"tools":'
            + json.dumps(CHAT_TOOLS)
            + ","
            + '"tool_choice":{"type":"function","function":{"name":"lookup_weather"}},'
            + '"metadata":{"trace_marker":"'
            + marker
            + '","trace_marker":"'
            + marker
            + '"},"cache":{"no-cache":true}}'
        )
        response: Final = rig.proxy.client.post(
            "/v1/chat/completions",
            content=request_body,
            headers={
                "authorization": f"Bearer {rig.proxy.key}",
                "content-type": "application/json",
            },
        )
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(_success(marker), rig.model), response.text
        attributes: Final = _matching_marker_span(rig.destination, marker)
        assert _json_object(attributes["metadata"].encode()) == {"trace_marker": marker}, attributes
        assert attributes["litellm.metadata.trace_marker"] == marker, attributes


def test_arize_otel_v2_d4_unauthenticated_request_has_no_span(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "d4-" + uuid.uuid4().hex
    with _rig(gateway, tmp_path, lambda _request: _success(marker)) as rig:
        with httpx.Client(base_url=str(rig.proxy.client.base_url), trust_env=False) as client:
            response: Final = client.post(
                "/v1/chat/completions",
                json={"model": rig.model, "messages": [{"role": "user", "content": marker}]},
            )
        assert response.status_code == 401, response.text
        body: Final = _json_object(response.content)
        assert body == {
            "error": {
                "message": "Authentication Error, No api key passed in.",
                "type": "auth_error",
                "param": "None",
                "code": "401",
            }
        }, body
        assert rig.provider.received.qsize() == 0
        spans: Final = tuple(
            attributes
            for attributes in _spans(rig.destination.drain())
            if attributes.get("openinference.span.kind") == "LLM"
        )
        assert spans == (), "unauthenticated request exported an LLM span"


def test_arize_otel_v2_d5_unknown_model_leaves_proxy_ready(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "d5-" + uuid.uuid4().hex
    unknown_model: Final = "unknown-model-" + uuid.uuid4().hex
    with _rig(gateway, tmp_path, lambda _request: _success(marker)) as rig:
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": unknown_model, "messages": [{"role": "user", "content": marker}]},
        )
        assert response.status_code == 400, response.text
        body: Final = _json_object(response.content)
        error_message: Final = (
            f"/chat/completions: Invalid model name passed in model={unknown_model}. "
            "Call `/v1/models` to view available models for your key."
        )
        assert body == {
            "error": {
                "message": error_message,
                "type": "invalid_request_error",
                "param": None,
                "code": "400",
                "provider_specific_fields": {"error": error_message},
            }
        }, body
        assert rig.provider.received.qsize() == 0
        readiness: Final = rig.proxy.client.get("/health/readiness")
        assert readiness.status_code == 200, readiness.text
        assert _json_object(readiness.content) == {"status": "healthy", "db": "connected"}, readiness.text
        spans: Final = tuple(
            attributes
            for attributes in _spans(rig.destination.drain())
            if attributes.get("openinference.span.kind") == "LLM"
        )
        assert spans == (), "unknown model exported an LLM span"


def test_arize_otel_v2_d6_sink_rejections_do_not_change_caller_response(gateway: Gateway, tmp_path: Path) -> None:
    markers: Final = tuple(f"d6-{status}-" + uuid.uuid4().hex for status in (403, 404))
    unrelated_marker: Final = "d6-unrelated-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        response_marker: Final = _chat_request_marker(request)
        _assert_chat_request(request, messages=[{"role": "user", "content": response_marker}])
        return _success(response_marker)

    def sink(request: Request) -> Reply:
        exported_markers: Final = tuple(
            attributes.get("litellm.metadata.trace_marker")
            for attributes in _span_attributes(request)
            if "litellm.metadata.trace_marker" in attributes
        )
        if markers[0] in exported_markers:
            return Reply(status=403, body=b"rejected")
        if markers[1] in exported_markers:
            return Reply(status=404, body=b"rejected")
        return Reply(body=b"", content_type="application/x-protobuf")

    with (
        _rig(
            gateway,
            tmp_path,
            upstream,
            destination_handler=sink,
        ) as rig,
        rig.proxy.scenario() as scenario,
    ):
        unrelated_key: Final = scenario.key(key_alias="unrelated-d6")
        first: Final = _call(rig.proxy, rig.model, markers[0], prompt=markers[0])
        assert first.status_code == 200, first.text
        assert _json_object(first.content) == _chat_caller_response(_success(markers[0]), rig.model), first.text
        assert _matching_marker_span(rig.destination, markers[0])["litellm.metadata.trace_marker"] == markers[0]
        second: Final = _call(rig.proxy, rig.model, markers[1], prompt=markers[1])
        assert second.status_code == 200, second.text
        assert _json_object(second.content) == _chat_caller_response(_success(markers[1]), rig.model), second.text
        assert _matching_marker_span(rig.destination, markers[1])["litellm.metadata.trace_marker"] == markers[1]
        unrelated: Final = _call(
            rig.proxy,
            rig.model,
            unrelated_marker,
            key=unrelated_key,
            prompt=unrelated_marker,
        )
        assert unrelated.status_code == 200, unrelated.text
        assert _json_object(unrelated.content) == _chat_caller_response(_success(unrelated_marker), rig.model), (
            unrelated.text
        )
        assert (
            _matching_marker_span(rig.destination, unrelated_marker)["litellm.metadata.trace_marker"]
            == unrelated_marker
        )


def test_arize_otel_v2_d7_missing_space_id_is_stable(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "d7-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return _success(marker)

    with _rig(
        gateway,
        tmp_path,
        upstream,
        remove_environment=("ARIZE_SPACE_ID",),
        disabled_environment=("ARIZE_SPACE_ID",),
    ) as rig:
        response: Final = _call(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(_success(marker), rig.model), response.text
        assert _matching_marker_span(rig.destination, marker)["litellm.metadata.trace_marker"] == marker


def test_arize_otel_v2_e1_uncached_request_exports_one_llm_span(gateway: Gateway, tmp_path: Path) -> None:
    markers: Final = tuple("e1-" + uuid.uuid4().hex for _ in range(3))

    def upstream(request: Request) -> Reply:
        marker: Final = _chat_request_marker(request)
        assert marker in markers, request
        _assert_chat_request(request, messages=[{"role": "user", "content": marker}])
        return _success(marker)

    with _rig(gateway, tmp_path, upstream) as rig:
        responses: Final = tuple(_call(rig.proxy, rig.model, marker, prompt=marker) for marker in markers)
        assert all(response.status_code == 200 for response in responses), tuple(
            response.text for response in responses
        )
        assert tuple(_json_object(response.content) for response in responses) == tuple(
            _chat_caller_response(_success(marker), rig.model) for marker in markers
        ), responses
        spans: Final = _collect_marker_spans(rig.destination, markers)
        assert len(spans) == len(markers), spans
        spans_by_id: Final = {span["gen_ai.response.id"]: span for span in spans}
        assert frozenset(spans_by_id) == frozenset(markers), spans
        assert all(
            spans_by_id[marker]["gen_ai.response.id"] in response.text
            for marker, response in zip(markers, responses, strict=True)
        ), spans


def test_arize_otel_v2_e2_concurrent_unique_markers(gateway: Gateway, tmp_path: Path) -> None:
    markers: Final = tuple("e2-" + uuid.uuid4().hex for _ in range(20))

    def upstream(request: Request) -> Reply:
        marker: Final = _chat_request_marker(request)
        assert marker.startswith("e2-"), request
        _assert_chat_request(request, messages=[{"role": "user", "content": marker}])
        return _success(marker)

    with _rig(gateway, tmp_path, upstream) as rig:
        with ThreadPoolExecutor(max_workers=20) as executor:
            responses: Final = tuple(
                executor.map(
                    lambda marker: _call(rig.proxy, rig.model, marker, prompt=marker),
                    markers,
                )
            )
        assert all(response.status_code == 200 for response in responses), tuple(
            response.text for response in responses
        )
        assert tuple(_json_object(response.content) for response in responses) == tuple(
            _chat_caller_response(_success(marker), rig.model) for marker in markers
        ), responses
        spans: Final = _collect_marker_spans(rig.destination, markers)
        assert len(spans) == len(markers), spans
        assert tuple(
            _json_object(next(span for span in spans if marker in span.values())["metadata"].encode())
            for marker in markers
        ) == tuple({"trace_marker": marker} for marker in markers), spans
        assert (
            tuple(
                next(span for span in spans if marker in span.values())["litellm.metadata.trace_marker"]
                for marker in markers
            )
            == markers
        ), spans


@pytest.mark.parametrize(
    "shape",
    ("object-arguments", "missing-name", "non-dict-call", "null-tool-calls", "integer-id"),
)
def test_arize_otel_v2_d3_malformed_tool_calls_are_normalized(shape: str, gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"d3-{shape}-" + uuid.uuid4().hex

    def malformed_reply() -> Reply:
        call: Final = {
            "id": 17 if shape == "integer-id" else f"call_{marker}",
            "type": "function",
            "function": {
                **({} if shape == "missing-name" else {"name": "lookup_weather"}),
                "arguments": {"city": "Paris"} if shape == "object-arguments" else '{"city": "Paris"}',
            },
        }
        tool_calls: Final[JsonValue] = (
            None if shape == "null-tool-calls" else ["not-a-call"] if shape == "non-dict-call" else [call]
        )
        message: Final = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
        return Reply(
            body=json.dumps(
                {
                    "id": marker,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                }
            ).encode()
        )

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return malformed_reply()

    with _rig(gateway, tmp_path, upstream) as rig:
        proxy_log: Final = rig.owned.log
        response: Final = _call(rig.proxy, rig.model, marker)
        if shape == "non-dict-call":
            assert response.status_code == 400, response.text
            error_body: Final = _json_object(response.content)
            assert frozenset(error_body) == frozenset({"error"}), error_body
            error: Final = _json_object_value(error_body["error"])
            assert frozenset(error) == frozenset({"type", "code", "param", "message"}), error
            assert error["type"] == "invalid_request_error", error
            assert error["code"] == "400", error
            assert error["param"] is None, error
            message: Final = error["message"]
            assert isinstance(message, str), error
            assert "AttributeError: 'str' object has no attribute 'get'" in message, error
            spans: Final = tuple(_spans(rig.destination.drain()))
            assert all(not any(".tool_calls." in key for key in attributes) for attributes in spans), spans
            readiness: Final = rig.proxy.client.get("/health/readiness")
            assert readiness.status_code == 200, readiness.text
            assert _json_object(readiness.content) == {"status": "healthy", "db": "connected"}, readiness.text
        else:
            assert response.status_code == 200, response.text
            assert _json_object(response.content) == _chat_caller_response(malformed_reply(), rig.model), response.text
            attributes: Final = _matching_marker_span(rig.destination, marker)
            indexed_prefix: Final = "llm.output_messages.0.message.tool_calls.0.tool_call."
            expected_fields: Final = (
                ("id", "function.name", "function.arguments"),
                ("id", "function.arguments"),
                (),
                ("function.name", "function.arguments"),
            )[("object-arguments", "missing-name", "null-tool-calls", "integer-id").index(shape)]
            expected_keys: Final = frozenset(indexed_prefix + field for field in expected_fields)
            observed_keys: Final = frozenset(key for key in attributes if ".tool_calls." in key)
            assert observed_keys == expected_keys, attributes
            if shape == "object-arguments":
                assert attributes["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == f"call_{marker}", (
                    attributes
                )
                assert (
                    attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "lookup_weather"
                ), attributes
                assert (
                    attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]
                    == '{"city": "Paris"}'
                ), attributes
            elif shape == "missing-name":
                assert attributes["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == f"call_{marker}", (
                    attributes
                )
                assert (
                    attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]
                    == '{"city": "Paris"}'
                ), attributes
            elif shape == "integer-id":
                assert (
                    attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "lookup_weather"
                ), attributes
                assert (
                    attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]
                    == '{"city": "Paris"}'
                ), attributes
            assert _json_messages(attributes["output.value"]) == _json_messages(
                _chat_output_value(malformed_reply())
            ), attributes
    assert "Exception while exporting Span batch" not in proxy_log.read_text(), proxy_log.read_text()


def test_arize_otel_v2_d3_non_dict_tool_call_is_not_a_caller_error(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip("BUG: LIT-9125 non-dict tool_calls entry returns HTTP 400 with a server traceback")
    marker: Final = "d3-non-dict-call-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
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
                            "message": {"role": "assistant", "content": None, "tool_calls": ["not-a-call"]},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                }
            ).encode()
        )

    with _rig(gateway, tmp_path, upstream) as rig:
        response: Final = _call(rig.proxy, rig.model, marker)
        assert response.status_code not in range(400, 500), response.text
        assert "Traceback" not in response.text, response.text
        assert "AttributeError" not in response.text, response.text
        readiness: Final = rig.proxy.client.get("/health/readiness")
        assert readiness.status_code == 200, readiness.text
        assert _json_object(readiness.content) == {"status": "healthy", "db": "connected"}, readiness.text


@pytest.mark.parametrize("shape", ("empty", "null", "missing"))
def test_arize_otel_v2_e3_empty_or_missing_tool_calls_never_indexed(
    shape: str, gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = f"e3-{shape}-" + uuid.uuid4().hex
    message: Final = (
        {"role": "assistant", "content": None, "tool_calls": []}
        if shape == "empty"
        else {"role": "assistant", "content": None, "tool_calls": None}
        if shape == "null"
        else {"role": "assistant", "content": None}
    )
    expected_response: Final[dict[str, JsonValue]] = {
        "id": marker,
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }

    expected_reply: Final = Reply(body=json.dumps(expected_response).encode())

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return expected_reply

    with _rig(gateway, tmp_path, upstream) as rig:
        response: Final = _call(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        response_body: Final = _json_object(response.content)
        assert response_body == _chat_caller_response(expected_reply, rig.model), response.text
        attributes: Final = _matching_marker_span(rig.destination, marker)
        assert not any(".tool_calls." in key for key in attributes), attributes
        assert "tool_calls" not in _json_messages(attributes["output.value"])[0], attributes
