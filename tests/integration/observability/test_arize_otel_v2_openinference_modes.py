from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final

import httpx
import pytest
from _openinference_support import (
    CHAT_TOOLS,
    _assert_chat_request,
    _chat_caller_response,
    _chat_plain_response,
    _chat_request_marker,
    _chat_response,
    _json_messages,
    _json_object,
    _matching_marker_span,
    _matching_output_value_span,
    _matching_span,
    _response_tool_calls,
    _rig,
)
from integration._support.client import Gateway
from integration._support.wire import Reply, Request

_DEFAULT_METADATA: Final = {
    "requester_ip_address": "127.0.0.1",
    "user_api_key_user_id": "default_user_id",
}
_DEFAULT_METADATA_BAGGAGE: Final = frozenset({("litellm.metadata.user_api_key_user_id", "default_user_id")})


def _request(
    proxy: Gateway,
    model: str,
    marker: str,
    *,
    prompt: str = "weather in Paris?",
    headers: dict[str, str] | None = None,
    key: str | None = None,
) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": CHAT_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
            "metadata": {"trace_marker": marker},
            "cache": {"no-cache": True},
        },
        key=key,
        headers=headers,
    )


def _assert_success_body(response: httpx.Response, marker: str, model: str) -> None:
    assert response.status_code == 200, response.text
    assert _json_object(response.content) == _chat_caller_response(_chat_response(marker), model), response.text


def _upstream(marker: str) -> Callable[[Request], Reply]:
    def reply(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return _chat_response(marker)

    return reply


def _assert_output_tool_call(attributes: dict[str, str], marker: str) -> None:
    assert attributes["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == f"call_{marker}", attributes
    assert attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "lookup_weather", (
        attributes
    )
    assert (
        attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"] == '{"city": "Paris"}'
    ), attributes
    assert _json_messages(attributes["output.value"]) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{marker}",
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        }
    ], attributes


def _assert_default_allowlist_attributes(attributes: dict[str, str], marker: str) -> None:
    observed_baggage: Final = frozenset(
        (key, value) for key, value in attributes.items() if key.startswith("litellm.metadata.")
    )
    assert observed_baggage == _DEFAULT_METADATA_BAGGAGE, attributes
    metadata: Final = _json_object(attributes["metadata"].encode())
    assert metadata == _DEFAULT_METADATA, attributes
    assert "trace_marker" not in metadata, attributes
    assert "litellm.metadata.trace_marker" not in attributes, attributes
    _assert_output_tool_call(attributes, marker)


def test_arize_otel_v2_c1_absent_allowlist(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "c1-" + uuid.uuid4().hex
    with _rig(
        gateway,
        tmp_path,
        _upstream(marker),
        remove_environment=("LITELLM_OTEL_BAGGAGE_METADATA_KEYS",),
        disabled_environment=("LITELLM_OTEL_BAGGAGE_METADATA_KEYS",),
    ) as rig:
        response: Final = _request(rig.proxy, rig.model, marker)
        _assert_success_body(response, marker, rig.model)
        attributes: Final = _matching_span(rig.destination, marker)
        _assert_default_allowlist_attributes(attributes, marker)


def test_arize_otel_v2_c2_empty_allowlist(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "c2-" + uuid.uuid4().hex
    with _rig(gateway, tmp_path, _upstream(marker), environment={"LITELLM_OTEL_BAGGAGE_METADATA_KEYS": ""}) as rig:
        response: Final = _request(rig.proxy, rig.model, marker)
        _assert_success_body(response, marker, rig.model)
        attributes: Final = _matching_span(rig.destination, marker)
        assert "metadata" not in attributes, attributes
        assert not any(key.startswith("litellm.metadata.") for key in attributes), attributes
        _assert_output_tool_call(attributes, marker)


def test_arize_otel_v2_c3_absent_allowlisted_key(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "c3-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return _chat_response(marker)

    with _rig(gateway, tmp_path, upstream) as rig:
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": rig.model,
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "tools": CHAT_TOOLS,
                "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
                "metadata": {"other": "value"},
                "cache": {"no-cache": True},
            },
        )
        _assert_success_body(response, marker, rig.model)
        attributes: Final = _matching_span(rig.destination, marker)
        assert "metadata" not in attributes, attributes
        assert "litellm.metadata.trace_marker" not in attributes, attributes
        _assert_output_tool_call(attributes, marker)


def test_arize_otel_v2_c4_promotes_marker_and_alias(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "c4-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return _chat_response(marker)

    with (
        _rig(
            gateway,
            tmp_path,
            upstream,
            environment={"LITELLM_OTEL_BAGGAGE_METADATA_KEYS": "requester_metadata.trace_marker,user_api_key_alias"},
        ) as rig,
        rig.proxy.scenario() as scenario,
    ):
        key: Final = scenario.key(key_alias="alias-c4")
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": rig.model,
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "tools": CHAT_TOOLS,
                "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
                "metadata": {"trace_marker": marker},
                "cache": {"no-cache": True},
            },
            key=key,
        )
        _assert_success_body(response, marker, rig.model)
        attributes: Final = _matching_marker_span(rig.destination, marker)
        assert _json_object(attributes["metadata"].encode()) == {
            "trace_marker": marker,
            "user_api_key_alias": "alias-c4",
        }, attributes
        assert attributes["litellm.metadata.trace_marker"] == marker, attributes
        assert attributes["litellm.metadata.user_api_key_alias"] == "alias-c4", attributes


def test_arize_otel_v2_c5_yaml_allowlist_does_not_reach_preset_so_default_applies(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "c5-" + uuid.uuid4().hex
    with _rig(
        gateway,
        tmp_path,
        _upstream(marker),
        callback_settings={"otel": {"baggage_metadata_keys": ["requester_metadata.trace_marker"]}},
        remove_environment=("LITELLM_OTEL_BAGGAGE_METADATA_KEYS",),
        disabled_environment=("LITELLM_OTEL_BAGGAGE_METADATA_KEYS",),
    ) as rig:
        response: Final = _request(rig.proxy, rig.model, marker)
        _assert_success_body(response, marker, rig.model)
        attributes: Final = _matching_marker_span(rig.destination, marker)
        _assert_default_allowlist_attributes(attributes, marker)


def test_arize_otel_v2_c6_content_capture_disabled(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "c6-" + uuid.uuid4().hex
    with _rig(
        gateway,
        tmp_path,
        _upstream(marker),
        environment={
            "LITELLM_OTEL_BAGGAGE_METADATA_KEYS": "requester_metadata.trace_marker",
        },
        remove_environment=("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",),
        disabled_environment=("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",),
    ) as rig:
        response: Final = _request(rig.proxy, rig.model, marker)
        _assert_success_body(response, marker, rig.model)
        attributes: Final = _matching_marker_span(rig.destination, marker)
        assert _json_object(attributes["metadata"].encode()) == {"trace_marker": marker}, attributes
        assert attributes["litellm.metadata.trace_marker"] == marker, attributes
        assert "gen_ai.input.messages" not in attributes, attributes
        assert "gen_ai.output.messages" not in attributes, attributes
        assert not any(
            key.startswith("llm.input_messages.") or key.startswith("llm.output_messages.") for key in attributes
        ), attributes
        assert "input.value" not in attributes, attributes
        assert "output.value" not in attributes, attributes
        assert not any(".tool_calls." in key for key in attributes), attributes


def test_arize_otel_v2_c7_key_and_team_logging_callbacks(gateway: Gateway, tmp_path: Path) -> None:
    key_marker: Final = "c7-key-" + uuid.uuid4().hex
    team_marker: Final = "c7-team-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        marker: Final = _chat_request_marker(request)
        assert marker in (key_marker, team_marker), request
        _assert_chat_request(request, messages=[{"role": "user", "content": marker}])
        return _chat_response(marker)

    logging_metadata: Final = {"logging": [{"callback_name": "arize", "callback_type": "success"}]}
    with _rig(gateway, tmp_path, upstream) as rig, rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias="key-c7", metadata=logging_metadata)
        team: Final = scenario.team(metadata=logging_metadata)
        team_key: Final = scenario.key(team_id=team)
        key_response: Final = _request(rig.proxy, rig.model, key_marker, prompt=key_marker, key=key)
        _assert_success_body(key_response, key_marker, rig.model)
        key_attributes: Final = _matching_marker_span(rig.destination, key_marker)
        team_response: Final = _request(rig.proxy, rig.model, team_marker, prompt=team_marker, key=team_key)
        _assert_success_body(team_response, team_marker, rig.model)
        team_attributes: Final = _matching_marker_span(rig.destination, team_marker)
        assert _json_object(key_attributes["metadata"].encode()) == {"trace_marker": key_marker}, key_attributes
        assert _json_object(team_attributes["metadata"].encode()) == {"trace_marker": team_marker}, team_attributes
        assert key_attributes["litellm.metadata.trace_marker"] == key_marker, key_attributes
        assert team_attributes["litellm.metadata.trace_marker"] == team_marker, team_attributes
        _assert_output_tool_call(key_attributes, key_marker)
        _assert_output_tool_call(team_attributes, team_marker)


def test_arize_otel_v2_c8_request_callback_disable(gateway: Gateway, tmp_path: Path) -> None:
    control_marker: Final = "c8-control-" + uuid.uuid4().hex
    disabled_marker: Final = "c8-disabled-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        marker: Final = _chat_request_marker(request)
        assert marker in (control_marker, disabled_marker), marker
        return _chat_response(marker)

    with _rig(
        gateway,
        tmp_path,
        upstream,
        litellm_settings={"allow_dynamic_callback_disabling": True},
    ) as rig:
        control_response: Final = _request(rig.proxy, rig.model, control_marker, prompt=control_marker)
        _assert_success_body(control_response, control_marker, rig.model)
        control_attributes: Final = _matching_marker_span(rig.destination, control_marker)
        _assert_output_tool_call(control_attributes, control_marker)

        disabled_response: Final = _request(
            rig.proxy,
            rig.model,
            disabled_marker,
            prompt=disabled_marker,
            headers={"x-litellm-disable-callbacks": "arize"},
        )
        _assert_success_body(disabled_response, disabled_marker, rig.model)


@pytest.mark.parametrize("failure_status", (401, 500))
def test_arize_otel_v2_c9_upstream_failures_are_recorded(failure_status: int, gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "c9-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        assert request.method == "POST", request.method
        assert request.body, f"{request.method} {request.target}"
        body: Final = _json_object(request.body)
        assert body == {
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "model": "gpt-4o-mini",
        }, body
        return Reply(
            status=failure_status,
            body=b'{"error":{"message":"upstream failure"}}',
            content_type="application/json",
        )

    with _rig(gateway, tmp_path, upstream) as rig:
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": rig.model,
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "metadata": {"trace_marker": marker, "failure_status": failure_status},
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == failure_status, response.text
        error_body: Final = _json_object(response.content)
        error_name: Final = {401: "AuthenticationError", 500: "InternalServerError"}[failure_status]
        error_type: Final = {401: "authentication_error", 500: "internal_server_error"}[failure_status]
        provider_message: Final = f"litellm.{error_name}: {error_name}: OpenAIException - upstream failure"
        caller_message: Final = (
            f"{provider_message}\n\nLiteLLM: model group '{rig.model}' failed with the error above. "
            "No fallback was attempted."
        )
        assert error_body == {
            "error": {
                "message": caller_message,
                "type": error_type,
                "param": None,
                "code": str(failure_status),
            }
        }, error_body
        attributes: Final = _matching_marker_span(rig.destination, marker)
        metadata: Final = attributes.get("metadata")
        assert metadata is not None, attributes
        assert _json_object(metadata.encode()) == {"trace_marker": marker}, attributes
        assert attributes["litellm.metadata.trace_marker"] == marker, attributes
        assert attributes["error.message"] == provider_message, attributes
        assert attributes["error.type"] == error_name, attributes
        assert not any(".tool_calls." in key for key in attributes), attributes


def test_arize_otel_v2_c10_attribute_limit_keeps_tool_prefix(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "c10-" + uuid.uuid4().hex
    calls: Final = _response_tool_calls(marker, ("Paris", "Berlin", "Rome", "Tokyo", "Oslo", "Lima", "Accra", "Delhi"))

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return _chat_response(marker, calls)

    with _rig(gateway, tmp_path, upstream, environment={"OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT": "57"}) as rig:
        response: Final = _request(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(_chat_response(marker, calls), rig.model), (
            response.text
        )
        attributes: Final = _matching_output_value_span(rig.destination, marker)
        indexes: Final = tuple(
            int(key.split(".tool_calls.")[1].split(".")[0])
            for key in attributes
            if ".tool_calls." in key and key.endswith(".tool_call.id")
        )
        assert indexes == (0,), attributes
        assert attributes["llm.output_messages.0.message.role"] == "assistant", attributes
        fields: Final = ("id", "function.name", "function.arguments")
        expected_tool_call_keys: Final = frozenset().union(
            *(
                frozenset(f"llm.output_messages.0.message.tool_calls.{index}.tool_call.{field}" for field in fields)
                for index in indexes
            )
        )
        observed_tool_call_keys: Final = frozenset(key for key in attributes if ".tool_calls." in key)
        assert observed_tool_call_keys == expected_tool_call_keys, attributes
        assert _json_messages(attributes["output.value"])[0]["tool_calls"] == calls, attributes
        assert _json_object(attributes["metadata"].encode()) == {"trace_marker": marker}, attributes
        assert attributes["litellm.metadata.trace_marker"] == marker, attributes

    history: Final = [{"role": "user", "content": f"history-{index}"} for index in range(40)]
    history_marker: Final = marker + "-history"

    def history_upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=history, include_tools=False)
        return _chat_plain_response(history_marker, "history retained")

    with _rig(gateway, tmp_path, history_upstream) as history_rig:
        history_response: Final = history_rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": history_rig.model,
                "messages": history,
                "metadata": {"trace_marker": history_marker},
                "cache": {"no-cache": True},
            },
        )
        assert history_response.status_code == 200, history_response.text
        assert _json_object(history_response.content) == _chat_caller_response(
            _chat_plain_response(history_marker, "history retained"), history_rig.model
        ), history_response.text
        history_attributes: Final = _matching_marker_span(history_rig.destination, history_marker)
        assert _json_object(history_attributes["metadata"].encode()) == {"trace_marker": history_marker}, (
            history_attributes
        )
        assert history_attributes["litellm.metadata.trace_marker"] == history_marker, history_attributes
        assert tuple(history_attributes[f"llm.input_messages.{index}.message.role"] for index in range(40)) == tuple(
            str(message["role"]) for message in history
        )
        assert tuple(history_attributes[f"llm.input_messages.{index}.message.content"] for index in range(40)) == tuple(
            str(message["content"]) for message in history
        )
        assert _json_messages(history_attributes["output.value"]) == [
            {"role": "assistant", "content": "history retained"}
        ], history_attributes
