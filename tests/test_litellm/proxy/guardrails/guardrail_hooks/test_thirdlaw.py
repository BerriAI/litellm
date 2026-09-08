import json
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from litellm.caching import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.thirdlaw import (
    ThirdlawGuardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.thirdlaw.thirdlaw import (
    ThirdlawGuardrailMissingConfig,
)
from litellm.types.guardrails import (
    GuardrailEventHooks,
    LitellmParams,
    SupportedGuardrailIntegrations,
)
from litellm.types.proxy.guardrails.guardrail_hooks.thirdlaw import (
    ThirdlawGuardrailConfigModel,
    ThirdlawGuardrailConfigModelOptionalParams,
)
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

_API_BASE = "https://thirdlaw.test"
_ENDPOINT = "https://thirdlaw.test/guardrails/litellm/v2"


def _decision_response(body: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("POST", _ENDPOINT),
    )


def _make_guardrail(*, decisions: list[httpx.Response] | None = None, **overrides) -> ThirdlawGuardrail:
    handler = AsyncMock(spec=AsyncHTTPHandler)
    if decisions is not None:
        handler.post.side_effect = decisions
    kwargs: dict[str, Any] = {
        "api_base": _API_BASE,
        "api_key": "thirdlaw_secret",
        "guardrail_name": "thirdlaw-guard",
        "event_hook": "pre_call",
        "default_on": True,
        "async_handler": handler,
        **overrides,
    }
    return ThirdlawGuardrail(**kwargs)


def _request_data() -> dict:
    body = {
        "model": "gpt-5.6",
        "messages": [{"role": "user", "content": "my api key is sk-user-secret"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "temperature": 0.2,
    }
    data: dict = {
        **body,
        "litellm_call_id": "call-123",
        "metadata": {
            "user_api_key_hash": "hash-1",
            "user_api_key_alias": "alias-1",
            "user_api_key_user_id": "user-1",
            "guardrails": ["thirdlaw-guard"],
        },
        "guardrails": ["thirdlaw-guard"],
        "api_key": "sk-forwarded-provider-key",
        "secret_fields": {
            "raw_headers": {
                "authorization": "Bearer sk-live-raw",
                "x-request-id": "req-9",
            }
        },
    }
    # The proxy strips the header used for LiteLLM auth (authorization here) from its
    # sanitized header copy; the raw value survives only in secret_fields.raw_headers.
    data["proxy_server_request"] = {
        "url": "http://localhost:4000/v1/chat/completions",
        "method": "POST",
        "headers": {
            "content-type": "application/json",
            "x-request-id": "req-9",
        },
        "body": {**body, "messages": [{"role": "user", "content": "snapshot message"}]},
    }
    return data


def _model_response() -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-1",
        model="gpt-5.6",
        choices=[
            Choices(
                index=0,
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content="calling tool",
                    tool_calls=[
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "sf", "token": "sk-leak"}'},
                        }
                    ],
                ),
            )
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _sent_payload(guardrail: ThirdlawGuardrail) -> dict:
    return guardrail.async_handler.post.call_args.kwargs["json"]


async def _run_pre_call(guardrail: ThirdlawGuardrail, data: dict) -> dict:
    return await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )


def test_requires_api_base(monkeypatch):
    monkeypatch.delenv("THIRDLAW_API_BASE", raising=False)
    with pytest.raises(ThirdlawGuardrailMissingConfig):
        ThirdlawGuardrail(api_key="k")


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("THIRDLAW_API_BASE", "https://env.thirdlaw.test")
    monkeypatch.setenv("THIRDLAW_API_KEY", "env_key")
    g = ThirdlawGuardrail(guardrail_name="thirdlaw", event_hook="pre_call", default_on=True)
    assert g.api_base == "https://env.thirdlaw.test/guardrails/litellm/v2"
    assert g.http_headers["Authorization"] == "Bearer env_key"


def test_endpoint_path_not_doubled():
    g = _make_guardrail(api_base=f"{_API_BASE}/guardrails/litellm/v2")
    assert g.api_base == _ENDPOINT


def test_default_supported_event_hooks():
    g = _make_guardrail()
    assert g.supported_event_hooks == [
        GuardrailEventHooks.pre_call,
        GuardrailEventHooks.post_call,
        GuardrailEventHooks.during_call,
    ]


def test_invalid_sampling_rate_rejected():
    with pytest.raises(ValueError, match="streaming_sampling_rate"):
        _make_guardrail(streaming_sampling_rate=0)


def test_native_hook_routing():
    assert "apply_guardrail" not in ThirdlawGuardrail.__dict__
    assert "async_post_call_streaming_iterator_hook" in ThirdlawGuardrail.__dict__


def test_enum_value():
    assert SupportedGuardrailIntegrations.THIRDLAW.value == "thirdlaw"


def test_config_model_ui_name():
    assert ThirdlawGuardrailConfigModel.ui_friendly_name() == "ThirdLaw"


def test_registries_expose_initializer_and_class():
    assert isinstance(guardrail_initializer_registry, dict)
    assert isinstance(guardrail_class_registry, dict)
    assert "thirdlaw" in guardrail_initializer_registry
    assert guardrail_class_registry["thirdlaw"] is ThirdlawGuardrail


def test_config_driven_initialization_creates_callback():
    lp = LitellmParams(guardrail="thirdlaw", mode="pre_call", api_base=_API_BASE, api_key="k")
    cb = initialize_guardrail(lp, {"guardrail_name": "thirdlaw-guard"})
    assert isinstance(cb, ThirdlawGuardrail)
    assert cb.api_base == _ENDPOINT
    assert cb.unreachable_fallback == "fail_closed"
    assert cb.guardrail_timeout == httpx.Timeout(timeout=60, connect=5.0)


def test_config_driven_initialization_propagates_streaming_overrides():
    lp = LitellmParams(
        guardrail="thirdlaw",
        mode="pre_call",
        api_base=_API_BASE,
        api_key="k",
        streaming_end_of_stream_only=False,
        streaming_sampling_rate=10,
        streaming_buffer_until_moderated=False,
    )
    cb = initialize_guardrail(lp, {"guardrail_name": "thirdlaw-guard"})
    assert cb.streaming_end_of_stream_only is False
    assert cb.streaming_sampling_rate == 10
    assert cb.streaming_buffer_until_moderated is False


def test_config_model_streaming_defaults():
    params = ThirdlawGuardrailConfigModelOptionalParams()
    assert params.streaming_end_of_stream_only is True
    assert params.streaming_buffer_until_moderated is True
    assert params.streaming_sampling_rate == 5


async def test_pre_call_payload_shape():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    data = _request_data()
    out = await _run_pre_call(g, data)

    assert out["messages"] == data["messages"]
    call_kwargs = g.async_handler.post.call_args.kwargs
    assert call_kwargs["url"] == _ENDPOINT
    assert call_kwargs["headers"]["Authorization"] == "Bearer thirdlaw_secret"

    payload = call_kwargs["json"]
    assert payload["event_type"] == "pre_call"
    assert payload["request_url"] == "http://localhost:4000/v1/chat/completions"
    assert payload["metadata"]["user_api_key_hash"] == "hash-1"
    assert payload["metadata"]["litellm_call_id"] == "call-123"
    assert payload["metadata"]["model"] == "gpt-5.6"
    assert payload["request_body"]["model"] == "gpt-5.6"
    assert payload["request_body"]["tools"][0]["function"]["name"] == "get_weather"
    assert payload["request_body"]["temperature"] == 0.2
    for stripped_key in ("secret_fields", "api_key", "metadata", "guardrails", "litellm_call_id"):
        assert stripped_key not in payload["request_body"]
    assert "response_body" not in payload
    assert "sk-forwarded-provider-key" not in json.dumps(payload)


async def test_pre_call_sends_live_body_not_snapshot():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    await _run_pre_call(g, _request_data())
    messages = _sent_payload(g)["request_body"]["messages"]
    assert messages == [{"role": "user", "content": "my api key is sk-user-secret"}]


async def test_all_headers_forwarded_without_credentials_by_default():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    await _run_pre_call(g, _request_data())
    headers = _sent_payload(g)["request_headers"]
    assert headers["content-type"] == "application/json"
    assert headers["x-request-id"] == "req-9"
    assert "authorization" not in headers
    assert "sk-live-raw" not in json.dumps(_sent_payload(g))


async def test_additional_headers_opts_into_raw_values():
    g = _make_guardrail(
        decisions=[_decision_response({"action": "allow"})],
        additional_headers="Authorization , x-missing",
    )
    await _run_pre_call(g, _request_data())
    headers = _sent_payload(g)["request_headers"]
    assert headers["authorization"] == "Bearer sk-live-raw"
    assert headers["x-request-id"] == "req-9"
    assert "x-missing" not in headers


async def test_pre_call_block_raises_with_status():
    g = _make_guardrail(
        decisions=[_decision_response({"action": "block", "message": "policy violation", "response_status": 422})]
    )
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await _run_pre_call(g, _request_data())
    assert exc_info.value.message == "policy violation"
    assert exc_info.value.status_code == 422
    assert exc_info.value.blocked_content is True


async def test_pre_call_modify_request_applies_content_keys_only():
    modified_tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "[sanitized]", "parameters": {"type": "object"}},
        }
    ]
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_request",
                    "request_body": {
                        "messages": [{"role": "user", "content": "my api key is [REDACTED]"}],
                        "tools": modified_tools,
                        "temperature": 0.9,
                        "model": "attacker-model",
                        "guardrails": [],
                        "metadata": {"user_api_key_hash": "forged"},
                        "stream": True,
                        "secret_fields": {"raw_headers": {}},
                    },
                }
            )
        ]
    )
    data = _request_data()
    out = await _run_pre_call(g, data)

    assert out["messages"] == [{"role": "user", "content": "my api key is [REDACTED]"}]
    assert out["tools"] == modified_tools
    assert out["temperature"] == 0.9
    assert out["model"] == "gpt-5.6"
    assert out["guardrails"] == ["thirdlaw-guard"]
    assert out["metadata"]["user_api_key_hash"] == "hash-1"
    assert "stream" not in out
    assert out["secret_fields"]["raw_headers"]["authorization"] == "Bearer sk-live-raw"
    assert data["messages"] == [{"role": "user", "content": "my api key is sk-user-secret"}]


async def test_pre_call_records_single_guardrail_trace():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    data = _request_data()
    await _run_pre_call(g, data)
    traces = data["metadata"]["standard_logging_guardrail_information"]
    assert len(traces) == 1
    assert traces[0]["guardrail_status"] == "success"
    assert traces[0]["guardrail_response"]["action"] == "allow"


async def test_pre_call_block_records_intervened_trace():
    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "no"})])
    data = _request_data()
    with pytest.raises(GuardrailRaisedException):
        await _run_pre_call(g, data)
    traces = data["metadata"]["standard_logging_guardrail_information"]
    assert len(traces) == 1
    assert traces[0]["guardrail_status"] == "guardrail_intervened"


async def test_during_call_block_raises():
    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "denied"})])
    with pytest.raises(GuardrailRaisedException):
        await g.async_moderation_hook(data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), call_type="completion")


async def test_during_call_modify_is_ignored():
    g = _make_guardrail(
        decisions=[_decision_response({"action": "modify_request", "request_body": {"messages": [{"role": "user"}]}})]
    )
    data = _request_data()
    out = await g.async_moderation_hook(data=data, user_api_key_dict=UserAPIKeyAuth(), call_type="completion")
    assert out is data
    assert data["messages"] == [{"role": "user", "content": "my api key is sk-user-secret"}]
    assert _sent_payload(g)["event_type"] == "during_call"


async def test_post_call_payload_includes_response_and_prefers_snapshot_body():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    response = _model_response()
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=response
    )
    assert out is response
    payload = _sent_payload(g)
    assert payload["event_type"] == "post_call"
    assert payload["response_body"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert payload["request_body"]["messages"] == [{"role": "user", "content": "snapshot message"}]


async def test_post_call_modify_response_rewrites_content_and_tool_calls():
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_response",
                    "response_body": {
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "tool_calls",
                                "message": {
                                    "role": "assistant",
                                    "content": "calling tool [sanitized]",
                                    "tool_calls": [
                                        {
                                            "id": "tool-1",
                                            "type": "function",
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city": "sf", "token": "[REDACTED]"}',
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    },
                }
            )
        ]
    )
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=_model_response()
    )
    assert isinstance(out, ModelResponse)
    assert out.choices[0].message.content == "calling tool [sanitized]"
    assert out.choices[0].message.tool_calls[0].function.arguments == '{"city": "sf", "token": "[REDACTED]"}'
    assert out.id == "chatcmpl-1"
    assert out.usage.total_tokens == 15


async def test_post_call_modify_response_merges_dict_responses():
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_response",
                    "response_body": {"content": [{"type": "text", "text": "[MASKED]"}]},
                }
            )
        ]
    )
    anthropic_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "the secret is sk-leak"}],
    }
    out = await g.async_post_call_success_hook(
        data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=anthropic_response
    )
    assert out == {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "[MASKED]"}],
    }


async def test_post_call_block_raises():
    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "leaked secret"})])
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.async_post_call_success_hook(
            data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=_model_response()
        )
    assert exc_info.value.message == "leaked secret"


async def test_post_call_malformed_modified_response_fails_closed():
    g = _make_guardrail(
        decisions=[_decision_response({"action": "modify_response", "response_body": {"choices": "garbage"}})]
    )
    with pytest.raises(GuardrailRaisedException):
        await g.async_post_call_success_hook(
            data=_request_data(), user_api_key_dict=UserAPIKeyAuth(), response=_model_response()
        )


def _connect_error() -> httpx.ConnectError:
    return httpx.ConnectError("connection refused", request=httpx.Request("POST", _ENDPOINT))


async def test_unreachable_fail_closed_raises():
    g = _make_guardrail()
    g.async_handler.post.side_effect = _connect_error()
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await _run_pre_call(g, _request_data())
    assert exc_info.value.blocked_content is False


async def test_unreachable_fail_open_passes_through():
    g = _make_guardrail(unreachable_fallback="fail_open")
    g.async_handler.post.side_effect = _connect_error()
    data = _request_data()
    out = await _run_pre_call(g, data)
    assert out is data
    traces = data["metadata"]["standard_logging_guardrail_information"]
    assert traces[0]["guardrail_status"] == "guardrail_failed_to_respond"


async def test_http_500_fails_closed_even_with_fail_open():
    g = _make_guardrail(
        unreachable_fallback="fail_open",
        decisions=[_decision_response({"detail": "boom"}, status_code=500)],
    )
    with pytest.raises(GuardrailRaisedException):
        await _run_pre_call(g, _request_data())


async def test_http_503_respects_fail_open():
    g = _make_guardrail(
        unreachable_fallback="fail_open",
        decisions=[_decision_response({"detail": "overloaded"}, status_code=503)],
    )
    data = _request_data()
    assert await _run_pre_call(g, data) is data


def _stream_chunks() -> list[ModelResponseStream]:
    return [
        ModelResponseStream(
            id="chunk-1",
            model="gpt-5.6",
            choices=[StreamingChoices(index=0, delta=Delta(role="assistant", content="the secret "))],
        ),
        ModelResponseStream(
            id="chunk-1",
            model="gpt-5.6",
            choices=[StreamingChoices(index=0, delta=Delta(content="is sk-leak"))],
        ),
        ModelResponseStream(
            id="chunk-1",
            model="gpt-5.6",
            choices=[StreamingChoices(index=0, delta=Delta(content=None), finish_reason="stop")],
        ),
    ]


async def _aiter(items: list) -> Any:
    for item in items:
        yield item


async def _collect(agen: Any) -> list:
    return [item async for item in agen]


async def test_streaming_buffered_allow_replays_original_chunks():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    chunks = _stream_chunks()
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(chunks), request_data=_request_data()
        )
    )
    assert out == chunks
    payload = _sent_payload(g)
    assert payload["event_type"] == "post_call"
    assert payload["response_body"]["choices"][0]["message"]["content"] == "the secret is sk-leak"


async def test_streaming_buffered_modify_emits_rewritten_response():
    g = _make_guardrail(
        decisions=[
            _decision_response(
                {
                    "action": "modify_response",
                    "response_body": {
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "the secret is [REDACTED]"},
                            }
                        ]
                    },
                }
            )
        ]
    )
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(_stream_chunks()), request_data=_request_data()
        )
    )
    emitted_text = "".join(
        choice.delta.content or ""
        for chunk in out
        if isinstance(chunk, ModelResponseStream)
        for choice in chunk.choices
    )
    assert emitted_text == "the secret is [REDACTED]"
    assert "sk-leak" not in emitted_text


async def test_streaming_buffered_block_raises_streaming_callback_error():
    from litellm.proxy.proxy_server import StreamingCallbackError

    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "leaked secret"})])
    with pytest.raises(StreamingCallbackError, match="leaked secret"):
        await _collect(
            g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(), response=_aiter(_stream_chunks()), request_data=_request_data()
            )
        )


async def test_streaming_buffered_holds_chunks_until_decision():
    call_order: list[str] = []

    async def _recording_stream() -> Any:
        for chunk in _stream_chunks():
            call_order.append("chunk_consumed")
            yield chunk

    g = _make_guardrail()

    async def _post(*args, **kwargs):
        call_order.append("guardrail_called")
        return _decision_response({"action": "allow"})

    g.async_handler.post.side_effect = _post
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_recording_stream(), request_data=_request_data()
        )
    )
    assert len(out) == 3
    assert call_order == ["chunk_consumed", "chunk_consumed", "chunk_consumed", "guardrail_called"]


async def test_streaming_sampled_interim_block_terminates_stream():
    from litellm.proxy.proxy_server import StreamingCallbackError

    g = _make_guardrail(
        streaming_buffer_until_moderated=False,
        streaming_end_of_stream_only=False,
        streaming_sampling_rate=1,
        decisions=[_decision_response({"action": "block", "message": "bad interim"})],
    )
    agen = g.async_post_call_streaming_iterator_hook(
        user_api_key_dict=UserAPIKeyAuth(), response=_aiter(_stream_chunks()), request_data=_request_data()
    )
    first = await agen.__anext__()
    assert isinstance(first, ModelResponseStream)
    with pytest.raises(StreamingCallbackError, match="bad interim"):
        await agen.__anext__()


def _anthropic_sse_frames() -> list[bytes]:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_abc",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 9, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello there"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return [f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in events]


async def test_streaming_raw_sse_allow_replays_frames():
    g = _make_guardrail(decisions=[_decision_response({"action": "allow"})])
    frames = _anthropic_sse_frames()
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(), response=_aiter(frames), request_data=_request_data()
        )
    )
    assert out == frames
    assert _sent_payload(g)["response_body"]["choices"][0]["message"]["content"] == "hello there"


async def test_streaming_raw_sse_block_emits_anthropic_error_frame():
    g = _make_guardrail(decisions=[_decision_response({"action": "block", "message": "leaked secret"})])
    out = await _collect(
        g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_aiter(_anthropic_sse_frames()),
            request_data=_request_data(),
        )
    )
    assert len(out) == 1
    assert isinstance(out[0], bytes)
    assert b"event: error" in out[0]
    assert b"leaked secret" in out[0]
