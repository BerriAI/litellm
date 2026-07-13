from unittest.mock import AsyncMock, MagicMock

import litellm
import httpx
import pytest
from pydantic import ValidationError
from starlette.exceptions import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.straiker import initialize_guardrail
from litellm.proxy.guardrails.guardrail_hooks.straiker.straiker import (
    StraikerGuardrail,
    _has_meaningful_tool_calls,
    _last_user_prompt,
    _resolve_provider,
    _resolve_session_id,
    _resolve_user_name,
)
from litellm.proxy.guardrails.guardrail_registry import (
    guardrail_class_registry,
    guardrail_initializer_registry,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.proxy.guardrails.guardrail_hooks.straiker import (
    StraikerGuardrailConfigModel,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
)


def _mock_response(score: float, debug: dict = None) -> MagicMock:
    body = {"score": score, "turnId": "test-turn-id"}
    if debug is not None:
        body["debug"] = debug
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = body
    resp.text = ""
    return resp


def _make_guardrail(**overrides) -> StraikerGuardrail:
    defaults = {
        "api_key": "test-key",
        "api_base": "https://test.straiker.ai",
        "threshold": 0.5,
        "max_retries": 0,
        "guardrail_name": "straiker-pre",
        "event_hook": "pre_call",
        "async_handler": MagicMock(spec=httpx.AsyncClient),
        "verbose": True,
    }
    defaults.update(overrides)
    g = StraikerGuardrail(**defaults)
    g.async_handler.post = AsyncMock()
    return g


def test_straiker_in_initializer_registry():
    assert "straiker" in guardrail_initializer_registry


def test_straiker_in_class_registry():
    assert "straiker" in guardrail_class_registry
    assert guardrail_class_registry["straiker"] is StraikerGuardrail


def test_ui_friendly_name():
    assert StraikerGuardrailConfigModel.ui_friendly_name() == "Straiker"
    assert StraikerGuardrail.get_config_model() is StraikerGuardrailConfigModel


def test_invalid_unreachable_fallback_rejected_at_init():
    with pytest.raises(ValueError):
        StraikerGuardrail(api_key="test", unreachable_fallback="invalid")


@pytest.mark.asyncio
async def test_pre_call_blocks_when_score_above_threshold():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response(0.9, debug={"detections": {"block": {"prompt_injection": 1}}})
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "test prompt"}],
    }
    with pytest.raises(HTTPException) as exc:
        await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
    assert exc.value.status_code == 403
    err = exc.value.detail["error"]
    assert err["x-straiker-verdict"] == "block"
    assert err["x-straiker-score"] == 0.9
    assert err["x-straiker-triggered-categories"] == ["prompt_injection"]


@pytest.mark.asyncio
async def test_pre_call_allows_when_score_below_threshold():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "benign"}],
    }
    result = await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
    assert result is data


@pytest.mark.asyncio
async def test_pre_call_fail_open_returns_data_when_straiker_unreachable():
    g = _make_guardrail(unreachable_fallback="fail_open")
    g.async_handler.post.side_effect = httpx.ConnectError("boom")
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}
    result = await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
    assert result is data


@pytest.mark.asyncio
async def test_pre_call_fail_closed_raises_503_when_straiker_unreachable():
    g = _make_guardrail(unreachable_fallback="fail_closed")
    g.async_handler.post.side_effect = httpx.ConnectError("boom")
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}
    with pytest.raises(HTTPException) as exc:
        await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["x-straiker-verdict"] == "error"


@pytest.mark.asyncio
async def test_pre_call_litellm_timeout_uses_fail_closed_fallback():
    g = _make_guardrail(unreachable_fallback="fail_closed")
    g.async_handler.post.side_effect = litellm.Timeout(
        message="Straiker timed out",
        model="straiker",
        llm_provider="straiker",
    )
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}

    with pytest.raises(HTTPException) as exc:
        await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["x-straiker-verdict"] == "error"


@pytest.mark.asyncio
async def test_pre_call_retries_transient_failure_then_succeeds():
    g = _make_guardrail(max_retries=1, initial_backoff=0, max_backoff=0)
    transient_response = MagicMock(spec=httpx.Response)
    transient_response.status_code = 503
    transient_response.text = "temporarily unavailable"
    g.async_handler.post.side_effect = [transient_response, _mock_response(0.1)]
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}

    result = await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert result is data
    assert g.async_handler.post.call_count == 2


@pytest.mark.asyncio
async def test_pre_call_missing_score_uses_fail_closed_fallback():
    g = _make_guardrail(unreachable_fallback="fail_closed")
    g.async_handler.post.return_value = _mock_response(0.1)
    g.async_handler.post.return_value.json.return_value = {"turnId": "missing-score"}
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}

    with pytest.raises(HTTPException) as exc:
        await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["message"] == "Straiker detection unavailable"
    assert "missing-score" not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_pre_call_serialization_type_error_uses_generic_fail_closed_response():
    g = _make_guardrail(unreachable_fallback="fail_closed")
    g.async_handler.post.side_effect = TypeError("raw serialization detail")
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}

    with pytest.raises(HTTPException) as exc:
        await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["message"] == "Straiker detection unavailable"
    assert "raw serialization detail" not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_pre_call_vendor_error_body_is_not_reflected():
    g = _make_guardrail(unreachable_fallback="fail_closed")
    response = MagicMock(spec=httpx.Response)
    response.status_code = 400
    response.text = "vendor secret detail"
    g.async_handler.post.return_value = response
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}

    with pytest.raises(HTTPException) as exc:
        await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert exc.value.detail["error"]["message"] == "Straiker detection unavailable"
    assert "vendor secret detail" not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_agentic_dedup_scans_pre_when_last_role_is_tool():
    g = _make_guardrail(agentic=True, dedup_iterations=True)
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "tool result"},
        ],
    }
    result = await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
    assert result is data
    g.async_handler.post.assert_called_once()


@pytest.mark.asyncio
async def test_agentic_dedup_suppresses_assistant_only_continuation():
    g = _make_guardrail(agentic=True, dedup_iterations=True)
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "continuing"},
        ],
    }

    result = await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert result is data
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_agentic_dedup_disabled_scans_assistant_continuation():
    g = _make_guardrail(agentic=True, dedup_iterations=False)
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "continuing"},
        ],
    }
    await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
    g.async_handler.post.assert_called_once()


@pytest.mark.asyncio
async def test_oversized_payload_fail_open_bypasses_detection():
    g = _make_guardrail(max_payload_bytes=100, unreachable_fallback="fail_open")
    huge_content = "x" * 1000
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": huge_content}],
    }
    result = await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
    assert result is data
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_oversized_payload_fail_closed_raises_503():
    g = _make_guardrail(max_payload_bytes=100, unreachable_fallback="fail_closed")
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "x" * 1000}],
    }

    with pytest.raises(HTTPException) as exc:
        await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert exc.value.status_code == 503
    assert "detection unavailable" in exc.value.detail["error"]["message"]
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_normalizes_responses_api_input():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {"model": "gpt-4o-mini", "input": "responses api prompt"}

    result = await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="responses")

    assert result is data
    payload = g.async_handler.post.call_args.kwargs["json"]
    assert payload["prompt"] == "responses api prompt"


@pytest.mark.asyncio
async def test_agentic_request_uses_configured_api_base_and_protected_headers():
    g = _make_guardrail(
        api_base="https://detect.example/",
        agentic=True,
        custom_headers={
            "authorization": "Bearer untrusted",
            "X-Straiker-Smart-Publish": "false",
            "X-Tenant": "tenant-1",
        },
    )
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}

    await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")

    assert g.async_handler.post.call_args.args == ("https://detect.example/api/v1/detect?agentic",)
    assert g.async_handler.post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
        "X-Straiker-Smart-Publish": "true",
        "Straiker-Debug": "TRUE",
        "X-Tenant": "tenant-1",
    }
    assert g.async_handler.post.call_args.kwargs["timeout"] == 5.0
    metadata = g.async_handler.post.call_args.kwargs["json"]["metadata"]
    assert metadata["event_type"] == "pre_call"
    assert "finish_reason" not in metadata


@pytest.mark.asyncio
async def test_pre_call_sends_app_session_id_when_provided():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "x"}],
        "metadata": {"session_id": "my-app-session-42"},
    }
    await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
    sent_payload = g.async_handler.post.call_args.kwargs["json"]
    assert sent_payload["session_id"] == "my-app-session-42"


@pytest.mark.asyncio
async def test_pre_call_falls_back_to_litellm_call_id_when_no_app_session():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "x"}],
        "litellm_call_id": "litellm-uuid-abc-123",
    }
    await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
    sent_payload = g.async_handler.post.call_args.kwargs["json"]
    assert sent_payload["session_id"] == "litellm-uuid-abc-123"


@pytest.mark.asyncio
async def test_pre_call_falls_back_to_placeholder_when_no_session_id_anywhere():
    g = _make_guardrail()
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}
    await g.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type="completion")
    sent_payload = g.async_handler.post.call_args.kwargs["json"]
    assert sent_payload["session_id"] == "litellm-session"


def test_resolve_session_id_precedence():
    assert _resolve_session_id({}, {"session_id": "from-meta"}) == "from-meta"
    assert (
        _resolve_session_id(
            {"litellm_call_id": "from-call"},
            {"requester_metadata": {"session_id": "from-requester"}},
        )
        == "from-requester"
    )
    data = {
        "litellm_metadata": {"session_id": "from-litellm-meta"},
        "litellm_call_id": "from-call",
    }
    assert _resolve_session_id(data, data["litellm_metadata"]) == "from-litellm-meta"
    assert _resolve_session_id({"litellm_call_id": "from-call"}, {}) == "from-call"
    assert _resolve_session_id({}, {}) == "litellm-session"


def test_resolve_user_name_precedence():
    assert _resolve_user_name({}, {"user_name": "explicit"}) == "explicit"
    assert _resolve_user_name({"user": "from-data"}, {}) == "from-data"
    assert _resolve_user_name({}, {}) == "litellm"


def test_last_user_prompt_returns_most_recent_user_turn():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "second"},
    ]
    assert _last_user_prompt(msgs) == "second"


def test_last_user_prompt_handles_multimodal_content():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "https://example.test"}},
                {"type": "text", "text": "second"},
            ],
        }
    ]
    assert _last_user_prompt(msgs) == "first\nsecond"


def test_last_user_prompt_handles_deeply_nested_content_without_recursion():
    content = {"type": "text", "text": "deep"}
    for _ in range(2000):
        content = [content]

    assert _last_user_prompt([{"role": "user", "content": content}]) == "deep"


def test_last_user_prompt_returns_empty_when_no_user_turn():
    assert _last_user_prompt([{"role": "assistant", "content": "x"}]) == ""
    assert _last_user_prompt([]) == ""


def test_has_meaningful_tool_calls_requires_function_name():
    assert _has_meaningful_tool_calls([{"function": {"name": "get_weather"}}]) is True
    assert _has_meaningful_tool_calls([{"name": "get_weather"}]) is True
    assert _has_meaningful_tool_calls([{"function": {}}]) is False
    assert _has_meaningful_tool_calls([]) is False
    assert _has_meaningful_tool_calls(None) is False


def test_has_meaningful_tool_calls_supports_litellm_objects():
    tool_call = ChatCompletionMessageToolCall(
        id="call-1",
        function=Function(name="get_weather", arguments='{"city":"Paris"}'),
    )

    assert _has_meaningful_tool_calls([tool_call]) is True


@pytest.mark.asyncio
async def test_during_call_blocks_when_score_above_threshold():
    g = _make_guardrail(event_hook="during_call")
    g.async_handler.post.return_value = _mock_response(0.9)
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "test prompt"}]}

    with pytest.raises(HTTPException) as exc:
        await g.async_moderation_hook(data=data, user_api_key_dict=None, call_type="completion")

    assert exc.value.status_code == 403
    assert exc.value.detail["error"]["message"] == "Straiker: threat detected (during-call)"
    assert g.async_handler.post.call_args.kwargs["json"]["annotations"]["hook"] == "moderation"
    assert g.async_handler.post.call_args.kwargs["json"]["metadata"]["event_type"] == "during_call"


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["pre", "during"])
async def test_anthropic_tool_result_is_scanned_and_tool_use_is_preserved(hook):
    g = _make_guardrail(agentic=True, dedup_iterations=True)
    g.async_handler.post.return_value = _mock_response(0.1)
    data = {
        "model": "anthropic/claude-sonnet-4-5",
        "messages": [
            {"role": "user", "content": "look this up"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "lookup",
                        "input": {"query": "sensitive"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": [{"type": "text", "text": "tool secret"}],
                    }
                ],
            },
        ],
    }

    if hook == "pre":
        result = await g.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=data,
            call_type="anthropic_messages",
        )
    else:
        result = await g.async_moderation_hook(
            data=data,
            user_api_key_dict=None,
            call_type="anthropic_messages",
        )

    assert result is data
    assert g.async_handler.post.call_args.kwargs["json"]["messages"] == [
        {"role": "user", "content": "look this up"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "tool-1",
                    "name": "lookup",
                    "input": {"query": "sensitive"},
                }
            ],
        },
        {
            "role": "tool",
            "content": "tool secret",
            "tool_call_id": "tool-1",
        },
    ]


@pytest.mark.asyncio
async def test_post_call_combines_all_choices_and_tool_calls_once():
    g = _make_guardrail(agentic=True)
    g.async_handler.post.return_value = _mock_response(0.1)
    response = ModelResponse(
        choices=[
            Choices(
                index=0,
                message=Message(
                    role="assistant",
                    content="first",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call-1",
                            function=Function(name="get_weather", arguments='{"city":"Paris"}'),
                        )
                    ],
                ),
            ),
            Choices(
                index=1,
                message=Message(
                    role="assistant",
                    content="second",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call-2",
                            function=Function(name="get_time", arguments="not-json"),
                        )
                    ],
                ),
            ),
        ]
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "question"}],
    }

    result = await g.async_post_call_success_hook(data=data, user_api_key_dict=None, response=response)

    assert result is response
    g.async_handler.post.assert_called_once()
    payload = g.async_handler.post.call_args.kwargs["json"]
    assert payload["messages"] == [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "first\nsecond",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
                {
                    "id": "call-2",
                    "name": "get_time",
                    "input": {"_raw": "not-json"},
                },
            ],
        },
    ]
    assert payload["metadata"]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_post_call_sends_provider_finish_reason_for_final_response():
    g = _make_guardrail(agentic=True)
    g.async_handler.post.return_value = _mock_response(0.1)
    response = ModelResponse(
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content="done"),
            )
        ]
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "question"}],
    }

    await g.async_post_call_success_hook(data=data, user_api_key_dict=None, response=response)

    payload = g.async_handler.post.call_args.kwargs["json"]
    assert payload["metadata"]["finish_reason"] == "stop"
    assert payload["metadata"]["event_type"] == "post_call"


@pytest.mark.asyncio
async def test_post_call_fail_closed_returns_response_for_malformed_detection():
    g = _make_guardrail(unreachable_fallback="fail_closed")
    g.async_handler.post.return_value = _mock_response(0.1)
    g.async_handler.post.return_value.json.return_value = {"turnId": "missing-score"}
    response = ModelResponse(
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="provider response"),
            )
        ]
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "question"}],
    }

    result = await g.async_post_call_success_hook(data=data, user_api_key_dict=None, response=response)

    assert result is response
    assert result.choices[0].message.content == "provider response"


@pytest.mark.asyncio
async def test_post_call_oversized_payload_remains_observability_only():
    g = _make_guardrail(max_payload_bytes=100, unreachable_fallback="fail_closed")
    response = ModelResponse(
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="provider response"),
            )
        ]
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "x" * 1000}],
    }

    result = await g.async_post_call_success_hook(data=data, user_api_key_dict=None, response=response)

    assert result is response
    assert result.choices[0].message.content == "provider response"
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_post_call_normalizes_malformed_nested_metadata_and_endpoint():
    g = _make_guardrail(agentic=True)
    g.async_handler.post.return_value = _mock_response(0.1)
    response = ModelResponse(
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="provider response"),
            )
        ]
    )
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "question"}],
        "litellm_metadata": {
            "network": "malformed",
            "requester_metadata": ["malformed"],
            "endpoint": "https://[invalid?token=secret#fragment",
        },
        "proxy_server_request": "malformed",
    }

    result = await g.async_post_call_success_hook(data=data, user_api_key_dict=None, response=response)

    assert result is response
    payload = g.async_handler.post.call_args.kwargs["json"]
    assert payload["network"]["IP"] == "127.0.0.1"
    assert payload["annotations"]["endpoint"] == "https://[invalid"


@pytest.mark.asyncio
async def test_post_call_scans_real_responses_api_output_once():
    g = _make_guardrail(agentic=True)
    g.async_handler.post.return_value = _mock_response(0.1)
    response = ResponsesAPIResponse(
        id="resp-1",
        created_at=0,
        output=[
            {
                "type": "message",
                "id": "msg-1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "first"},
                    {"type": "output_text", "text": "second"},
                ],
            },
            {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": '{"query":"secret"}',
            },
        ],
    )
    data = {
        "model": "openai/gpt-4o-mini",
        "call_type": "responses",
        "input": "question",
    }

    result = await g.async_post_call_success_hook(data=data, user_api_key_dict=None, response=response)

    assert result is response
    g.async_handler.post.assert_called_once()
    assert g.async_handler.post.call_args.kwargs["json"]["messages"] == [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "first\nsecond",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "lookup",
                    "input": {"query": "secret"},
                }
            ],
        },
    ]
    assert g.async_handler.post.call_args.kwargs["json"]["metadata"]["finish_reason"] == "tool_calls"


def test_agent_id_is_used_as_source_and_app_name():
    g = _make_guardrail(agentic=True, source="litellm-gw")
    payload = g._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        app_response="N/A",
        data={
            "model": "openai/gpt-4o-mini",
            "litellm_metadata": {"agent_id": "agent-one"},
        },
        hook="pre_call",
    )
    assert payload["source"] == "agent-one"
    assert payload["metadata"]["app_name"] == "agent-one"
    assert "agent_role" not in payload["metadata"]
    assert "agent_role" not in payload["annotations"]


def test_two_agent_ids_sharing_one_model_remain_distinct():
    g = _make_guardrail(agentic=True, source="litellm-gw")
    identities = {
        (
            payload["source"],
            payload["metadata"]["app_name"],
        )
        for agent_id in ("agent-one", "agent-two")
        for payload in (
            g._build_payload(
                messages=[{"role": "user", "content": "hi"}],
                app_response="N/A",
                data={
                    "model": "openai/gpt-4o-mini",
                    "litellm_metadata": {"agent_id": agent_id},
                },
                hook="pre_call",
            ),
        )
    }
    assert identities == {
        ("agent-one", "agent-one"),
        ("agent-two", "agent-two"),
    }


def test_one_agent_id_remains_stable_across_models():
    g = _make_guardrail(agentic=True, source="litellm-gw")
    sources = {
        g._build_payload(
            messages=[{"role": "user", "content": "hi"}],
            app_response="N/A",
            data={
                "model": model,
                "litellm_metadata": {"agent_id": "stable-agent"},
            },
            hook="pre_call",
        )["source"]
        for model in ("openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet")
    }
    assert sources == {"stable-agent"}


def test_agent_source_falls_back_to_configured_source():
    g = _make_guardrail(agentic=True, source="litellm-gw")
    payload = g._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        app_response="N/A",
        data={"model": "openai/gpt-4o-mini"},
        hook="pre_call",
    )
    assert payload["source"] == "litellm-gw"
    assert payload["metadata"]["app_name"] == "litellm-gw"


def test_authoritative_litellm_metadata_takes_precedence():
    g = _make_guardrail(agentic=True)
    payload = g._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        app_response="N/A",
        data={
            "model": "openai/gpt-4o-mini",
            "metadata": {
                "agent_id": "caller-agent",
                "session_id": "caller-session",
                "user_api_key_alias": "caller-alias",
            },
            "litellm_metadata": {
                "agent_id": "trusted-agent",
                "session_id": "trusted-session",
                "user_api_key_alias": "trusted-alias",
            },
        },
        hook="pre_call",
    )
    assert payload["source"] == "trusted-agent"
    assert payload["session_id"] == "trusted-session"
    assert payload["annotations"]["key_alias"] == "trusted-alias"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"custom_llm_provider": "azure"}, "azure"),
        ({"litellm_params": {"custom_llm_provider": "bedrock"}}, "bedrock"),
    ],
)
def test_provider_prefers_custom_llm_provider(data, expected):
    assert _resolve_provider(data, "unregistered-model") == expected


def test_provider_uses_canonical_resolver_fallback():
    assert _resolve_provider({}, "openai/gpt-4o-mini") == "openai"


def test_agentic_payload_marks_litellm_gateway_and_model():
    g = _make_guardrail(agentic=True)
    payload = g._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        app_response="N/A",
        data={
            "model": "customer-facing-alias",
            "litellm_call_id": "request-123",
            "litellm_trace_id": "trace-123",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "custom_llm_provider": "openai",
            },
            "litellm_metadata": {
                "agent_id": "agent-123",
                "agent_role": "researcher",
                "requester_ip_address": "203.0.113.10",
                "user_agent": "test-client/1.0",
            },
        },
        hook="pre_call",
    )
    md = payload["metadata"]
    assert md["gateway"] == "litellm"
    assert md["integration"] == "litellm-straiker"
    assert md["model_id"] == "openai/gpt-4o-mini"
    assert md["model_provider"] == "openai"
    assert md["event_type"] == "pre_call"
    assert md["client_ip"] == "203.0.113.10"
    assert md["user_agent"] == "test-client/1.0"
    assert md["trace_id"] == "trace-123"
    assert md["request_id"] == "request-123"
    assert md["agent_id"] == "agent-123"
    assert md["agent_role"] == "researcher"
    assert md["app_name"] == "agent-123"
    assert "model" not in md
    assert "provider" not in md
    assert "remote_ip" not in md
    assert "source" not in md
    assert payload["network"] == {
        "IP": "203.0.113.10",
        "User-Agent": "test-client/1.0",
        "Content-Type": "application/json",
    }
    assert payload["annotations"]["model"] == "customer-facing-alias"
    assert payload["annotations"]["gateway"] == "litellm"


def test_normalized_metadata_is_forwarded_and_endpoint_is_redacted():
    g = _make_guardrail(agentic=True)
    data = {
        "model": "openai/gpt-4o-mini",
        "litellm_call_id": "call-123",
        "temperature": 0.2,
        "litellm_metadata": {
            "user_api_key_alias": "team-key",
            "user_api_key_team_alias": "team",
            "user_api_key_end_user_id": "end-user",
        },
        "proxy_server_request": {"url": "https://proxy.test/v1/chat/completions?api_key=secret#fragment"},
    }
    ann = g._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        app_response="N/A",
        data=data,
        hook="pre_call",
    )["annotations"]
    assert ann["temperature"] == 0.2
    assert {key: ann[key] for key in ("key_alias", "key_team", "end_user", "request_id", "endpoint")} == {
        "key_alias": "team-key",
        "key_team": "team",
        "end_user": "end-user",
        "request_id": "call-123",
        "endpoint": "https://proxy.test/v1/chat/completions",
    }


def test_all_normalized_identity_fields_are_forwarded_when_missing():
    g = _make_guardrail(agentic=True)
    annotations = g._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        app_response="N/A",
        data={"model": "openai/gpt-4o-mini"},
        hook="pre_call",
    )["annotations"]

    assert {key: annotations[key] for key in ("key_alias", "key_team", "end_user", "request_id", "endpoint")} == {
        "key_alias": None,
        "key_team": None,
        "end_user": None,
        "request_id": None,
        "endpoint": None,
    }


def test_destination_uses_request_api_base_hostname():
    g = _make_guardrail(agentic=True)
    payload = g._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        app_response="N/A",
        data={
            "model": "openai/gpt-4o-mini",
            "api_base": "https://user:secret@dynamic.example:8443/v1?token=secret#fragment",
            "litellm_params": {"api_base": "https://other.example/v1"},
        },
        hook="pre_call",
    )
    assert payload["destination"] == "dynamic.example"
    assert "upstream_api_base" not in payload["annotations"]


def test_destination_uses_litellm_params_api_base_hostname():
    g = _make_guardrail(agentic=True)
    payload = g._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        app_response="N/A",
        data={
            "model": "openai/gpt-4o-mini",
            "litellm_params": {"api_base": "https://nested.example/v1?token=secret"},
        },
        hook="pre_call",
    )
    assert payload["destination"] == "nested.example"


def test_destination_is_unknown_without_api_base():
    g = _make_guardrail(agentic=True)
    payload = g._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        app_response="N/A",
        data={"model": "openai/gpt-4o-mini"},
        hook="pre_call",
    )
    assert payload["destination"] == "unknown"


def test_supported_event_hooks_are_exact():
    expected_hooks = [
        GuardrailEventHooks.pre_call,
        GuardrailEventHooks.during_call,
        GuardrailEventHooks.post_call,
    ]
    assert StraikerGuardrail.get_supported_event_hooks() == expected_hooks
    assert _make_guardrail().supported_event_hooks == expected_hooks


def test_initializer_forwards_schema_visible_settings():
    params = LitellmParams(
        guardrail="straiker",
        mode="pre_call",
        api_key="initializer-key",
        api_base="https://initializer.example",
        timeout=3.5,
        max_retries=4,
        initial_backoff=0.25,
        max_backoff=1.25,
        max_payload_bytes=12345,
        custom_headers={"X-Tenant": "tenant-1"},
        verbose=False,
        dedup_iterations=False,
    )
    callback = initialize_guardrail(params, {"guardrail_name": "straiker-initializer"})

    try:
        assert callback.timeout == 3.5
        assert callback.max_retries == 4
        assert callback.initial_backoff == 0.25
        assert callback.max_backoff == 1.25
        assert callback.max_payload_bytes == 12345
        assert callback.custom_headers == {"X-Tenant": "tenant-1"}
        assert callback.verbose is False
        assert callback.dedup_iterations is False
        assert callback.api_base == "https://initializer.example"
        assert callback._detect_url() == "https://initializer.example/api/v1/detect"
        assert not hasattr(callback, "destination")
        assert not hasattr(callback, "enabled_models")
        assert not hasattr(callback, "enumerate_agents")
        assert not hasattr(callback, "forward_metadata_fields")
    finally:
        litellm.callbacks.remove(callback)


def test_initializer_uses_schema_defaults_for_omitted_values():
    params = LitellmParams(
        guardrail="straiker",
        mode="pre_call",
        api_key="initializer-key",
    )
    callback = initialize_guardrail(params, {"guardrail_name": "straiker-defaults"})

    try:
        assert callback.timeout == 5.0
        assert callback.verbose is False
    finally:
        litellm.callbacks.remove(callback)


def test_initializer_rejects_invalid_threshold():
    params = LitellmParams(
        guardrail="straiker",
        mode="pre_call",
        api_key="initializer-key",
        threshold=1.1,
    )

    with pytest.raises(ValidationError):
        initialize_guardrail(params, {"guardrail_name": "straiker-invalid"})


def test_config_defaults_use_production_api_origin():
    config = StraikerGuardrailConfigModel(api_key="test")

    assert config.api_base == "https://api.prod.straiker.ai"
    assert config.verbose is False
    callback = StraikerGuardrail(api_key="test")
    assert callback.api_base == "https://api.prod.straiker.ai"
    assert callback._detect_url() == "https://api.prod.straiker.ai/api/v1/detect"
    assert callback.verbose is False


def test_agentic_mode_uses_agentic_detect_endpoint():
    callback = StraikerGuardrail(api_key="test", agentic=True)
    assert callback._detect_url() == "https://api.prod.straiker.ai/api/v1/detect?agentic"


def test_config_rejects_omitted_api_key() -> None:
    with pytest.raises(ValidationError):
        StraikerGuardrailConfigModel()


def test_config_rejects_empty_api_key() -> None:
    with pytest.raises(ValidationError):
        StraikerGuardrailConfigModel(api_key="")


def test_initializer_rejects_omitted_api_key() -> None:
    params = LitellmParams(guardrail="straiker", mode="pre_call")

    with pytest.raises(ValidationError):
        initialize_guardrail(params, {"guardrail_name": "straiker-missing-key"})


def test_initializer_rejects_empty_api_key() -> None:
    params = LitellmParams(guardrail="straiker", mode="pre_call", api_key="")

    with pytest.raises(ValidationError):
        initialize_guardrail(params, {"guardrail_name": "straiker-empty-key"})


def test_runtime_rejects_empty_api_key_even_when_environment_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRAIKER_API_KEY", "environment-key")

    with pytest.raises(ValueError, match="api_key must be non-empty"):
        StraikerGuardrail(api_key="")


def test_config_model_contains_all_initializer_options():
    assert {
        "api_key",
        "api_base",
        "agentic",
        "source",
        "threshold",
        "timeout",
        "unreachable_fallback",
        "max_retries",
        "initial_backoff",
        "max_backoff",
        "max_payload_bytes",
        "custom_headers",
        "verbose",
        "dedup_iterations",
    }.issubset(StraikerGuardrailConfigModel.model_fields)
    assert {
        "destination",
        "enabled_models",
        "enumerate_agents",
        "forward_metadata_fields",
    }.isdisjoint(StraikerGuardrailConfigModel.model_fields)
    schema = StraikerGuardrailConfigModel.model_json_schema()
    assert "api_key" in schema["required"]
    assert schema["properties"]["api_key"]["minLength"] == 1
    assert schema["properties"]["api_base"]["default"] == "https://api.prod.straiker.ai"
