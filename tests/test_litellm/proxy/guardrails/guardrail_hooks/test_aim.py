"""Tests for the AIM guardrail's inspection-payload construction."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import Request, Response

from litellm import DualCache
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail


def test_aim_inspection_messages_coerces_chat_completions_tool_role_to_user():
    """LIT-4294: A valid chat-completions ``role: "tool"`` message carries a
    ``tool_call_id``, but the inspection flatten drops every field except
    ``role`` and ``content``. A bare ``tool`` message without ``tool_call_id``
    is schema-invalid per the OpenAI chat schema, and the customer's writeup
    reproduced AIM's ``/fw/v1/analyze`` returning 422 on exactly that shape.
    The AIM POST collapses the role to ``user``; the outbound request to the
    LLM is untouched."""
    data = {
        "messages": [
            {"role": "user", "content": "weather in SF"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ]
    }
    assert AimGuardrail._build_aim_inspection_messages(data) == [
        {"role": "user", "content": "weather in SF"},
        {"role": "user", "content": "sunny"},
    ]


def test_aim_inspection_messages_coerces_non_standard_caller_role_to_user():
    """LIT-4294: A caller-supplied role outside {system, user, assistant}
    (e.g. ``developer``, ``function``) is coerced to ``user`` for the AIM
    POST, since AIM validates the payload against the OpenAI chat schema
    and rejects unknown roles the same way it rejects bare ``tool``."""
    data = {
        "messages": [
            {"role": "developer", "content": "system-ish instruction"},
            {"role": "user", "content": "normal user text"},
        ]
    }
    assert AimGuardrail._build_aim_inspection_messages(data) == [
        {"role": "user", "content": "system-ish instruction"},
        {"role": "user", "content": "normal user text"},
    ]


def test_aim_inspection_messages_coerces_responses_function_call_output_role():
    """LIT-4294: the shared helper synthesises ``role: "tool"`` for a
    Responses ``function_call_output`` item (semantic equivalent of
    chat-completions tool messages). AIM's schema-validating POST cannot
    carry ``tool_call_id`` in the flat inspection payload, so AIM collapses
    that ``tool`` role to ``user`` locally before POSTing."""
    data = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "sunny"}],
            },
        ]
    }
    assert AimGuardrail._build_aim_inspection_messages(data) == [
        {"role": "user", "content": "sunny"},
    ]


def test_aim_inspection_messages_preserves_safe_roles():
    """Safe roles pass through untouched — the coercion only fires for
    roles the OpenAI chat schema flatten cannot represent standalone."""
    data = {
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    assert AimGuardrail._build_aim_inspection_messages(data) == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["pre_call", "moderation"])
@pytest.mark.parametrize("call_type", ["embedding", "aembedding"])
async def test_aim_skips_embeddings_without_calling_the_guardrail(hook: str, call_type: str):
    """/embeddings is not a conversation, so neither hook should reach AIM."""
    guardrail = AimGuardrail(api_key="hs-aim-key", guardrail_name="aim", event_hook="pre_call")
    data = {"model": "text-embedding-3-small", "input": ["first chunk", "second chunk"]}

    with patch(  # test-quality-ok: transport is litellm's aiohttp-backed handler; respx cannot intercept it
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        if hook == "pre_call":
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type=call_type,
            )
        else:
            result = await guardrail.async_moderation_hook(
                data=data,
                user_api_key_dict=UserAPIKeyAuth(),
                call_type=call_type,
            )

    mock_post.assert_not_called()
    assert result == {"model": "text-embedding-3-small", "input": ["first chunk", "second chunk"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    ["completion", "acompletion", "responses", "aresponses", "anthropic_messages", "call_mcp_tool"],
)
async def test_aim_still_inspects_every_conversational_call_type(call_type: str):
    """Deny-list, not allow-list: ``TEXT_CONTENT_CALL_TYPES`` omits these, so gating
    on it would silently stop inspecting real chat traffic."""
    guardrail = AimGuardrail(api_key="hs-aim-key", guardrail_name="aim", event_hook="pre_call")
    data = {"messages": [{"role": "user", "content": "Hi my name is Brian"}]}

    with patch(  # test-quality-ok: transport is litellm's aiohttp-backed handler; respx cannot intercept it
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=Response(
            json={
                "analysis_result": {"analysis_time_ms": 1, "policy_drill_down": {}},
                "required_action": {
                    "action_type": "block_action",
                    "detection_message": "PII detected",
                },
            },
            status_code=200,
            request=Request(method="POST", url="http://aim"),
        ),
    ) as mock_post:
        with pytest.raises(ProxyException, match="PII detected"):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type=call_type,
            )

    mock_post.assert_called_once()

