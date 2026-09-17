"""Tests for the AIM guardrail's inspection-payload construction."""

from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest
from httpx import Request, Response

from litellm import DualCache
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail
from litellm.types.utils import ModelResponse


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


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ({}, False),
        ({"inspect_embeddings": True}, True),
        ({"inspect_embeddings": "true"}, True),
        ({"inspect_embeddings": "false"}, False),
    ],
)
def test_aim_config_plumbs_inspect_embeddings(
    configured: dict[str, object], expected: bool, monkeypatch: pytest.MonkeyPatch
):
    import litellm
    from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2

    monkeypatch.setattr(litellm, "guardrail_name_config_map", {})
    monkeypatch.setattr(litellm, "callbacks", [])

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "aim-guard",
                "litellm_params": {
                    "guardrail": "aim",
                    "mode": "pre_call",
                    "api_key": "hs-aim-key",
                    **configured,
                },
            },
        ],
        config_file_path="",
    )

    aim_guardrails = [callback for callback in litellm.callbacks if isinstance(callback, AimGuardrail)]
    assert len(aim_guardrails) == 1
    assert aim_guardrails[0].inspect_embeddings is expected


@pytest.mark.asyncio
async def test_aim_anonymize_action_redacts_batched_embeddings():
    """A batched ``input`` list of plain strings is redactable: AIM returns one
    redacted message per string, so the list is rewritten element-wise instead
    of being hard-blocked as non-text content."""
    guardrail = AimGuardrail(
        api_key="hs-aim-key",
        guardrail_name="aim",
        event_hook="pre_call",
        inspect_embeddings=True,
    )
    data = {"model": "text-embedding-3-small", "input": ["first chunk", "second chunk"]}
    response = Response(
        json={
            "required_action": {"action_type": "anonymize_action"},
            "analysis_result": {"policy_drill_down": {}},
            "redacted_chat": {
                "all_redacted_messages": [
                    {"role": "user", "content": "first [REDACTED]"},
                    {"role": "user", "content": "second [REDACTED]"},
                ]
            },
        },
        status_code=200,
        request=Request(method="POST", url="http://aim"),
    )

    with patch.object(guardrail.async_handler, "post", return_value=response):
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="aembedding",
        )

    assert result is not None
    assert result["input"] == ["first [REDACTED]", "second [REDACTED]"]


@pytest.mark.asyncio
async def test_aim_anonymize_action_blocks_when_batch_redaction_count_differs():
    """AIM returning fewer redacted messages than the batch carries cannot be
    applied element-wise. Blocking is the only safe answer: a partial rewrite
    would forward the unmatched elements to the provider unredacted."""
    guardrail = AimGuardrail(
        api_key="hs-aim-key",
        guardrail_name="aim",
        event_hook="pre_call",
        inspect_embeddings=True,
    )
    data = {"model": "text-embedding-3-small", "input": ["first SSN", "second SSN", "third SSN"]}
    response = Response(
        json={
            "required_action": {"action_type": "anonymize_action"},
            "analysis_result": {"policy_drill_down": {}},
            "redacted_chat": {"all_redacted_messages": [{"role": "user", "content": "first [REDACTED]"}]},
        },
        status_code=200,
        request=Request(method="POST", url="http://aim"),
    )

    with patch.object(guardrail.async_handler, "post", return_value=response):
        with pytest.raises(ProxyException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="aembedding",
            )

    assert exc_info.value.code == "400"
    assert data["input"] == ["first SSN", "second SSN", "third SSN"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "all_redacted_messages",
    [
        pytest.param([{"role": "user"}], id="content-missing"),
        pytest.param([{"role": "user", "content": None}], id="content-null"),
        pytest.param(["first [REDACTED]"], id="not-a-mapping"),
        pytest.param([], id="empty-list"),
        pytest.param("invalid", id="missing-collection"),
    ],
)
@pytest.mark.parametrize(
    ("request_body", "call_type"),
    [
        pytest.param({"model": "text-embedding-3-small", "input": ["first SSN"]}, "aembedding", id="batch-input"),
        pytest.param(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "first SSN"}]},
            "acompletion",
            id="chat-messages",
        ),
    ],
)
async def test_aim_anonymize_action_blocks_malformed_redacted_messages(
    all_redacted_messages: object, request_body: dict, call_type: str
):
    """Malformed AIM redactions return a controlled 400 without changing the request."""
    guardrail = AimGuardrail(
        api_key="hs-aim-key",
        guardrail_name="aim",
        event_hook="pre_call",
        inspect_embeddings=True,
    )
    data = deepcopy(request_body)
    response = Response(
        json={
            "required_action": {"action_type": "anonymize_action"},
            "analysis_result": {"policy_drill_down": {}},
            "redacted_chat": {"all_redacted_messages": all_redacted_messages},
        },
        status_code=200,
        request=Request(method="POST", url="http://aim"),
    )

    with patch.object(guardrail.async_handler, "post", return_value=response):
        with pytest.raises(ProxyException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type=call_type,
            )

    assert exc_info.value.code == "400"
    assert data == request_body


_OUTPUT_REQUEST = {
    "messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "repeat my SSN"},
    ]
}
_OUTPUT_ECHO = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "repeat my SSN"},
]


def _completion(content: str) -> ModelResponse:
    return ModelResponse(
        choices=[{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": content}}]
    )


def _anonymize_response(all_redacted_messages: object) -> Response:
    return Response(
        json={
            "required_action": {"action_type": "anonymize_action"},
            "analysis_result": {"policy_drill_down": {}},
            "redacted_chat": {"all_redacted_messages": all_redacted_messages},
        },
        status_code=200,
        request=Request(method="POST", url="http://aim"),
    )


@pytest.mark.asyncio
async def test_aim_output_anonymize_takes_the_assistant_entry_after_the_echoed_request():
    """AIM echoes every inspected request message before the assistant turn, so the
    redacted completion is the final entry of a batch one longer than the request."""
    guardrail = AimGuardrail(api_key="hs-aim-key", guardrail_name="aim", event_hook="post_call")
    response = _completion("your SSN is 123-45-6789")

    with patch.object(
        guardrail.async_handler,
        "post",
        return_value=_anonymize_response([*_OUTPUT_ECHO, {"role": "assistant", "content": "your SSN is [REDACTED]"}]),
    ):
        result = await guardrail.async_post_call_success_hook(
            data=deepcopy(_OUTPUT_REQUEST), user_api_key_dict=UserAPIKeyAuth(), response=response
        )

    assert result["choices"][0]["message"]["content"] == "your SSN is [REDACTED]"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "all_redacted_messages",
    [
        pytest.param(_OUTPUT_ECHO, id="assistant-entry-missing"),
        pytest.param([{"role": "assistant", "content": "your SSN is [REDACTED]"}], id="request-echo-missing"),
        pytest.param([*_OUTPUT_ECHO, {"role": "assistant", "content": ""}], id="assistant-content-empty"),
        pytest.param([*_OUTPUT_ECHO, {"role": "assistant", "content": None}], id="assistant-content-null"),
        pytest.param([*_OUTPUT_ECHO, "your SSN is [REDACTED]"], id="not-a-mapping"),
        pytest.param([], id="empty-list"),
        pytest.param("invalid", id="missing-collection"),
    ],
)
async def test_aim_output_anonymize_blocks_malformed_redactions(all_redacted_messages: object):
    """A redaction AIM cannot be aligned to the completion is a 400, never the
    unredacted completion and never a 500."""
    guardrail = AimGuardrail(api_key="hs-aim-key", guardrail_name="aim", event_hook="post_call")
    response = _completion("your SSN is 123-45-6789")

    with patch.object(guardrail.async_handler, "post", return_value=_anonymize_response(all_redacted_messages)):
        with pytest.raises(ProxyException) as exc_info:
            await guardrail.async_post_call_success_hook(
                data=deepcopy(_OUTPUT_REQUEST), user_api_key_dict=UserAPIKeyAuth(), response=response
            )

    assert exc_info.value.code == "400"
    assert response["choices"][0]["message"]["content"] == "your SSN is 123-45-6789"


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["pre_call", "moderation"])
@pytest.mark.parametrize("call_type", ["embedding", "aembedding"])
async def test_aim_inspects_embeddings_when_enabled(hook: str, call_type: str):
    guardrail = AimGuardrail(
        api_key="hs-aim-key",
        guardrail_name="aim",
        event_hook="pre_call",
        inspect_embeddings=True,
    )
    data = {"model": "text-embedding-3-small", "input": ["first chunk", "second chunk"]}

    with patch.object(
        guardrail.async_handler,
        "post",
        return_value=Response(
            json={"required_action": None, "analysis_result": {"policy_drill_down": {}}},
            status_code=200,
            request=Request(method="POST", url="http://aim"),
        ),
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

    mock_post.assert_called_once()
    assert result == data


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
