"""Tests for ``item_reference`` resolution in the Responses API bridge.

Vercel ``@ai-sdk/openai`` with default ``store: true`` sends prior turn items
as opaque ``item_reference`` entries instead of inlining their content:

    input=[
        {"type": "item_reference", "id": "rs_<envelope>"},  # prior reasoning
        {"type": "item_reference", "id": "msg_<envelope>"}, # prior message
        {"role": "user", "content": "next message"}
    ]

Without resolution the bridge drops the references silently and the
conversation degrades to a single-turn fresh request. The handler now
decodes the envelope (introduced in the message-item id wrapping change),
sets ``previous_response_id``, and falls through to the existing session
handler that rebuilds the conversation from SpendLogs.
"""

from unittest.mock import AsyncMock, patch

import pytest

from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
    _resolve_item_references_to_previous_response_id,
    _strip_item_references,
)
from litellm.responses.utils import ResponsesAPIRequestUtils


def _encode_msg(raw_response_id: str) -> str:
    return ResponsesAPIRequestUtils._encode_item_envelope(
        raw_response_id,
        prefix="msg",
        custom_llm_provider="hosted_vllm",
        model_id=None,
    )


def _encode_rs(raw_response_id: str) -> str:
    return ResponsesAPIRequestUtils._encode_item_envelope(
        raw_response_id,
        prefix="rs",
        custom_llm_provider="hosted_vllm",
        model_id=None,
    )


# ---------------------------------------------------------------------------
# _resolve_item_references_to_previous_response_id (pure)
# ---------------------------------------------------------------------------


def test_resolve_returns_none_for_string_input():
    assert _resolve_item_references_to_previous_response_id("hello") is None


def test_resolve_returns_none_when_no_references():
    request_input = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert _resolve_item_references_to_previous_response_id(request_input) is None


def test_resolve_decodes_msg_envelope():
    envelope = _encode_msg("chatcmpl-abc")
    request_input = [
        {"type": "item_reference", "id": envelope},
        {"role": "user", "content": "next"},
    ]
    resolved = _resolve_item_references_to_previous_response_id(request_input)
    assert resolved is not None
    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(resolved)
    assert decoded["response_id"] == "chatcmpl-abc"


def test_resolve_decodes_rs_envelope():
    envelope = _encode_rs("chatcmpl-xyz")
    request_input = [
        {"type": "item_reference", "id": envelope},
        {"role": "user", "content": "next"},
    ]
    resolved = _resolve_item_references_to_previous_response_id(request_input)
    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(resolved)
    assert decoded["response_id"] == "chatcmpl-xyz"


def test_resolve_picks_most_recent_when_multiple_references():
    """When the input has multiple references, the most recent (last in list)
    wins so the resolved response_id points at the latest assistant turn.
    """
    older = _encode_msg("chatcmpl-older")
    newer = _encode_msg("chatcmpl-newer")
    request_input = [
        {"type": "item_reference", "id": older},
        {"role": "user", "content": "x"},
        {"type": "item_reference", "id": newer},
        {"role": "user", "content": "y"},
    ]
    resolved = _resolve_item_references_to_previous_response_id(request_input)
    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(resolved)
    assert decoded["response_id"] == "chatcmpl-newer"


def test_resolve_returns_none_for_malformed_envelopes():
    request_input = [
        {"type": "item_reference", "id": "not-an-envelope"},
        {"type": "item_reference", "id": ""},
    ]
    assert _resolve_item_references_to_previous_response_id(request_input) is None


# ---------------------------------------------------------------------------
# _strip_item_references (pure)
# ---------------------------------------------------------------------------


def test_strip_removes_only_item_references():
    request_input = [
        {"type": "item_reference", "id": "rs_xxx"},
        {"role": "user", "content": "hello"},
        {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        {"type": "item_reference", "id": "msg_yyy"},
    ]
    stripped = _strip_item_references(request_input)
    assert stripped == [
        {"role": "user", "content": "hello"},
        {"type": "function_call_output", "call_id": "call_1", "output": "result"},
    ]


def test_strip_passes_through_string_input():
    assert _strip_item_references("hello") == "hello"


def test_strip_passes_through_none():
    assert _strip_item_references(None) is None


def test_strip_returns_empty_list_when_all_references():
    stripped = _strip_item_references(
        [
            {"type": "item_reference", "id": "rs_a"},
            {"type": "item_reference", "id": "msg_b"},
        ]
    )
    assert stripped == []


# ---------------------------------------------------------------------------
# Handler integration: async_response_api_handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_handler_resolves_item_reference_and_calls_session_handler():
    """When `previous_response_id` is absent but item_references decode to a
    valid response_id, the handler delegates to the existing session handler
    with the resolved id and the stripped input.
    """
    envelope = _encode_msg("chatcmpl-prior")
    request_input = [
        {"type": "item_reference", "id": envelope},
        {"role": "user", "content": "next question"},
    ]

    with patch(
        "litellm.responses.litellm_completion_transformation.handler.LiteLLMCompletionResponsesConfig.async_responses_api_session_handler",
        new_callable=AsyncMock,
    ) as mock_session_handler, patch(
        "litellm.acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_session_handler.side_effect = (
            lambda previous_response_id, litellm_completion_request: litellm_completion_request
        )
        # Make acompletion raise so we exit before the response transformation;
        # we only want to confirm the session handler was reached with the
        # decoded id.
        mock_acompletion.side_effect = RuntimeError("stop here")

        handler = LiteLLMCompletionTransformationHandler()
        with pytest.raises(RuntimeError, match="stop here"):
            await handler.async_response_api_handler(
                litellm_completion_request={"messages": [], "model": "vllm/x"},
                request_input=request_input,
                responses_api_request={},
            )

    assert mock_session_handler.await_count == 1
    call_kwargs = mock_session_handler.await_args.kwargs
    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
        call_kwargs["previous_response_id"]
    )
    assert decoded["response_id"] == "chatcmpl-prior"


@pytest.mark.asyncio
async def test_async_handler_explicit_previous_response_id_wins_over_references():
    """If both an explicit `previous_response_id` and item_references are
    present, the explicit value wins so existing callers keep their behavior.
    """
    envelope = _encode_msg("chatcmpl-from-ref")
    request_input = [
        {"type": "item_reference", "id": envelope},
        {"role": "user", "content": "next"},
    ]

    with patch(
        "litellm.responses.litellm_completion_transformation.handler.LiteLLMCompletionResponsesConfig.async_responses_api_session_handler",
        new_callable=AsyncMock,
    ) as mock_session_handler, patch(
        "litellm.acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_session_handler.side_effect = (
            lambda previous_response_id, litellm_completion_request: litellm_completion_request
        )
        mock_acompletion.side_effect = RuntimeError("stop here")

        handler = LiteLLMCompletionTransformationHandler()
        with pytest.raises(RuntimeError):
            await handler.async_response_api_handler(
                litellm_completion_request={"messages": [], "model": "vllm/x"},
                request_input=request_input,
                responses_api_request={"previous_response_id": "explicit-id"},
            )

    assert (
        mock_session_handler.await_args.kwargs["previous_response_id"] == "explicit-id"
    )


@pytest.mark.asyncio
async def test_async_handler_no_resolution_when_no_references_or_explicit_id():
    """No `previous_response_id`, no item_references → session handler is not
    called; the request goes through with whatever messages were already
    built.
    """
    with patch(
        "litellm.responses.litellm_completion_transformation.handler.LiteLLMCompletionResponsesConfig.async_responses_api_session_handler",
        new_callable=AsyncMock,
    ) as mock_session_handler, patch(
        "litellm.acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_acompletion.side_effect = RuntimeError("stop here")

        handler = LiteLLMCompletionTransformationHandler()
        with pytest.raises(RuntimeError):
            await handler.async_response_api_handler(
                litellm_completion_request={"messages": [], "model": "vllm/x"},
                request_input=[{"role": "user", "content": "hi"}],
                responses_api_request={},
            )

    assert mock_session_handler.await_count == 0
