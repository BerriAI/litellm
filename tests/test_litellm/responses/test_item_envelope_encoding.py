"""Tests for envelope-encoded output item IDs in Responses API responses.

When the Responses API bridge fronts a chat-completions provider, message
output items historically inherit the upstream ``chatcmpl-*`` id from the
chat-completion response. Clients that follow the OpenAI Responses spec
expect typed prefixes (``msg_*``, ``rs_*``) and break when they receive an
``chatcmpl-*`` id (e.g. Vercel AI SDK 6.x "text part {id} not found").

These tests cover the envelope codec helpers and the integration into
``_update_responses_api_response_id_with_model_id``.
"""

from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse


# ---------------------------------------------------------------------------
# Envelope codec (pure)
# ---------------------------------------------------------------------------


def test_encode_decode_round_trip_chatcmpl_id():
    raw = "chatcmpl-abc123"
    envelope = ResponsesAPIRequestUtils._encode_item_envelope(
        raw,
        prefix="msg",
        custom_llm_provider="hosted_vllm",
        model_id="m-1",
    )
    assert envelope.startswith("msg_")
    decoded_resp_form = ResponsesAPIRequestUtils._decode_item_envelope(envelope)
    assert decoded_resp_form is not None
    assert decoded_resp_form.startswith("resp_")
    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
        decoded_resp_form
    )
    assert decoded["response_id"] == raw
    assert decoded["custom_llm_provider"] == "hosted_vllm"
    assert decoded["model_id"] == "m-1"


def test_encode_decode_round_trip_reasoning_prefix():
    raw = "chatcmpl-xyz789"
    envelope = ResponsesAPIRequestUtils._encode_item_envelope(
        raw,
        prefix="rs",
        custom_llm_provider=None,
        model_id=None,
    )
    assert envelope.startswith("rs_")
    decoded_resp_form = ResponsesAPIRequestUtils._decode_item_envelope(envelope)
    assert decoded_resp_form is not None
    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
        decoded_resp_form
    )
    assert decoded["response_id"] == raw


def test_decode_item_envelope_returns_none_for_missing_prefix():
    assert ResponsesAPIRequestUtils._decode_item_envelope("chatcmpl-abc") is None
    assert ResponsesAPIRequestUtils._decode_item_envelope("encitem_abc") is None
    assert ResponsesAPIRequestUtils._decode_item_envelope("resp_abc") is None


def test_decode_item_envelope_returns_none_for_empty_input():
    assert ResponsesAPIRequestUtils._decode_item_envelope("") is None


def test_decode_item_envelope_returns_none_for_empty_payload():
    """A degenerate ``msg_`` or ``rs_`` prefix with no payload must decode to
    ``None`` so callers (e.g. the item_reference resolver) can fall through
    to first-turn behavior instead of propagating an empty response_id to
    the session handler.
    """
    assert ResponsesAPIRequestUtils._decode_item_envelope("msg_") is None
    assert ResponsesAPIRequestUtils._decode_item_envelope("rs_") is None


def test_encode_decode_round_trip_with_item_position():
    """Distinct item positions yield distinct encoded ids but decode back to
    the same response_id payload.
    """
    raw = "chatcmpl-multi"
    first = ResponsesAPIRequestUtils._encode_item_envelope(
        raw,
        prefix="msg",
        custom_llm_provider="hosted_vllm",
        model_id="m-1",
        item_position=0,
    )
    second = ResponsesAPIRequestUtils._encode_item_envelope(
        raw,
        prefix="msg",
        custom_llm_provider="hosted_vllm",
        model_id="m-1",
        item_position=1,
    )
    assert first != second
    assert first.endswith(".0")
    assert second.endswith(".1")
    for envelope in (first, second):
        decoded_resp_form = ResponsesAPIRequestUtils._decode_item_envelope(envelope)
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
            decoded_resp_form
        )
        assert decoded["response_id"] == raw
        assert decoded["custom_llm_provider"] == "hosted_vllm"
        assert decoded["model_id"] == "m-1"


# ---------------------------------------------------------------------------
# _envelope_encode_output_item_ids
# ---------------------------------------------------------------------------


def _resp_with_message_id(message_id: str) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="chatcmpl-abc",
        object="response",
        created_at=0,
        model="hosted_vllm/test-model",
        output=[{"type": "message", "id": message_id}],
        parallel_tool_calls=False,
        temperature=0,
        tool_choice="auto",
        tools=[],
        top_p=None,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text={},
        truncation=None,
        usage=None,
        user=None,
    )


def test_envelope_encode_rewrites_chatcmpl_message_id():
    response = _resp_with_message_id("chatcmpl-abc")
    encoded = ResponsesAPIRequestUtils._envelope_encode_output_item_ids(
        response=response,
        raw_response_id="chatcmpl-abc",
        custom_llm_provider="hosted_vllm",
        model_id=None,
    )
    new_id = encoded.output[0]["id"]
    assert new_id.startswith("msg_")
    # Verify round-trip
    resp_form = ResponsesAPIRequestUtils._decode_item_envelope(new_id)
    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(resp_form)
    assert decoded["response_id"] == "chatcmpl-abc"


def test_envelope_encode_skips_already_prefixed_message_id():
    response = _resp_with_message_id("msg_existing")
    encoded = ResponsesAPIRequestUtils._envelope_encode_output_item_ids(
        response=response,
        raw_response_id="chatcmpl-abc",
        custom_llm_provider=None,
        model_id=None,
    )
    assert encoded.output[0]["id"] == "msg_existing"


def test_envelope_encode_skips_rs_prefixed_message_id():
    response = _resp_with_message_id("rs_hash123")
    encoded = ResponsesAPIRequestUtils._envelope_encode_output_item_ids(
        response=response,
        raw_response_id="chatcmpl-abc",
        custom_llm_provider=None,
        model_id=None,
    )
    assert encoded.output[0]["id"] == "rs_hash123"


def test_envelope_encode_skips_encitem_prefixed_message_id():
    response = _resp_with_message_id("encitem_abc123")
    encoded = ResponsesAPIRequestUtils._envelope_encode_output_item_ids(
        response=response,
        raw_response_id="chatcmpl-abc",
        custom_llm_provider=None,
        model_id=None,
    )
    assert encoded.output[0]["id"] == "encitem_abc123"


def test_envelope_encode_multiple_message_items_get_distinct_ids():
    """When a response carries multiple ``message`` items (parallel ``n>1``
    choices), each rewritten id must be distinct so downstream clients can
    address them individually. All ids still decode to the same response_id
    payload via :meth:`_decode_item_envelope`.
    """
    response = ResponsesAPIResponse(
        id="chatcmpl-multi",
        object="response",
        created_at=0,
        model="hosted_vllm/test-model",
        output=[
            {"type": "message", "id": "chatcmpl-multi"},
            {"type": "message", "id": "chatcmpl-multi"},
            {"type": "message", "id": "chatcmpl-multi"},
        ],
        parallel_tool_calls=False,
        temperature=0,
        tool_choice="auto",
        tools=[],
        top_p=None,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text={},
        truncation=None,
        usage=None,
        user=None,
    )
    encoded = ResponsesAPIRequestUtils._envelope_encode_output_item_ids(
        response=response,
        raw_response_id="chatcmpl-multi",
        custom_llm_provider="hosted_vllm",
        model_id=None,
    )
    ids = [item["id"] for item in encoded.output]
    assert len(set(ids)) == 3
    for new_id in ids:
        assert new_id.startswith("msg_")
        resp_form = ResponsesAPIRequestUtils._decode_item_envelope(new_id)
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
            resp_form
        )
        assert decoded["response_id"] == "chatcmpl-multi"


def test_envelope_encode_leaves_function_call_items_untouched():
    """function_call items keep their call_id-based id."""
    response = ResponsesAPIResponse(
        id="chatcmpl-abc",
        object="response",
        created_at=0,
        model="hosted_vllm/test-model",
        output=[
            {"type": "function_call", "id": "chatcmpl-fc", "call_id": "call_xyz"},
            {"type": "message", "id": "chatcmpl-abc"},
        ],
        parallel_tool_calls=False,
        temperature=0,
        tool_choice="auto",
        tools=[],
        top_p=None,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text={},
        truncation=None,
        usage=None,
        user=None,
    )
    encoded = ResponsesAPIRequestUtils._envelope_encode_output_item_ids(
        response=response,
        raw_response_id="chatcmpl-abc",
        custom_llm_provider=None,
        model_id=None,
    )
    assert encoded.output[0]["id"] == "chatcmpl-fc"
    assert encoded.output[1]["id"].startswith("msg_")


def test_envelope_encode_handles_empty_output():
    response = ResponsesAPIResponse(
        id="chatcmpl-abc",
        object="response",
        created_at=0,
        model="hosted_vllm/test-model",
        output=[],
        parallel_tool_calls=False,
        temperature=0,
        tool_choice="auto",
        tools=[],
        top_p=None,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text={},
        truncation=None,
        usage=None,
        user=None,
    )
    encoded = ResponsesAPIRequestUtils._envelope_encode_output_item_ids(
        response=response,
        raw_response_id="chatcmpl-abc",
        custom_llm_provider=None,
        model_id=None,
    )
    assert encoded.output == []


# ---------------------------------------------------------------------------
# Integration: _update_responses_api_response_id_with_model_id
# ---------------------------------------------------------------------------


def test_update_responses_api_response_id_rewrites_message_item_id():
    """The full update_responses_api_response_id flow wraps both the top-level
    response.id AND each message item id whose value is a raw chatcmpl-* form.
    """
    response = _resp_with_message_id("chatcmpl-abc")
    updated = ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
        responses_api_response=response,
        custom_llm_provider="hosted_vllm",
        litellm_metadata={"model_info": {"id": "deploy-1"}},
    )
    assert updated.id.startswith("resp_")
    assert updated.output[0]["id"].startswith("msg_")
    # The message id and the response id encode the SAME response_id payload
    msg_resp_form = ResponsesAPIRequestUtils._decode_item_envelope(
        updated.output[0]["id"]
    )
    msg_decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
        msg_resp_form
    )
    resp_decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
        updated.id
    )
    assert msg_decoded["response_id"] == resp_decoded["response_id"] == "chatcmpl-abc"
