"""
Tests for encrypted_content_affinity pre-call check.

The mechanism works without any cache and supports two encoding strategies:

1. **Items with IDs**: item IDs for output items with `encrypted_content` are rewritten to
   `encitem_{base64("litellm:model_id:{model_id};item_id:{original_id}")}`.

2. **Items without IDs** (Codex): encrypted_content itself is wrapped with model_id metadata:
   `litellm_enc:{base64("model_id:{model_id}")};{original_encrypted_content}`.

- On routing: `EncryptedContentAffinityCheck` decodes from either item IDs or wrapped
  encrypted_content to extract `model_id` and pins the request to that deployment.
- Before forwarding: `_restore_encrypted_content_item_ids_in_input` decodes IDs and unwraps
  encrypted_content back to their original forms before sending to the upstream provider.
"""

import time
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mock_response(output_items, response_id="resp_mock-123"):
    """Build a ResponsesAPIResponse that ``async_response_api_handler`` would return."""
    return ResponsesAPIResponse(
        id=response_id,
        created_at=1741476542,
        status="completed",
        model="openai/gpt-5.1-codex",
        output=output_items,
        usage={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
    )


def _get_item_id(item) -> str:
    """Extract item ID from either a Pydantic model or a dict."""
    if isinstance(item, dict):
        return item.get("id", "")
    return getattr(item, "id", "") or ""


def _extract_encoded_item_id(response) -> str:
    """Return the first ``encitem_``-prefixed item ID from the response output."""
    for item in response.output or []:
        item_id = _get_item_id(item)
        if item_id.startswith("encitem_"):
            return item_id
    return ""


# ---------------------------------------------------------------------------
# Unit tests for encoding / decoding utilities
# ---------------------------------------------------------------------------


class TestEncryptedItemIdCodec:
    def test_roundtrip(self):
        model_id = "deployment-1"
        original_item_id = "rs_abc123def456"
        encoded = ResponsesAPIRequestUtils._build_encrypted_item_id(model_id, original_item_id)
        assert encoded.startswith("encitem_")
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(encoded)
        assert decoded is not None
        assert decoded["model_id"] == model_id
        assert decoded["item_id"] == original_item_id

    def test_decode_without_padding(self):
        """Decoding must succeed even if base64 padding (=) was stripped in transit."""
        model_id = "gpt-5.1-codex-openai-2"
        original_item_id = "rs_0efb96cb222403210069a01d5d52588196a9dc394ffdb89d00"
        encoded = ResponsesAPIRequestUtils._build_encrypted_item_id(model_id, original_item_id)
        # Strip any trailing '=' to simulate what happens in transit
        stripped = encoded.rstrip("=")
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(stripped)
        assert decoded is not None
        assert decoded["model_id"] == model_id
        assert decoded["item_id"] == original_item_id

    def test_non_encoded_id_returns_none(self):
        assert ResponsesAPIRequestUtils._decode_encrypted_item_id("rs_abc123") is None
        assert ResponsesAPIRequestUtils._decode_encrypted_item_id("msg_abc") is None
        assert ResponsesAPIRequestUtils._decode_encrypted_item_id("") is None

    def test_semicolon_in_item_id(self):
        """item_id values containing ';' must survive the roundtrip."""
        model_id = "deployment-1"
        original_item_id = "rs_part1;part2;part3"
        encoded = ResponsesAPIRequestUtils._build_encrypted_item_id(model_id, original_item_id)
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(encoded)
        assert decoded is not None
        assert decoded["item_id"] == original_item_id


class TestUpdateEncryptedContentItemIds:
    def test_rewrites_encrypted_items_in_dict_response(self):
        model_id = "deployment-1"
        response = {
            "id": "resp_123",
            "output": [
                {"id": "msg_abc", "type": "message", "content": []},
                {"id": "rs_xyz", "type": "reasoning", "encrypted_content": "secret"},
            ],
        }
        result = ResponsesAPIRequestUtils._update_encrypted_content_item_ids_in_response(response, model_id)
        # Plain message item untouched
        assert result["output"][0]["id"] == "msg_abc"
        # Reasoning item with encrypted_content gets encoded
        encoded_id = result["output"][1]["id"]
        assert encoded_id.startswith("encitem_")
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(encoded_id)
        assert decoded["model_id"] == model_id
        assert decoded["item_id"] == "rs_xyz"

    def test_no_op_when_model_id_is_none(self):
        response = {"output": [{"id": "rs_xyz", "type": "reasoning", "encrypted_content": "secret"}]}
        result = ResponsesAPIRequestUtils._update_encrypted_content_item_ids_in_response(response, None)
        assert result["output"][0]["id"] == "rs_xyz"


class TestEncryptedContentWrapping:
    def test_wrap_and_unwrap_encrypted_content(self):
        """Test wrapping encrypted_content with model_id metadata."""
        model_id = "deployment-1"
        original_content = "gAAAAABpnW_yEYmSNEyOG_original_encrypted_data"
        wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(original_content, model_id)
        assert wrapped.startswith("litellm_enc:")
        assert wrapped != original_content

        (
            unwrapped_model_id,
            unwrapped_content,
        ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(wrapped)
        assert unwrapped_model_id == model_id
        assert unwrapped_content == original_content

    def test_unwrap_plain_encrypted_content(self):
        """Unwrapping plain encrypted_content returns None for model_id."""
        plain_content = "gAAAAABpnW_yEYmSNEyOG_plain_content"
        (
            model_id,
            content,
        ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(plain_content)
        assert model_id is None
        assert content == plain_content

    def test_update_response_wraps_encrypted_content_without_id(self):
        """Items with encrypted_content but no ID get the content wrapped."""
        model_id = "deployment-1"
        response = {
            "id": "resp_123",
            "output": [
                {"type": "message", "content": []},
                {
                    "type": "reasoning",
                    "encrypted_content": "gAAAAABpnW_yEYmSNEyOG_secret",
                },
            ],
        }
        result = ResponsesAPIRequestUtils._update_encrypted_content_item_ids_in_response(response, model_id)
        assert result["output"][0].get("encrypted_content") is None
        wrapped = result["output"][1]["encrypted_content"]
        assert wrapped.startswith("litellm_enc:")

        (
            model_id_extracted,
            unwrapped,
        ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(wrapped)
        assert model_id_extracted == model_id
        assert unwrapped == "gAAAAABpnW_yEYmSNEyOG_secret"


class TestRestoreEncryptedContentItemIds:
    def test_restores_encoded_ids(self):
        model_id = "deployment-1"
        original_id = "rs_encrypted_item_456"
        encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id(model_id, original_id)

        request_input = [
            {"type": "message", "id": "msg_abc123", "role": "assistant"},
            {"type": "reasoning", "id": encoded_id, "encrypted_content": "secret"},
        ]
        restored = ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input(request_input)
        assert restored[0]["id"] == "msg_abc123"
        assert restored[1]["id"] == original_id

    def test_unwraps_encrypted_content(self):
        """Test that wrapped encrypted_content is unwrapped before forwarding."""
        model_id = "deployment-1"
        original_content = "gAAAAABpnW_yEYmSNEyOG_original"
        wrapped_content = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(original_content, model_id)

        request_input = [
            {"type": "reasoning", "encrypted_content": wrapped_content},
        ]
        restored = ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input(request_input)
        assert restored[0]["encrypted_content"] == original_content

    def test_no_op_for_plain_string_input(self):
        result = ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input("Hello world")
        assert result == "Hello world"

    def test_no_op_for_unencoded_ids(self):
        request_input = [{"type": "message", "id": "msg_plain"}]
        result = ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input(request_input)
        assert result[0]["id"] == "msg_plain"


# ---------------------------------------------------------------------------
# Integration tests (router-level)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encrypted_content_affinity_tracks_and_routes():
    """
    The first response rewrites encrypted-content item IDs to encoded form.
    The follow-up request with those encoded IDs is pinned to the same deployment.

    Mocks ``async_response_api_handler`` (the method that makes the HTTP call)
    so the test is deterministic regardless of the HTTP transport in use.
    The ``@client`` decorator and ``_update_responses_api_response_id_with_model_id``
    post-processing still run, so item-ID rewriting is exercised end-to-end.
    """
    mock_resp = _build_mock_response(
        output_items=[
            {
                "type": "message",
                "id": "msg_abc123",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
            },
            {
                "type": "reasoning",
                "id": "rs_encrypted_item_456",
                "status": "completed",
                "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
            },
        ],
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                },
                "model_info": {"id": "deployment-1"},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                },
                "model_info": {"id": "deployment-2"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    selected_deployments = []

    def deterministic_choice(seq):
        if len(selected_deployments) == 0:
            return seq[0]
        return seq[1] if len(seq) > 1 else seq[0]

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=deterministic_choice,
        ),
    ):
        # First request — goes to deployment-1 via deterministic_choice
        first_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input="Hello, how are you?",
        )
        first_model_id = first_response._hidden_params["model_id"]
        selected_deployments.append(first_model_id)

        # The response must have rewritten the encrypted item's ID to encoded form
        encoded_item_id = _extract_encoded_item_id(first_response)
        assert encoded_item_id.startswith("encitem_"), (
            f"Expected output item ID to be rewritten to encitem_... but got {encoded_item_id!r}"
        )

        # Verify the encoded ID decodes back to the correct deployment + original ID
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(encoded_item_id)
        assert decoded is not None
        assert decoded["model_id"] == first_model_id
        assert decoded["item_id"] == "rs_encrypted_item_456"

        # Second request: use the encoded item IDs from the first response
        second_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input=[
                {"type": "message", "id": "msg_abc123", "role": "assistant"},
                {
                    "type": "reasoning",
                    "id": encoded_item_id,
                    "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
                },
            ],
        )
        second_model_id = second_response._hidden_params["model_id"]

        assert second_model_id == first_model_id, (
            f"Expected affinity to route to {first_model_id}, but got {second_model_id}"
        )


@pytest.mark.asyncio
async def test_encrypted_content_affinity_no_effect_on_chat_completions():
    """
    Encrypted content affinity should not affect regular chat completions.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "mock_response": "Hello from chat completion!",
                },
                "model_info": {"id": "chat-deployment-1"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    response1 = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
    )
    response2 = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello again"}],
    )
    assert response1.id is not None
    assert response2.id is not None


@pytest.mark.asyncio
async def test_encrypted_content_affinity_bypasses_rpm_limits():
    """
    When encrypted content affinity pins to a deployment, the request
    goes through even if normal routing would avoid it (usage-based-routing-v2).
    """
    mock_resp = _build_mock_response(
        output_items=[
            {
                "type": "reasoning",
                "id": "rs_encrypted_must_pin",
                "status": "completed",
                "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
            },
        ],
        response_id="resp_mock-rpm-test",
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                },
                "model_info": {"id": "deployment-alpha"},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                },
                "model_info": {"id": "deployment-beta"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
        routing_strategy="usage-based-routing-v2",
        num_retries=0,
    )

    selected_deployments = []

    def deterministic_choice(seq):
        if len(selected_deployments) == 0:
            return seq[0]
        return seq[1] if len(seq) > 1 else seq[0]

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=deterministic_choice,
        ),
    ):
        first_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input="Initial request",
        )
        first_model_id = first_response._hidden_params["model_id"]
        selected_deployments.append(first_model_id)

        # Extract encoded item ID from the first response output
        encoded_item_id = _extract_encoded_item_id(first_response)
        assert encoded_item_id.startswith("encitem_"), f"Expected encitem_... but got {encoded_item_id!r}"

        # Follow-up with the encoded item ID — should pin to same deployment
        second_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input=[
                {
                    "type": "reasoning",
                    "id": encoded_item_id,
                    "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
                },
            ],
        )
        second_model_id = second_response._hidden_params["model_id"]

        assert second_model_id == first_model_id


@pytest.mark.asyncio
async def test_encrypted_content_affinity_no_match_normal_routing():
    """
    Input items with non-encoded IDs (no encitem_ prefix) fall through to
    normal load balancing.
    """
    mock_resp = _build_mock_response(
        output_items=[
            {
                "type": "message",
                "id": "msg_new",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Response"}],
            },
        ],
        response_id="resp_mock-no-match",
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                },
                "model_info": {"id": "deployment-a"},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                },
                "model_info": {"id": "deployment-b"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
        new_callable=AsyncMock,
        return_value=mock_resp,
    ):
        # Non-encoded item ID — no affinity should kick in
        response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input=[
                {"type": "message", "id": "unknown_item_id_12345"},
            ],
        )
        assert response.id is not None


@pytest.mark.asyncio
async def test_encrypted_content_affinity_with_wrapped_content_no_id():
    """
    Test affinity routing when items have wrapped encrypted_content but no ID.
    This simulates Codex client behavior where IDs are omitted.
    """
    mock_resp = _build_mock_response(
        output_items=[
            {
                "type": "reasoning",
                "status": "completed",
                "encrypted_content": "gAAAAABpnW_yEYmSNEyOG_original_content",
            },
        ],
        response_id="resp_mock-wrapped-content",
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                },
                "model_info": {"id": "deployment-1"},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                },
                "model_info": {"id": "deployment-2"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    selected_deployments = []

    def deterministic_choice(seq):
        if len(selected_deployments) == 0:
            return seq[0]
        return seq[1] if len(seq) > 1 else seq[0]

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=deterministic_choice,
        ),
    ):
        # First request — goes to deployment-1
        first_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input="Hello, how are you?",
        )
        first_model_id = first_response._hidden_params["model_id"]
        selected_deployments.append(first_model_id)

        # Extract wrapped encrypted_content from first response
        first_item = first_response.output[0]
        wrapped_content = (
            first_item.encrypted_content
            if hasattr(first_item, "encrypted_content")
            else first_item.get("encrypted_content")
        )
        assert wrapped_content.startswith("litellm_enc:"), f"Expected wrapped content but got {wrapped_content[:50]}..."

        # Verify we can extract model_id from wrapped content
        (
            extracted_model_id,
            _,
        ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(wrapped_content)
        assert extracted_model_id == first_model_id

        # Second request: use wrapped encrypted_content WITHOUT an ID (Codex behavior)
        second_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input=[
                {
                    "type": "reasoning",
                    "encrypted_content": wrapped_content,
                },
            ],
        )
        second_model_id = second_response._hidden_params["model_id"]

        assert second_model_id == first_model_id, (
            f"Expected affinity to route to {first_model_id}, but got {second_model_id}"
        )


def test_encrypted_content_wrapping_preserves_original_content():
    """
    Test that wrapping and unwrapping encrypted_content preserves the original content.
    This is critical for streaming responses where content must round-trip correctly.
    """
    model_id = "test-deployment-1"
    original_encrypted_content = "gAAAAABpnW_yEYmSNEyOG_streaming_test_content_with_special_chars==+/"

    wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(original_encrypted_content, model_id)

    assert wrapped.startswith("litellm_enc:")
    assert wrapped != original_encrypted_content

    (
        extracted_model_id,
        unwrapped_content,
    ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(wrapped)

    assert extracted_model_id == model_id
    assert unwrapped_content == original_encrypted_content


def test_encrypted_content_wrapping_with_multiple_semicolons():
    """
    Test that encrypted_content containing semicolons is handled correctly.
    """
    model_id = "deployment-with-semicolons"
    original_content = "gAAAAAB;some;content;with;semicolons"

    wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(original_content, model_id)

    (
        extracted_model_id,
        unwrapped,
    ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(wrapped)

    assert extracted_model_id == model_id
    assert unwrapped == original_content


# ---------------------------------------------------------------------------
# Regression tests: affinity check must not break tag-based routing
# ---------------------------------------------------------------------------

from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
    EncryptedContentAffinityCheck,
)


@pytest.mark.asyncio
async def test_encrypted_content_affinity_does_not_create_litellm_metadata_for_chat():
    """
    For chat completions / embeddings, request_kwargs uses 'metadata' (not
    'litellm_metadata').  The affinity check must NOT create a spurious
    'litellm_metadata' key, because that would cause
    _get_metadata_variable_name_from_kwargs to return 'litellm_metadata'
    and tag-based routing would look for tags in the wrong dict.
    """
    check = EncryptedContentAffinityCheck()
    deployments = [
        {"model_info": {"id": "dep-1"}, "litellm_params": {"model": "gpt-4"}},
    ]
    request_kwargs = {"metadata": {"tags": ["prod"]}}

    result = await check.async_filter_deployments(
        model="gpt-4",
        healthy_deployments=deployments,
        messages=[{"role": "user", "content": "hi"}],
        request_kwargs=request_kwargs,
    )

    # Must not inject litellm_metadata
    assert "litellm_metadata" not in request_kwargs
    # Tags must be untouched
    assert request_kwargs["metadata"]["tags"] == ["prod"]
    # All deployments returned (no pinning)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_encrypted_content_affinity_preserves_litellm_metadata_for_responses():
    """
    For Responses API calls, litellm_metadata already exists.  The affinity
    check should set the flag there and preserve existing keys.
    """
    check = EncryptedContentAffinityCheck()
    deployments = [
        {"model_info": {"id": "dep-1"}, "litellm_params": {"model": "gpt-5.1-codex"}},
    ]
    request_kwargs = {
        "litellm_metadata": {"model_info": {"id": "dep-1"}},
    }

    await check.async_filter_deployments(
        model="gpt-5.1-codex",
        healthy_deployments=deployments,
        messages=None,
        request_kwargs=request_kwargs,
    )

    assert request_kwargs["litellm_metadata"]["encrypted_content_affinity_enabled"] is True
    assert request_kwargs["litellm_metadata"]["model_info"] == {"id": "dep-1"}


def test_encrypted_content_wrapping_empty_string():
    """
    Test that empty encrypted_content is handled gracefully.
    """
    model_id = "test-deployment"
    original_content = ""

    wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(original_content, model_id)

    assert wrapped.startswith("litellm_enc:")

    (
        extracted_model_id,
        unwrapped,
    ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(wrapped)

    assert extracted_model_id == model_id
    assert unwrapped == original_content


# ---------------------------------------------------------------------------
# LIT-2531: cross-model-group fallback via encryption boundary (api_base + api_key)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_affinity_falls_back_to_same_encryption_boundary_on_model_group_switch():
    """
    LIT-2531: Client starts a session on gpt-5.3-codex, follow-up switches to
    gpt-5.4 mid-chat (e.g. via Codex `model_migrations`). Affinity must pin to
    the gpt-5.4 deployment on the SAME Azure resource as the originating
    gpt-5.3-codex deployment -- otherwise Azure rejects the encrypted_content.
    """
    first_resp = _build_mock_response(
        output_items=[
            {
                "type": "reasoning",
                "id": "rs_encrypted_xyz",
                "status": "completed",
                "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
            },
        ],
        response_id="resp_first",
    )
    second_resp = _build_mock_response(
        output_items=[
            {
                "type": "message",
                "id": "msg_ok",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            },
        ],
        response_id="resp_second",
    )

    ACCOUNT_A_BASE = "https://account-a.openai.azure.com/"
    ACCOUNT_A_KEY = "key-a"
    ACCOUNT_B_BASE = "https://account-b.openai.azure.com/"
    ACCOUNT_B_KEY = "key-b"

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5.3-codex",
                "litellm_params": {
                    "model": "azure/gpt-5.3-codex",
                    "api_base": ACCOUNT_A_BASE,
                    "api_key": ACCOUNT_A_KEY,
                    "api_version": "2025-04-01-preview",
                },
                "model_info": {"id": "gpt-5.3-codex-account-a"},
            },
            {
                "model_name": "gpt-5.3-codex",
                "litellm_params": {
                    "model": "azure/gpt-5.3-codex",
                    "api_base": ACCOUNT_B_BASE,
                    "api_key": ACCOUNT_B_KEY,
                    "api_version": "2025-04-01-preview",
                },
                "model_info": {"id": "gpt-5.3-codex-account-b"},
            },
            {
                "model_name": "gpt-5.4",
                "litellm_params": {
                    "model": "azure/gpt-5.4",
                    "api_base": ACCOUNT_A_BASE,
                    "api_key": ACCOUNT_A_KEY,
                    "api_version": "2025-04-01-preview",
                },
                "model_info": {"id": "gpt-5.4-account-a"},
            },
            {
                "model_name": "gpt-5.4",
                "litellm_params": {
                    "model": "azure/gpt-5.4",
                    "api_base": ACCOUNT_B_BASE,
                    "api_key": ACCOUNT_B_KEY,
                    "api_version": "2025-04-01-preview",
                },
                "model_info": {"id": "gpt-5.4-account-b"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    def first_call_picks_account_a(seq):
        for d in seq:
            if d["model_info"]["id"] == "gpt-5.3-codex-account-a":
                return d
        return seq[0]

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new_callable=AsyncMock,
            return_value=first_resp,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=first_call_picks_account_a,
        ),
    ):
        r1 = await router.aresponses(model="gpt-5.3-codex", input="hi")

    assert r1._hidden_params["model_id"] == "gpt-5.3-codex-account-a"
    encoded_id = _extract_encoded_item_id(r1)
    assert encoded_id.startswith("encitem_")

    # simple_shuffle.random.choice NOT patched: prove affinity narrows the
    # candidate pool to a single deployment regardless of which one shuffle picks.
    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
        new_callable=AsyncMock,
        return_value=second_resp,
    ):
        r2 = await router.aresponses(
            model="gpt-5.4",
            input=[
                {
                    "type": "reasoning",
                    "id": encoded_id,
                    "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
                },
            ],
        )

    assert r2._hidden_params["model_id"] == "gpt-5.4-account-a"


@pytest.mark.asyncio
async def test_affinity_falls_back_to_same_boundary_on_alias_switch():
    """
    LIT-2531 alias path: gpt-5.2-codex is a LiteLLM alias that points at the
    same underlying Azure model as gpt-5.3-codex. Different model_name groups
    in the router, so model_id-based pinning misses, but the encryption
    boundary (api_base + api_key) is identical -> follow-up must still pin.
    """
    first_resp = _build_mock_response(
        output_items=[
            {
                "type": "reasoning",
                "id": "rs_alias_xyz",
                "status": "completed",
                "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
            },
        ],
        response_id="resp_alias_first",
    )
    second_resp = _build_mock_response(
        output_items=[
            {
                "type": "message",
                "id": "msg_alias_ok",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            },
        ],
        response_id="resp_alias_second",
    )

    ACCOUNT_A_BASE = "https://account-a.openai.azure.com/"
    ACCOUNT_A_KEY = "key-a"
    ACCOUNT_B_BASE = "https://account-b.openai.azure.com/"
    ACCOUNT_B_KEY = "key-b"

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5.3-codex",
                "litellm_params": {
                    "model": "azure/gpt-5.3-codex",
                    "api_base": ACCOUNT_A_BASE,
                    "api_key": ACCOUNT_A_KEY,
                    "api_version": "2025-04-01-preview",
                },
                "model_info": {"id": "gpt-5.3-codex-account-a"},
            },
            {
                "model_name": "gpt-5.3-codex",
                "litellm_params": {
                    "model": "azure/gpt-5.3-codex",
                    "api_base": ACCOUNT_B_BASE,
                    "api_key": ACCOUNT_B_KEY,
                    "api_version": "2025-04-01-preview",
                },
                "model_info": {"id": "gpt-5.3-codex-account-b"},
            },
            {
                "model_name": "gpt-5.2-codex",
                "litellm_params": {
                    "model": "azure/gpt-5.3-codex",
                    "api_base": ACCOUNT_A_BASE,
                    "api_key": ACCOUNT_A_KEY,
                    "api_version": "2025-04-01-preview",
                },
                "model_info": {"id": "gpt-5.2-codex-account-a"},
            },
            {
                "model_name": "gpt-5.2-codex",
                "litellm_params": {
                    "model": "azure/gpt-5.3-codex",
                    "api_base": ACCOUNT_B_BASE,
                    "api_key": ACCOUNT_B_KEY,
                    "api_version": "2025-04-01-preview",
                },
                "model_info": {"id": "gpt-5.2-codex-account-b"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=0,
    )

    def pick_account_a(seq):
        for d in seq:
            if d["model_info"]["id"] == "gpt-5.3-codex-account-a":
                return d
        return seq[0]

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new_callable=AsyncMock,
            return_value=first_resp,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=pick_account_a,
        ),
    ):
        r1 = await router.aresponses(model="gpt-5.3-codex", input="hi")

    encoded_id = _extract_encoded_item_id(r1)
    assert encoded_id.startswith("encitem_")

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
        new_callable=AsyncMock,
        return_value=second_resp,
    ):
        r2 = await router.aresponses(
            model="gpt-5.2-codex",
            input=[
                {
                    "type": "reasoning",
                    "id": encoded_id,
                    "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
                },
            ],
        )

    assert r2._hidden_params["model_id"] == "gpt-5.2-codex-account-a"


def test_boundary_fallback_no_router_ref_returns_empty():
    """
    Standalone use (no router wired in) -> the boundary lookup short-circuits
    to ``[]`` instead of crashing on ``None.get_deployment``.
    """
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    check = EncryptedContentAffinityCheck(router=None)
    healthy = [
        {
            "model_info": {"id": "dep-1"},
            "litellm_params": {"api_base": "https://x", "api_key": "k"},
        }
    ]
    matches, originating = check._find_deployments_on_same_encryption_boundary(
        healthy_deployments=healthy,
        model_id="dep-2",
    )
    assert matches == []
    assert originating is None


def test_boundary_fallback_originating_deployment_removed_returns_empty():
    """
    If the originating deployment has been removed from the router (e.g. via
    /model/delete), ``router.get_deployment`` returns None and we return [] so
    the caller falls back to the full healthy_deployments list.
    """
    from unittest.mock import MagicMock

    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    mock_router = MagicMock()
    mock_router.get_deployment.return_value = None

    check = EncryptedContentAffinityCheck(router=mock_router)
    healthy = [
        {
            "model_info": {"id": "dep-1"},
            "litellm_params": {"api_base": "https://x", "api_key": "k"},
        }
    ]
    matches, originating = check._find_deployments_on_same_encryption_boundary(
        healthy_deployments=healthy,
        model_id="dep-removed",
    )
    assert matches == []
    assert originating is None
    mock_router.get_deployment.assert_called_once_with(model_id="dep-removed")


def test_boundary_key_accepts_pydantic_litellm_params_instance():
    """
    Regression: ``_encryption_boundary_key`` must accept any object exposing
    dict-style ``.get()`` (incl. ``LiteLLM_Params`` Pydantic instances) — not
    just plain dicts.

    A stricter ``isinstance(dict)`` guard would silently return ``None`` for a
    ``LiteLLM_Params`` value, drop the deployment from boundary matching, and
    fall back to the full pool — which is the exact ``invalid_encrypted_content``
    failure this check exists to prevent.
    """
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )
    from litellm.types.router import LiteLLM_Params

    pydantic_params = LiteLLM_Params(
        model="azure/gpt-5.3-codex",
        api_base="https://mateo-resource.openai.azure.com",
        api_key="fake-azure-resource-key-a",
    )
    plain_params = {
        "model": "azure/gpt-5.3-codex",
        "api_base": "https://mateo-resource.openai.azure.com",
        "api_key": "fake-azure-resource-key-a",
    }

    pydantic_key = EncryptedContentAffinityCheck._encryption_boundary_key(pydantic_params)
    plain_key = EncryptedContentAffinityCheck._encryption_boundary_key(plain_params)

    assert pydantic_key is not None
    assert (
        pydantic_key
        == plain_key
        == (
            "https://mateo-resource.openai.azure.com",
            "fake-azure-resource-key-a",
        )
    )


def test_boundary_key_rejects_non_dict_like_inputs():
    """
    Inputs that don't expose ``.get()`` (None, lists, strings, ints) -> None.
    Guards against accidentally treating a stray non-dict-like value as a
    valid boundary.
    """
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    for bad in (None, [], "not a dict", 42, object()):
        assert EncryptedContentAffinityCheck._encryption_boundary_key(bad) is None

    assert EncryptedContentAffinityCheck._encryption_boundary_key({"api_base": "", "api_key": "k"}) is None
    assert EncryptedContentAffinityCheck._encryption_boundary_key({"api_base": "https://x"}) is None


# ---------------------------------------------------------------------------
# Fail-fast when originating deployment is unavailable and no boundary peer
# ---------------------------------------------------------------------------


def _make_originating_mock(api_base: str, api_key: str, model_name: str = "gpt-5.4"):
    from unittest.mock import MagicMock

    originating = MagicMock()
    originating.model_name = model_name
    originating.litellm_params.model_dump.return_value = {
        "api_base": api_base,
        "api_key": api_key,
    }
    return originating


def _make_router_mock_with_cooldown(
    originating,
    cooldown_entries: list[tuple] | None = None,
    routed_group_model_ids: list[str] | None = None,
):
    """
    Build a MagicMock router whose ``cooldown_cache.async_get_active_cooldowns``
    returns ``cooldown_entries`` (defaulting to ``[]`` — no active cooldown), and
    whose ``get_candidate_model_ids_for_route`` returns ``routed_group_model_ids``
    (the deployment ids the router resolves for the routed model; defaulting to ``[]``
    — origin absent from the routed group, i.e. a tier change).
    """
    from unittest.mock import AsyncMock, MagicMock

    mock_router = MagicMock()
    mock_router.get_deployment.return_value = originating
    mock_router.cooldown_cache.async_get_active_cooldowns = AsyncMock(return_value=list(cooldown_entries or []))
    mock_router.get_candidate_model_ids_for_route.return_value = frozenset(routed_group_model_ids or [])
    return mock_router


@pytest.mark.asyncio
async def test_affinity_raises_service_unavailable_when_origin_cooled_for_non_429():
    """
    Originating deployment is in the router config, in cooldown for a non-429
    cause (e.g. a 500), and no boundary peer is configured. The check must
    surface this as a 503 (transient, but not rate-limit-specific) rather than
    dispatching to a non-peer deployment.
    """
    from litellm.exceptions import ServiceUnavailableError
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    originating = _make_originating_mock("https://account-a.openai.azure.com/", "key-a")
    mock_router = _make_router_mock_with_cooldown(
        originating,
        cooldown_entries=[
            (
                "deployment-a-cooled",
                {
                    "exception_received": "boom",
                    "status_code": "500",
                    "timestamp": time.time(),
                    "cooldown_time": 60.0,
                },
            )
        ],
        routed_group_model_ids=["deployment-a-cooled", "deployment-b"],
    )

    check = EncryptedContentAffinityCheck(router=mock_router)
    encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id("deployment-a-cooled", "rs_test")
    healthy_only_b = [
        {
            "model_info": {"id": "deployment-b"},
            "model_name": "gpt-5.4",
            "litellm_params": {
                "api_base": "https://account-b.openai.azure.com/",
                "api_key": "key-b",
                "model": "azure/gpt-5.4",
            },
        }
    ]
    request_kwargs = {
        "input": [{"id": encoded_id, "type": "reasoning"}],
    }

    with pytest.raises(ServiceUnavailableError) as excinfo:
        await check.async_filter_deployments(
            model="gpt-5.4",
            healthy_deployments=healthy_only_b,
            messages=None,
            request_kwargs=request_kwargs,
        )

    # Public error message intentionally omits the originating model_id to
    # avoid an authenticated-caller probing oracle.
    assert "deployment-a-cooled" not in str(excinfo.value)
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_affinity_raises_rate_limit_with_retry_after_when_origin_cooled_for_429():
    """
    Originating deployment is in cooldown specifically because of a 429.
    The check must surface this as a 429 RateLimitError with a Retry-After
    header derived from the cooldown's remaining window, so OpenAI-compatible
    clients respect the backoff instead of giving up on a 503.
    """
    from litellm.exceptions import RateLimitError
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    originating = _make_originating_mock("https://account-a.openai.azure.com/", "key-a")
    cooldown_started = time.time() - 5.0
    mock_router = _make_router_mock_with_cooldown(
        originating,
        cooldown_entries=[
            (
                "deployment-a-cooled-429",
                {
                    "exception_received": "rate limited",
                    "status_code": "429",
                    "timestamp": cooldown_started,
                    "cooldown_time": 60.0,
                },
            )
        ],
        routed_group_model_ids=["deployment-a-cooled-429", "deployment-b"],
    )

    check = EncryptedContentAffinityCheck(router=mock_router)
    encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id("deployment-a-cooled-429", "rs_test")
    healthy_only_b = [
        {
            "model_info": {"id": "deployment-b"},
            "model_name": "gpt-5.4",
            "litellm_params": {
                "api_base": "https://account-b.openai.azure.com/",
                "api_key": "key-b",
                "model": "azure/gpt-5.4",
            },
        }
    ]
    request_kwargs = {
        "input": [{"id": encoded_id, "type": "reasoning"}],
    }

    with pytest.raises(RateLimitError) as excinfo:
        await check.async_filter_deployments(
            model="gpt-5.4",
            healthy_deployments=healthy_only_b,
            messages=None,
            request_kwargs=request_kwargs,
        )

    assert "deployment-a-cooled-429" not in str(excinfo.value)
    assert excinfo.value.status_code == 429
    retry_after = excinfo.value.response.headers.get("retry-after")
    assert retry_after is not None
    assert 1 <= int(retry_after) <= 60


@pytest.mark.asyncio
async def test_affinity_raises_service_unavailable_when_origin_filtered_without_cooldown_entry():
    """
    Originating deployment is configured but absent from healthy_deployments
    with no active cooldown entry. Surface as 503 (we cannot prove the cause
    was rate-limiting) rather than guessing 429.
    """
    from litellm.exceptions import ServiceUnavailableError
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    originating = _make_originating_mock("https://account-a.openai.azure.com/", "key-a")
    mock_router = _make_router_mock_with_cooldown(
        originating, cooldown_entries=[], routed_group_model_ids=["deployment-a-filtered", "deployment-b"]
    )

    check = EncryptedContentAffinityCheck(router=mock_router)
    encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id("deployment-a-filtered", "rs_test")
    healthy_only_b = [
        {
            "model_info": {"id": "deployment-b"},
            "model_name": "gpt-5.4",
            "litellm_params": {
                "api_base": "https://account-b.openai.azure.com/",
                "api_key": "key-b",
                "model": "azure/gpt-5.4",
            },
        }
    ]
    request_kwargs = {
        "input": [{"id": encoded_id, "type": "reasoning"}],
    }

    with pytest.raises(ServiceUnavailableError) as excinfo:
        await check.async_filter_deployments(
            model="gpt-5.4",
            healthy_deployments=healthy_only_b,
            messages=None,
            request_kwargs=request_kwargs,
        )

    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_affinity_strips_and_dispatches_when_origin_is_unknown_or_removed():
    """
    A removed deployment, or a forged/unknown affinity marker, resolves to no
    originating deployment. It is handled like a cross-group origin: the encrypted
    reasoning is stripped and the request dispatches with its readable history,
    rather than returning a distinguishable error. That uniform handling denies an
    authenticated caller a deployment-id existence oracle, an existing cross-group id
    and a nonexistent id both strip and proceed, so responses cannot be told apart.
    The membership lookup is skipped entirely when the origin is unknown.
    """
    from unittest.mock import MagicMock

    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    mock_router = MagicMock()
    mock_router.get_deployment.return_value = None

    check = EncryptedContentAffinityCheck(router=mock_router)
    wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id("gAAAAA-blob", "deployment-removed")
    routed_pool = [
        {
            "model_info": {"id": "deployment-b"},
            "model_name": "gpt-5.4",
            "litellm_params": {
                "api_base": "https://account-b.openai.azure.com/",
                "api_key": "key-b",
                "model": "azure/gpt-5.4",
            },
        }
    ]
    request_kwargs = {
        "litellm_metadata": {},
        "input": [
            {"role": "user", "content": "why is the sky blue?"},
            {
                "type": "reasoning",
                "encrypted_content": wrapped,
                "summary": [{"type": "summary_text", "text": "scattering"}],
            },
            {"role": "user", "content": "and sunsets?"},
        ],
    }

    result = await check.async_filter_deployments(
        model="gpt-5.4",
        healthy_deployments=routed_pool,
        messages=None,
        request_kwargs=request_kwargs,
    )

    assert result is routed_pool
    assert not any(isinstance(item, dict) and item.get("encrypted_content") for item in request_kwargs["input"])
    mock_router.get_candidate_model_ids_for_route.assert_not_called()


@pytest.mark.asyncio
async def test_affinity_does_not_raise_when_boundary_peer_available():
    """
    Even when the originating deployment is filtered out, if a peer on the
    same (api_base, api_key) is in healthy_deployments, the boundary-match
    path must succeed silently — no exception.
    """
    from unittest.mock import MagicMock

    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    originating = MagicMock()
    originating.litellm_params.model_dump.return_value = {
        "api_base": "https://account-a.openai.azure.com/",
        "api_key": "key-a",
    }
    mock_router = MagicMock()
    mock_router.get_deployment.return_value = originating

    check = EncryptedContentAffinityCheck(router=mock_router)
    encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id("deployment-a", "rs_test")
    peer = {
        "model_info": {"id": "deployment-a-peer"},
        "litellm_params": {
            "api_base": "https://account-a.openai.azure.com/",
            "api_key": "key-a",
            "model": "azure/gpt-5.4",
        },
    }
    request_kwargs = {
        "input": [{"id": encoded_id, "type": "reasoning"}],
    }

    result = await check.async_filter_deployments(
        model="gpt-5.4",
        healthy_deployments=[peer],
        messages=None,
        request_kwargs=request_kwargs,
    )

    assert result == [peer]
    assert request_kwargs.get("_encrypted_content_affinity_pinned") is True


@pytest.mark.asyncio
async def test_model_group_affinity_config_enables_encrypted_content_affinity():
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    model_group = "openai.gpt-5.1-codex"
    target_deployment = {
        "model_name": model_group,
        "litellm_params": {"model": "openai/gpt-5.1-codex"},
        "model_info": {"id": "deployment-b"},
    }
    healthy_deployments = [
        {
            "model_name": model_group,
            "litellm_params": {"model": "openai/gpt-5.1-codex"},
            "model_info": {"id": "deployment-a"},
        },
        target_deployment,
    ]
    encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id("deployment-b", "rs_test")
    request_kwargs = {
        "input": [{"type": "reasoning", "id": encoded_id}],
        "litellm_metadata": {},
    }
    check = EncryptedContentAffinityCheck(
        enable_global_affinity=False,
        model_group_affinity_config={
            model_group: ["encrypted_content_affinity"],
        },
    )

    filtered = await check.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs=request_kwargs,
    )

    assert filtered == [target_deployment]
    assert request_kwargs["litellm_metadata"]["encrypted_content_affinity_enabled"]
    assert request_kwargs.get("_encrypted_content_affinity_pinned") is True


@pytest.mark.asyncio
async def test_model_group_affinity_config_does_not_disable_global_encrypted_content_affinity():
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    model_group = "openai.gpt-5.1-codex"
    target_deployment = {
        "model_name": model_group,
        "litellm_params": {"model": "openai/gpt-5.1-codex"},
        "model_info": {"id": "deployment-b"},
    }
    healthy_deployments = [
        {
            "model_name": model_group,
            "litellm_params": {"model": "openai/gpt-5.1-codex"},
            "model_info": {"id": "deployment-a"},
        },
        target_deployment,
    ]
    encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id("deployment-b", "rs_test")
    request_kwargs = {
        "input": [{"type": "reasoning", "id": encoded_id}],
        "litellm_metadata": {},
    }
    check = EncryptedContentAffinityCheck(
        enable_global_affinity=True,
        model_group_affinity_config={
            model_group: ["deployment_affinity"],
        },
    )

    filtered = await check.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs=request_kwargs,
    )

    assert filtered == [target_deployment]
    assert request_kwargs["litellm_metadata"]["encrypted_content_affinity_enabled"]
    assert request_kwargs.get("_encrypted_content_affinity_pinned") is True


@pytest.mark.asyncio
async def test_model_group_encrypted_content_affinity_overrides_global_deployment_affinity():
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
        DeploymentAffinityCheck,
    )
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    model_group = "openai.gpt-5.1-codex"
    user_api_key_hash = "test-user-key"
    deployment_a = {
        "model_name": model_group,
        "litellm_params": {
            "model": "openai/gpt-5.1-codex",
            "api_key": "mock-api-key-a",
        },
        "model_info": {"id": "deployment-a"},
    }
    deployment_b = {
        "model_name": model_group,
        "litellm_params": {
            "model": "openai/gpt-5.1-codex",
            "api_key": "mock-api-key-b",
        },
        "model_info": {"id": "deployment-b"},
    }
    router = litellm.Router(
        model_list=[deployment_a, deployment_b],
        optional_pre_call_checks=["deployment_affinity"],
        model_group_affinity_config={
            model_group: ["encrypted_content_affinity"],
        },
        num_retries=0,
    )

    try:
        callbacks = router.optional_callbacks or []
        deployment_callback = next(cb for cb in callbacks if isinstance(cb, DeploymentAffinityCheck))
        encrypted_content_callback = next(cb for cb in callbacks if isinstance(cb, EncryptedContentAffinityCheck))
        assert callbacks.index(encrypted_content_callback) < callbacks.index(deployment_callback)
        assert encrypted_content_callback.enable_global_affinity is False

        cache_key = DeploymentAffinityCheck.get_affinity_cache_key(
            model_group=model_group,
            user_key=user_api_key_hash,
        )
        await deployment_callback.cache.async_set_cache(
            key=cache_key,
            value={"model_id": "deployment-a"},
            ttl=60,
        )
        encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id("deployment-b", "rs_test")
        request_kwargs = {
            "input": [
                {
                    "type": "reasoning",
                    "id": encoded_id,
                    "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
                }
            ],
            "metadata": {"user_api_key_hash": user_api_key_hash},
            "litellm_metadata": {},
        }

        after_deployment_affinity = await deployment_callback.async_filter_deployments(
            model=model_group,
            healthy_deployments=[deployment_a, deployment_b],
            messages=None,
            request_kwargs=request_kwargs,
        )
        assert after_deployment_affinity == [deployment_a, deployment_b]

        after_encrypted_content_affinity = await encrypted_content_callback.async_filter_deployments(
            model=model_group,
            healthy_deployments=after_deployment_affinity,
            messages=None,
            request_kwargs=request_kwargs,
        )

        assert after_encrypted_content_affinity == [deployment_b]
        assert request_kwargs.get("_encrypted_content_affinity_pinned") is True
    finally:
        router.discard()


class TestStripEncryptedReasoningFromInput:
    def test_keeps_summary_and_drops_encrypted_content_and_id(self):
        wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id("gAAAAA-blob", "deployment-a")
        encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id("deployment-a", "rs_1")
        request_input = [
            {"role": "user", "content": "first turn"},
            {
                "type": "reasoning",
                "id": encoded_id,
                "encrypted_content": wrapped,
                "summary": [{"type": "summary_text", "text": "thought about it"}],
            },
            {"type": "reasoning", "id": encoded_id, "encrypted_content": wrapped},
            {"type": "reasoning", "encrypted_content": wrapped, "summary": []},
            {"type": "message", "id": "msg_1", "role": "assistant", "content": "hi"},
            {"role": "user", "content": "second turn"},
        ]
        ResponsesAPIRequestUtils.strip_encrypted_reasoning_from_input(request_input)
        assert request_input == [
            {"role": "user", "content": "first turn"},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thought about it"}],
            },
            {"type": "message", "id": "msg_1", "role": "assistant", "content": "hi"},
            {"role": "user", "content": "second turn"},
        ]

    def test_keeps_string_form_summary_when_stripping(self):
        wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id("gAAAAA-blob", "deployment-a")
        request_input = [
            {"type": "reasoning", "encrypted_content": wrapped, "summary": "plain string thought"},
            {
                "type": "reasoning",
                "encrypted_content": wrapped,
                "content": [{"type": "output_text", "text": "in content"}],
            },
            {"type": "reasoning", "encrypted_content": wrapped, "summary": "", "content": []},
        ]
        ResponsesAPIRequestUtils.strip_encrypted_reasoning_from_input(request_input)
        assert request_input == [
            {"type": "reasoning", "summary": "plain string thought"},
            {"type": "reasoning", "content": [{"type": "output_text", "text": "in content"}]},
        ]

    def test_leaves_input_untouched_when_no_encrypted_reasoning(self):
        request_input = [
            {"role": "user", "content": "first turn"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "no blob here"}]},
            {"role": "user", "content": "second turn"},
        ]
        before = [dict(item) for item in request_input]
        ResponsesAPIRequestUtils.strip_encrypted_reasoning_from_input(request_input)
        assert request_input == before


def _cross_group_request_kwargs():
    wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id("gAAAAA-blob", "deployment-a")
    return {
        "litellm_metadata": {},
        "input": [
            {"role": "user", "content": "ZEBRA: why is the sky blue?"},
            {
                "type": "reasoning",
                "encrypted_content": wrapped,
                "summary": [{"type": "summary_text", "text": "scattering"}],
            },
            {"type": "message", "role": "assistant", "content": "Rayleigh scattering."},
            {"role": "user", "content": "KIWI: and sunsets?"},
        ],
    }


@pytest.mark.asyncio
async def test_affinity_strips_encrypted_reasoning_when_routed_to_another_model_group():
    """
    An auto-router tier change (or a model switch with no boundary peer): the
    routed pool holds no deployment of the origin's model group. The origin is
    healthy, so a 503 would be wrong; the follow-up dispatches to the routed
    pool with the origin's encrypted reasoning stripped and its summary kept.
    """
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    originating = _make_originating_mock(None, "key-a", model_name="gpt-reasoning-tier")
    mock_router = _make_router_mock_with_cooldown(
        originating, cooldown_entries=[], routed_group_model_ids=["deployment-b"]
    )
    check = EncryptedContentAffinityCheck(router=mock_router)
    routed_pool = [
        {
            "model_info": {"id": "deployment-b"},
            "model_name": "gpt-simple-tier",
            "litellm_params": {
                "api_base": "https://gateway.example/v1",
                "api_key": "key-b",
                "model": "openai/gpt-5-nano",
            },
        }
    ]
    request_kwargs = _cross_group_request_kwargs()
    original_input = request_kwargs["input"]

    result = await check.async_filter_deployments(
        model="gpt-simple-tier",
        healthy_deployments=routed_pool,
        messages=None,
        request_kwargs=request_kwargs,
    )

    assert result is routed_pool
    assert "_encrypted_content_affinity_pinned" not in request_kwargs
    assert request_kwargs["litellm_metadata"]["encrypted_content_affinity_enabled"] is True
    assert request_kwargs["input"] is original_input
    assert [item.get("type") or item["role"] for item in original_input] == [
        "user",
        "reasoning",
        "message",
        "user",
    ]
    assert original_input[1] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "scattering"}],
    }
    assert not any(isinstance(item, dict) and item.get("encrypted_content") for item in original_input)


@pytest.mark.asyncio
async def test_affinity_fails_fast_within_the_origins_own_group():
    """
    Negative class for the tier-change discriminator: the routed group IS the
    origin's group (a same-group cooldown, not a tier change), so even with a
    healthy non-origin sibling that cannot decrypt the content, the request
    still fails fast and the encrypted reasoning is left intact rather than
    stripped. Preserves the LIT-3051 cooldown contract.
    """
    from litellm.exceptions import ServiceUnavailableError
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    originating = _make_originating_mock(
        "https://account-a.openai.azure.com/", "key-a", model_name="gpt-reasoning-tier"
    )
    mock_router = _make_router_mock_with_cooldown(
        originating, cooldown_entries=[], routed_group_model_ids=["deployment-a", "deployment-a2"]
    )
    check = EncryptedContentAffinityCheck(router=mock_router)
    sibling_pool = [
        {
            "model_info": {"id": "deployment-a2"},
            "model_name": "gpt-reasoning-tier",
            "litellm_params": {
                "api_base": "https://account-a2.openai.azure.com/",
                "api_key": "key-a2",
                "model": "azure/gpt-5.4",
            },
        }
    ]
    request_kwargs = _cross_group_request_kwargs()

    with pytest.raises(ServiceUnavailableError):
        await check.async_filter_deployments(
            model="gpt-reasoning-tier",
            healthy_deployments=sibling_pool,
            messages=None,
            request_kwargs=request_kwargs,
        )

    assert request_kwargs["input"][1].get("encrypted_content")


@pytest.mark.asyncio
async def test_affinity_does_not_strip_when_group_is_spelled_differently_but_same_by_id():
    """
    The discriminator must key on deployment-id membership, not on the model-group
    name string. Here the origin's configured group is spelled ``openai/gpt-5.4-mini``
    while the routed group is the canonical ``gpt-5.4-mini``: same group, different
    spelling. A name compare (``originating.model_name != model``) would read this as
    a tier change and strip the reasoning it did not have to. Because the origin's id
    is a member of the routed group, this is a same-group cooldown instead: the request
    fails fast and the encrypted reasoning is left intact.
    """
    from litellm.exceptions import ServiceUnavailableError
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    originating = _make_originating_mock(None, "key-a", model_name="openai/gpt-5.4-mini")
    mock_router = _make_router_mock_with_cooldown(
        originating, cooldown_entries=[], routed_group_model_ids=["deployment-mini-a", "deployment-mini-b"]
    )
    check = EncryptedContentAffinityCheck(router=mock_router)
    wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id("gAAAAA-blob", "deployment-mini-a")
    sibling_pool = [
        {
            "model_info": {"id": "deployment-mini-b"},
            "model_name": "gpt-5.4-mini",
            "litellm_params": {
                "api_base": "https://gateway.example/v1",
                "api_key": "key-b",
                "model": "openai/gpt-5.4-mini",
            },
        }
    ]
    request_kwargs = {
        "litellm_metadata": {},
        "input": [
            {"role": "user", "content": "why is the sky blue?"},
            {
                "type": "reasoning",
                "encrypted_content": wrapped,
                "summary": [{"type": "summary_text", "text": "scattering"}],
            },
            {"role": "user", "content": "and sunsets?"},
        ],
    }

    with pytest.raises(ServiceUnavailableError):
        await check.async_filter_deployments(
            model="gpt-5.4-mini",
            healthy_deployments=sibling_pool,
            messages=None,
            request_kwargs=request_kwargs,
        )

    assert request_kwargs["input"][1].get("encrypted_content")


@pytest.mark.asyncio
async def test_affinity_honors_router_candidate_ids_for_team_and_pattern_routes():
    """
    The exact `model_name` index does not include team-public or pattern routes, so a
    same-group cooldown reached only through one of those would be misread as a tier change
    and stripped. The check asks the router for the candidate ids it resolves for the route
    (`get_candidate_model_ids_for_route`), which covers those paths, rather than the bare
    index. Here that set marks the origin as a candidate, so the request fails fast with its
    reasoning intact, and the routed group and team are passed through to the router.
    """
    from litellm.exceptions import ServiceUnavailableError
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    originating = _make_originating_mock(None, "key-a", model_name="model_name_teamA_uuid")
    mock_router = _make_router_mock_with_cooldown(
        originating, cooldown_entries=[], routed_group_model_ids=["deployment-team-a", "deployment-team-b"]
    )
    check = EncryptedContentAffinityCheck(router=mock_router)
    wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id("gAAAAA-blob", "deployment-team-a")
    sibling_pool = [
        {
            "model_info": {"id": "deployment-team-b"},
            "model_name": "team-public-model",
            "litellm_params": {
                "api_base": "https://gateway.example/v1",
                "api_key": "key-b",
                "model": "openai/gpt-5.4-mini",
            },
        }
    ]
    request_kwargs = {
        "litellm_metadata": {"user_api_key_team_id": "teamA"},
        "input": [
            {"role": "user", "content": "why is the sky blue?"},
            {
                "type": "reasoning",
                "encrypted_content": wrapped,
                "summary": [{"type": "summary_text", "text": "scattering"}],
            },
            {"role": "user", "content": "and sunsets?"},
        ],
    }

    with pytest.raises(ServiceUnavailableError):
        await check.async_filter_deployments(
            model="team-public-model",
            healthy_deployments=sibling_pool,
            messages=None,
            request_kwargs=request_kwargs,
        )

    assert request_kwargs["input"][1].get("encrypted_content")
    mock_router.get_candidate_model_ids_for_route.assert_called_once_with(model="team-public-model", team_id="teamA")
