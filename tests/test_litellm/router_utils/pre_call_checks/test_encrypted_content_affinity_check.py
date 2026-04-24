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

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

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
        encoded = ResponsesAPIRequestUtils._build_encrypted_item_id(
            model_id, original_item_id
        )
        assert encoded.startswith("encitem_")
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(encoded)
        assert decoded is not None
        assert decoded["model_id"] == model_id
        assert decoded["item_id"] == original_item_id

    def test_decode_without_padding(self):
        """Decoding must succeed even if base64 padding (=) was stripped in transit."""
        model_id = "gpt-5.1-codex-openai-2"
        original_item_id = "rs_0efb96cb222403210069a01d5d52588196a9dc394ffdb89d00"
        encoded = ResponsesAPIRequestUtils._build_encrypted_item_id(
            model_id, original_item_id
        )
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
        encoded = ResponsesAPIRequestUtils._build_encrypted_item_id(
            model_id, original_item_id
        )
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
        result = (
            ResponsesAPIRequestUtils._update_encrypted_content_item_ids_in_response(
                response, model_id
            )
        )
        # Plain message item untouched
        assert result["output"][0]["id"] == "msg_abc"
        # Reasoning item with encrypted_content gets encoded
        encoded_id = result["output"][1]["id"]
        assert encoded_id.startswith("encitem_")
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(encoded_id)
        assert decoded["model_id"] == model_id
        assert decoded["item_id"] == "rs_xyz"

    def test_no_op_when_model_id_is_none(self):
        response = {
            "output": [
                {"id": "rs_xyz", "type": "reasoning", "encrypted_content": "secret"}
            ]
        }
        result = (
            ResponsesAPIRequestUtils._update_encrypted_content_item_ids_in_response(
                response, None
            )
        )
        assert result["output"][0]["id"] == "rs_xyz"


class TestEncryptedContentWrapping:
    def test_wrap_and_unwrap_encrypted_content(self):
        """Test wrapping encrypted_content with model_id metadata."""
        model_id = "deployment-1"
        original_content = "gAAAAABpnW_yEYmSNEyOG_original_encrypted_data"
        wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
            original_content, model_id
        )
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
        ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(
            plain_content
        )
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
        result = (
            ResponsesAPIRequestUtils._update_encrypted_content_item_ids_in_response(
                response, model_id
            )
        )
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
        encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id(
            model_id, original_id
        )

        request_input = [
            {"type": "message", "id": "msg_abc123", "role": "assistant"},
            {"type": "reasoning", "id": encoded_id, "encrypted_content": "secret"},
        ]
        restored = (
            ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input(
                request_input
            )
        )
        assert restored[0]["id"] == "msg_abc123"
        assert restored[1]["id"] == original_id

    def test_unwraps_encrypted_content(self):
        """Test that wrapped encrypted_content is unwrapped before forwarding."""
        model_id = "deployment-1"
        original_content = "gAAAAABpnW_yEYmSNEyOG_original"
        wrapped_content = (
            ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
                original_content, model_id
            )
        )

        request_input = [
            {"type": "reasoning", "encrypted_content": wrapped_content},
        ]
        restored = (
            ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input(
                request_input
            )
        )
        assert restored[0]["encrypted_content"] == original_content

    def test_no_op_for_plain_string_input(self):
        result = ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input(
            "Hello world"
        )
        assert result == "Hello world"

    def test_no_op_for_unencoded_ids(self):
        request_input = [{"type": "message", "id": "msg_plain"}]
        result = ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input(
            request_input
        )
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
                "content": [
                    {"type": "output_text", "text": "Hello!", "annotations": []}
                ],
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
        assert encoded_item_id.startswith(
            "encitem_"
        ), f"Expected output item ID to be rewritten to encitem_... but got {encoded_item_id!r}"

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

        assert (
            second_model_id == first_model_id
        ), f"Expected affinity to route to {first_model_id}, but got {second_model_id}"


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
        assert encoded_item_id.startswith(
            "encitem_"
        ), f"Expected encitem_... but got {encoded_item_id!r}"

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
        assert wrapped_content.startswith(
            "litellm_enc:"
        ), f"Expected wrapped content but got {wrapped_content[:50]}..."

        # Verify we can extract model_id from wrapped content
        (
            extracted_model_id,
            _,
        ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(
            wrapped_content
        )
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

        assert (
            second_model_id == first_model_id
        ), f"Expected affinity to route to {first_model_id}, but got {second_model_id}"


def test_encrypted_content_wrapping_preserves_original_content():
    """
    Test that wrapping and unwrapping encrypted_content preserves the original content.
    This is critical for streaming responses where content must round-trip correctly.
    """
    model_id = "test-deployment-1"
    original_encrypted_content = (
        "gAAAAABpnW_yEYmSNEyOG_streaming_test_content_with_special_chars==+/"
    )

    wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
        original_encrypted_content, model_id
    )

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

    wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
        original_content, model_id
    )

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

    assert (
        request_kwargs["litellm_metadata"]["encrypted_content_affinity_enabled"] is True
    )
    assert request_kwargs["litellm_metadata"]["model_info"] == {"id": "dep-1"}


def test_encrypted_content_wrapping_empty_string():
    """
    Test that empty encrypted_content is handled gracefully.
    """
    model_id = "test-deployment"
    original_content = ""

    wrapped = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
        original_content, model_id
    )

    assert wrapped.startswith("litellm_enc:")

    (
        extracted_model_id,
        unwrapped,
    ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(wrapped)

    assert extracted_model_id == model_id
    assert unwrapped == original_content
