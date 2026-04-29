"""
Tests for OCI Cohere chat fixes:
1. Dynamic maxTokens default via model cost map lookup
2. Filtering empty messages from Cohere chat history
"""

import os
import sys

import pytest

# Adds the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.oci.chat.transformation import OCIChatConfig
from litellm.types.llms.oci import OCIVendors
from litellm.constants import DEFAULT_MAX_TOKENS


TEST_COMPARTMENT_ID = "ocid1.compartment.oc1..xxxxxx"
BASE_OCI_PARAMS = {
    "oci_region": "us-ashburn-1",
    "oci_user": "ocid1.user.oc1..xxxxxxEXAMPLExxxxxx",
    "oci_fingerprint": "4f:29:77:cc:b1:3e:55:ab:61:2a:de:47:f1:38:4c:90",
    "oci_tenancy": "ocid1.tenancy.oc1..xxxxxxEXAMPLExxxxxx",
    "oci_compartment_id": TEST_COMPARTMENT_ID,
    "oci_key": "<private_key.pem as string>",
}


class TestOCICohereMaxTokens:
    """Tests for dynamic maxTokens default lookup."""

    def test_max_tokens_uses_model_cost_map(self):
        """When a model is in the cost map, maxTokens should use its max_output_tokens."""
        config = OCIChatConfig()

        # oci/cohere.command-a-03-2025 has max_output_tokens=4000 in the cost map
        params = config._get_optional_params(
            OCIVendors.COHERE, BASE_OCI_PARAMS, model="oci/cohere.command-a-03-2025"
        )

        assert params["maxTokens"] == 4000

    def test_max_tokens_falls_back_to_default(self):
        """When a model is NOT in the cost map, maxTokens should fall back to DEFAULT_MAX_TOKENS."""
        config = OCIChatConfig()

        params = config._get_optional_params(
            OCIVendors.COHERE, BASE_OCI_PARAMS, model="oci/cohere.nonexistent-model"
        )

        assert params["maxTokens"] == DEFAULT_MAX_TOKENS

    def test_max_tokens_falls_back_when_model_is_none(self):
        """When model is None, maxTokens should fall back to DEFAULT_MAX_TOKENS."""
        config = OCIChatConfig()

        params = config._get_optional_params(
            OCIVendors.COHERE, BASE_OCI_PARAMS, model=None
        )

        assert params["maxTokens"] == DEFAULT_MAX_TOKENS

    def test_user_provided_max_tokens_overrides_default(self):
        """User-provided max_tokens should override the dynamic default."""
        config = OCIChatConfig()

        params_with_override = {**BASE_OCI_PARAMS, "max_tokens": 1024}
        params = config._get_optional_params(
            OCIVendors.COHERE, params_with_override, model="oci/cohere.command-a-03-2025"
        )

        assert params["maxTokens"] == 1024

    def test_get_max_tokens_for_model_with_oci_prefix(self):
        """_get_max_tokens_for_model should work with the oci/ prefix."""
        result = OCIChatConfig._get_max_tokens_for_model("oci/cohere.command-a-03-2025")
        assert result == 4000

    def test_get_max_tokens_for_model_without_oci_prefix(self):
        """_get_max_tokens_for_model should also work without the oci/ prefix.

        litellm strips the provider prefix before calling transform_request,
        so the method must retry with the oci/ prefix to find the cost map entry.
        """
        result = OCIChatConfig._get_max_tokens_for_model("cohere.command-a-03-2025")
        assert result == 4000

    def test_get_max_tokens_for_model_unknown_model(self):
        """_get_max_tokens_for_model should return DEFAULT_MAX_TOKENS for an unknown model."""
        result = OCIChatConfig._get_max_tokens_for_model("oci/unknown-model-xyz")
        assert result == DEFAULT_MAX_TOKENS

    def test_get_max_tokens_for_model_none(self):
        """_get_max_tokens_for_model should return DEFAULT_MAX_TOKENS when model is None."""
        result = OCIChatConfig._get_max_tokens_for_model(None)
        assert result == DEFAULT_MAX_TOKENS


class TestOCICohereEmptyMessages:
    """Tests for filtering empty messages from Cohere chat history.

    Note: adapt_messages_to_cohere_standard() processes messages[:-1]
    (all messages except the last one, which is the current user message).
    The returned CohereMessage objects are Pydantic models with .role and
    .message attributes.
    """

    def test_empty_assistant_message_is_filtered(self):
        """An assistant message with empty content should be excluded from chat history."""
        config = OCIChatConfig()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "How are you?"},
        ]

        # messages[:-1] = ["Hello" (user), "" (assistant)]
        # Empty assistant is filtered, so only "Hello" remains
        history = config.adapt_messages_to_cohere_standard(messages)

        assert len(history) == 1
        assert history[0].role == "USER"
        assert history[0].message == "Hello"

    def test_empty_system_message_is_filtered(self):
        """A system message with empty content should be excluded from chat history."""
        config = OCIChatConfig()
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Last message"},
        ]

        # messages[:-1] = ["" (system), "Hello" (user)]
        # Empty system is filtered, so only "Hello" remains
        history = config.adapt_messages_to_cohere_standard(messages)

        assert len(history) == 1
        assert history[0].role == "USER"
        assert history[0].message == "Hello"

    def test_none_content_message_is_filtered(self):
        """A message with None content should be excluded from chat history."""
        config = OCIChatConfig()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Last message"},
        ]

        # messages[:-1] = ["Hello" (user), None (assistant)]
        # None assistant is filtered, so only "Hello" remains
        history = config.adapt_messages_to_cohere_standard(messages)

        assert len(history) == 1
        assert history[0].role == "USER"
        assert history[0].message == "Hello"

    def test_non_empty_messages_are_preserved(self):
        """Non-empty messages should be included in chat history normally."""
        config = OCIChatConfig()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Last message"},
        ]

        # messages[:-1] = ["Hello" (user), "Hi there!" (assistant)]
        history = config.adapt_messages_to_cohere_standard(messages)

        assert len(history) == 2
        assert history[0].role == "USER"
        assert history[0].message == "Hello"
        assert history[1].role == "CHATBOT"
        assert history[1].message == "Hi there!"

    def test_mixed_empty_and_nonempty_messages(self):
        """Only empty messages should be filtered; non-empty ones kept."""
        config = OCIChatConfig()
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "A real answer"},
            {"role": "user", "content": "Last message"},
        ]

        # messages[:-1] excludes "Last message"
        # Empty system and empty assistant are filtered
        # Remaining: "First question" (user), "Second question" (user), "A real answer" (chatbot)
        history = config.adapt_messages_to_cohere_standard(messages)

        assert len(history) == 3
        assert history[0].role == "USER"
        assert history[0].message == "First question"
        assert history[1].role == "USER"
        assert history[1].message == "Second question"
        assert history[2].role == "CHATBOT"
        assert history[2].message == "A real answer"
