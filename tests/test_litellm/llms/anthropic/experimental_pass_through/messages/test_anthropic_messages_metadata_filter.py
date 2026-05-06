import os
import sys

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams


def _transform(metadata):
    config = AnthropicMessagesConfig()
    optional_params = {"max_tokens": 100, "metadata": metadata}
    return config.transform_anthropic_messages_request(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


def test_metadata_strips_non_user_id_keys():
    """
    LiteLLM injects routing/budget tags into metadata. The Anthropic Messages
    API (and Vertex AI / Azure AI mirrors) only accept `user_id` inside
    `metadata` and 400 on any other key. Verify the unified `/v1/messages` flow
    strips non-`user_id` keys before forwarding upstream.
    """
    request = _transform({"user_id": "claudio", "tags": ["team:foo"], "ip": "1.2.3.4"})

    assert request["metadata"] == {"user_id": "claudio"}


def test_metadata_dropped_when_only_disallowed_keys():
    request = _transform({"tags": ["team:foo"]})

    assert "metadata" not in request


def test_metadata_dropped_when_user_id_is_none():
    request = _transform({"user_id": None, "tags": ["team:foo"]})

    assert "metadata" not in request


def test_metadata_passes_through_when_only_user_id():
    request = _transform({"user_id": "claudio"})

    assert request["metadata"] == {"user_id": "claudio"}


def test_no_metadata_field_is_no_op():
    config = AnthropicMessagesConfig()
    optional_params = {"max_tokens": 100}
    request = config.transform_anthropic_messages_request(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "metadata" not in request
