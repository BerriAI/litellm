import json
import os
import sys
import traceback
from unittest import mock
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient

from litellm.passthrough.utils import CommonUtils

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import Mock

from litellm.proxy.pass_through_endpoints.common_utils import get_litellm_virtual_key


@pytest.mark.asyncio
async def test_get_litellm_virtual_key():
    """
    Test that the get_litellm_virtual_key function correctly handles the API key authentication
    """
    # Test with x-litellm-api-key
    mock_request = Mock()
    mock_request.headers = {"x-litellm-api-key": "test-key-123"}
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer test-key-123"

    # Test with Authorization header
    mock_request.headers = {"Authorization": "Bearer auth-key-456"}
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer auth-key-456"

    # Test with both headers (x-litellm-api-key should take precedence)
    mock_request.headers = {
        "x-litellm-api-key": "test-key-123",
        "Authorization": "Bearer auth-key-456",
    }
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer test-key-123"


def test_encode_bedrock_runtime_modelid_arn():
    # Test application-inference-profile ARN
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789123:application-inference-profile/r742sbn2zckd/converse"
    expected = "model/arn:aws:bedrock:us-east-1:123456789123:application-inference-profile%2Fr742sbn2zckd/converse"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected

    # Test inference-profile ARN
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:inference-profile/test-profile/invoke"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:inference-profile%2Ftest-profile/invoke"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected

    # Test foundation-model ARN
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3/converse"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:foundation-model%2Fanthropic.claude-3/converse"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected

    # Test custom-model ARN (2 slashes)
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model.fine-tuned/abc123/invoke"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:custom-model%2Fmy-model.fine-tuned%2Fabc123/invoke"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected

    # Test provisioned-model ARN
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:provisioned-model/test-model/converse"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:provisioned-model%2Ftest-model/converse"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected


def test_encode_bedrock_runtime_modelid_arn_no_arn():
    # Test regular model ID (no ARN)
    endpoint = "model/anthropic.claude-3-sonnet-20240229-v1:0/converse"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == endpoint


def test_encode_bedrock_runtime_modelid_arn_edge_cases():
    # Test multiple ARN types (should only encode first match)
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/test1/converse"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile%2Ftest1/converse"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected

    # Test ARN with special characters in resource ID
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/test-profile.v1/invoke"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile%2Ftest-profile.v1/invoke"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected


class TestOpenAIWireCompatibleScope:
    """`is_openai_wire_compatible_route` is the single scope predicate shared by
    the pass-through success handler and the OpenAI route helpers."""

    def test_provider_widens_scope_beyond_the_hostname_allow_list(self):
        from litellm.proxy.pass_through_endpoints.common_utils import (
            is_openai_wire_compatible_route,
        )

        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        assert is_openai_wire_compatible_route(url, "fireworks_ai") is True
        assert (
            is_openai_wire_compatible_route(
                "https://api.groq.com/openai/v1/chat/completions", "groq"
            )
            is True
        )

    def test_shared_azure_domains_keep_the_path_marker_guard(self):
        """A provider label must not let Speech / Vision / Language on the
        shared Azure Cognitive Services domains be costed as OpenAI."""
        from litellm.proxy.pass_through_endpoints.common_utils import (
            is_openai_wire_compatible_route,
        )

        speech = (
            "https://my-resource.cognitiveservices.azure.com"
            "/speechtotext/v3.1/transcriptions"
        )
        assert is_openai_wire_compatible_route(speech, "azure") is False
        assert is_openai_wire_compatible_route(speech, "openai") is False
        # ... while a real Azure OpenAI path still matches.
        assert (
            is_openai_wire_compatible_route(
                "https://my-resource.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions",
                "azure",
            )
            is True
        )

    def test_unknown_provider_and_host_is_out_of_scope(self):
        from litellm.proxy.pass_through_endpoints.common_utils import (
            is_openai_wire_compatible_route,
        )

        assert (
            is_openai_wire_compatible_route(
                "https://api.example.com/v1/chat/completions", "some_random_provider"
            )
            is False
        )
        assert is_openai_wire_compatible_route(None, None) is False


class TestFireworksModelIdHelpers:
    def test_is_fireworks_model_id(self):
        from litellm.proxy.pass_through_endpoints.common_utils import (
            is_fireworks_model_id,
        )

        assert is_fireworks_model_id("accounts/fireworks/models/deepseek-v3") is True
        assert (
            is_fireworks_model_id("fireworks_ai/accounts/fireworks/models/deepseek-v3")
            is True
        )
        assert is_fireworks_model_id("gpt-4o") is False
        assert is_fireworks_model_id("azure/my-deployment") is False
        assert is_fireworks_model_id(None) is False

    def test_resolve_provider_prefers_the_configured_one(self):
        from litellm.proxy.pass_through_endpoints.common_utils import (
            resolve_openai_passthrough_provider,
        )

        assert (
            resolve_openai_passthrough_provider(
                model="accounts/fireworks/models/deepseek-v3",
                custom_llm_provider="openai",
            )
            == "openai"
        )

    def test_resolve_provider_infers_fireworks_when_unset(self):
        """A generic pass-through declares no provider; defaulting to "openai"
        made the price lookup raise and the call record $0."""
        from litellm.proxy.pass_through_endpoints.common_utils import (
            resolve_openai_passthrough_provider,
        )

        assert (
            resolve_openai_passthrough_provider(
                model="accounts/fireworks/models/deepseek-v3"
            )
            == "fireworks_ai"
        )
        assert (
            resolve_openai_passthrough_provider(
                url_route="https://api.fireworks.ai/inference/v1/chat/completions"
            )
            == "fireworks_ai"
        )
        assert resolve_openai_passthrough_provider(model="gpt-4o") == "openai"
