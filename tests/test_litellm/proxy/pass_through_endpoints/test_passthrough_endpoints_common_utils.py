import json
import traceback
from unittest import mock
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient

from litellm.passthrough.utils import CommonUtils


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


def test_encode_bedrock_runtime_modelid_arn_partition_arns() -> None:
    endpoint = "model/arn:aws-cn:bedrock:cn-north-1:123456789012:application-inference-profile/r742sbn2zckd/converse"
    expected = "model/arn:aws-cn:bedrock:cn-north-1:123456789012:application-inference-profile%2Fr742sbn2zckd/converse"
    assert CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint) == expected

    endpoint = "model/arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:inference-profile/test-profile/invoke"
    expected = "model/arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:inference-profile%2Ftest-profile/invoke"
    assert CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint) == expected


from litellm.proxy.pass_through_endpoints.common_utils import (
    decrypt_pass_through_headers,
    encrypt_pass_through_endpoints,
    undecryptable_pass_through_header_names,
    reencrypt_general_settings_pass_through,
)

_ENC = "litellm_enc::"


@pytest.fixture
def salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-pass-through-tests")


def test_encrypt_pass_through_endpoints_encrypts_every_header_of_every_endpoint(salt_key):
    stored = encrypt_pass_through_endpoints(
        [
            {"path": "/a", "headers": {"x-plain": "value-a"}},
            {
                "path": "/b",
                "headers": {
                    "Authorization": "Bearer sk-literal",
                    "x-env": "Bearer os.environ/UPSTREAM_KEY",
                    "litellm_user_api_key": "x-my-key",
                    "x-number": 5,
                    "x-empty": "",
                },
            },
            {"path": "/c"},
        ]
    )

    assert stored[0]["headers"]["x-plain"].startswith(_ENC)
    headers_b = stored[1]["headers"]
    assert headers_b["Authorization"].startswith(_ENC)
    assert "sk-literal" not in headers_b["Authorization"]
    assert headers_b["x-env"].startswith(_ENC)
    assert headers_b["litellm_user_api_key"] == "x-my-key"
    assert headers_b["x-number"] == 5
    assert headers_b["x-empty"] == ""
    assert stored[2] == {"path": "/c"}
    assert decrypt_pass_through_headers(headers_b) == {
        "Authorization": "Bearer sk-literal",
        "x-env": "Bearer os.environ/UPSTREAM_KEY",
        "litellm_user_api_key": "x-my-key",
        "x-number": 5,
        "x-empty": "",
    }


def test_encrypt_pass_through_endpoints_keeps_existing_ciphertext_and_input(salt_key):
    endpoints = [{"path": "/a", "headers": {"Authorization": "Bearer sk-literal"}}]
    first = encrypt_pass_through_endpoints(endpoints)
    second = encrypt_pass_through_endpoints(first)

    assert second == first
    assert endpoints == [{"path": "/a", "headers": {"Authorization": "Bearer sk-literal"}}]
    assert encrypt_pass_through_endpoints(None) is None


def test_decrypt_pass_through_headers_passes_legacy_plaintext_through(salt_key):
    legacy = {"Authorization": "Bearer sk-legacy", "x-env": "os.environ/UPSTREAM_KEY"}

    assert decrypt_pass_through_headers(legacy) == legacy
    assert decrypt_pass_through_headers(None) is None


def test_decrypt_pass_through_headers_keeps_value_it_cannot_decrypt(salt_key, monkeypatch):
    stored = encrypt_pass_through_endpoints([{"path": "/a", "headers": {"Authorization": "Bearer sk-literal"}}])
    ciphertext = stored[0]["headers"]["Authorization"]
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-some-other-salt")

    assert decrypt_pass_through_headers({"Authorization": ciphertext}) == {"Authorization": ciphertext}


def test_reencrypt_general_settings_pass_through_moves_headers_to_new_key(salt_key, monkeypatch):
    stored = encrypt_pass_through_endpoints(
        [
            {"path": "/a", "headers": {"x-a": "value-a"}},
            {"path": "/b", "headers": {"Authorization": "Bearer sk-b"}},
        ]
    )
    general_settings = {
        "store_model_in_db": True,
        "pass_through_endpoints": [
            stored[0],
            {**stored[1], "headers": {**stored[1]["headers"], "x-legacy": "plain-b"}},
        ],
    }

    rotated = reencrypt_general_settings_pass_through(general_settings, "sk-new-master")

    assert rotated is not None
    assert rotated["store_model_in_db"] is True
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-new-master")
    assert decrypt_pass_through_headers(rotated["pass_through_endpoints"][0]["headers"]) == {"x-a": "value-a"}
    assert decrypt_pass_through_headers(rotated["pass_through_endpoints"][1]["headers"]) == {
        "Authorization": "Bearer sk-b",
        "x-legacy": "plain-b",
    }
    assert all(
        value.startswith(_ENC)
        for endpoint in rotated["pass_through_endpoints"]
        for value in endpoint["headers"].values()
    )
    assert reencrypt_general_settings_pass_through({"store_model_in_db": True}, "sk-new-master") is None
    assert reencrypt_general_settings_pass_through(None, "sk-new-master") is None


def test_undecryptable_pass_through_header_names(salt_key):
    [stored] = encrypt_pass_through_endpoints([{"path": "/a", "headers": {"x-a": "plain-a"}}])
    rotated = reencrypt_general_settings_pass_through({"pass_through_endpoints": [stored]}, "sk-next-key")

    assert undecryptable_pass_through_header_names(stored["headers"]) == frozenset()
    assert undecryptable_pass_through_header_names({"x-legacy": "plain", "x-n": 5}) == frozenset()
    assert undecryptable_pass_through_header_names(None) == frozenset()
    assert undecryptable_pass_through_header_names(rotated["pass_through_endpoints"][0]["headers"]) == {"x-a"}
    assert undecryptable_pass_through_header_names(
        {"x-ok": stored["headers"]["x-a"], "x-bad": _ENC + "garbage", "x-plain": "p"}
    ) == {"x-bad"}


def test_decrypt_pass_through_headers_keeps_a_bare_marker_literal(salt_key):
    assert decrypt_pass_through_headers({"x-tag": _ENC}) == {"x-tag": _ENC}
    assert undecryptable_pass_through_header_names({"x-tag": _ENC}) == frozenset()
