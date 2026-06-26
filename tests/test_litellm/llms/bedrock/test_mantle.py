"""
Unit tests for the Bedrock Mantle (Claude Mythos Preview) integration.

Tests cover route detection, URL construction, config dispatch for both
the /chat/completions and /messages endpoints, and project (workspace)
association via `aws_bedrock_project_id`.
"""

import json
from unittest.mock import patch

import httpx
import pytest

from litellm.llms.bedrock.common_utils import BedrockModelInfo, get_bedrock_chat_config
from litellm.llms.bedrock.chat.mantle.transformation import AmazonMantleConfig
from litellm.llms.bedrock.messages.mantle_transformation import (
    AmazonMantleMessagesConfig,
)


def _anthropic_response(url: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "anthropic.claude-mythos-preview",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        request=httpx.Request("POST", url),
    )


def _capture_request(url: str, headers: dict, data) -> dict:
    raw_body = data.decode("utf-8") if isinstance(data, bytes) else data or "{}"
    return {
        "path": httpx.URL(url).path,
        "headers": headers,
        "body": json.loads(raw_body),
    }


def test_get_bedrock_route_mantle():
    assert (
        BedrockModelInfo.get_bedrock_route("mantle/anthropic.claude-mythos-preview")
        == "mantle"
    )


def test_get_bedrock_route_mantle_does_not_match_other_routes():
    assert (
        BedrockModelInfo.get_bedrock_route("anthropic.claude-3-sonnet-20240229-v1:0")
        != "mantle"
    )
    assert (
        BedrockModelInfo.get_bedrock_route("converse/anthropic.claude-3-sonnet")
        != "mantle"
    )


def test_explicit_mantle_route_flag():
    assert (
        BedrockModelInfo._explicit_mantle_route(
            "mantle/anthropic.claude-mythos-preview"
        )
        is True
    )
    assert BedrockModelInfo._explicit_mantle_route("anthropic.claude-3-sonnet") is False
    assert (
        BedrockModelInfo._explicit_mantle_route("converse/anthropic.claude-3-sonnet")
        is False
    )


def test_mantle_url_construction():
    config = AmazonMantleConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-east-1"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"


def test_mantle_url_construction_different_region():
    config = AmazonMantleConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-west-2"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-west-2.api.aws/anthropic/v1/messages"


def test_get_bedrock_chat_config_returns_mantle_config():
    config = get_bedrock_chat_config("mantle/anthropic.claude-mythos-preview")
    assert isinstance(config, AmazonMantleConfig)


def test_get_bedrock_provider_config_for_messages_api_mantle():
    config = BedrockModelInfo.get_bedrock_provider_config_for_messages_api(
        "mantle/anthropic.claude-mythos-preview"
    )
    assert isinstance(config, AmazonMantleMessagesConfig)


def test_mantle_messages_url_construction():
    config = AmazonMantleMessagesConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-east-1"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"


_VPC_ENDPOINT = "https://vpce-0a1b2c3d.bedrock-mantle.us-gov-west-1.vpce.amazonaws.com"


def test_mantle_chat_url_honors_api_base_host():
    config = AmazonMantleConfig()
    url = config.get_complete_url(
        api_base=_VPC_ENDPOINT,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-gov-west-1"},
        litellm_params={},
    )
    assert url == f"{_VPC_ENDPOINT}/anthropic/v1/messages"


def test_mantle_chat_url_honors_api_base_full_path_without_duplication():
    config = AmazonMantleConfig()
    full = f"{_VPC_ENDPOINT}/anthropic/v1/messages"
    url = config.get_complete_url(
        api_base=full,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-gov-west-1"},
        litellm_params={},
    )
    assert url == full


def test_mantle_messages_url_honors_api_base_host():
    config = AmazonMantleMessagesConfig()
    url = config.get_complete_url(
        api_base=_VPC_ENDPOINT,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-gov-west-1"},
        litellm_params={},
    )
    assert url == f"{_VPC_ENDPOINT}/anthropic/v1/messages"
    assert "api.aws" not in url


def test_mantle_messages_url_honors_api_base_with_trailing_slash():
    config = AmazonMantleMessagesConfig()
    url = config.get_complete_url(
        api_base=f"{_VPC_ENDPOINT}/",
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-gov-west-1"},
        litellm_params={},
    )
    assert url == f"{_VPC_ENDPOINT}/anthropic/v1/messages"


def test_mantle_messages_url_honors_aws_bedrock_runtime_endpoint():
    config = AmazonMantleMessagesConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={
            "aws_region_name": "us-gov-west-1",
            "aws_bedrock_runtime_endpoint": _VPC_ENDPOINT,
        },
        litellm_params={},
    )
    assert url == f"{_VPC_ENDPOINT}/anthropic/v1/messages"


def test_mantle_transform_request_strips_prefix_and_adds_model():
    config = AmazonMantleConfig()
    request = config.transform_request(
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "anthropic.claude-mythos-preview"
    assert "mantle/" not in request["model"]


def test_mantle_validate_environment_sets_workspace_header():
    config = AmazonMantleConfig()
    headers = config.validate_environment(
        headers={},
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={"aws_bedrock_project_id": "proj_abc123def456"},
    )
    assert headers["anthropic-workspace"] == "proj_abc123def456"


def test_mantle_validate_environment_without_project_id():
    config = AmazonMantleConfig()
    headers = config.validate_environment(
        headers={},
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={"aws_bedrock_project_id": None},
    )
    assert "anthropic-workspace" not in headers


def test_mantle_messages_validate_environment_sets_workspace_header():
    config = AmazonMantleMessagesConfig()
    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={"aws_bedrock_project_id": "proj_abc123def456"},
        api_base="https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
    )
    assert headers["anthropic-workspace"] == "proj_abc123def456"
    assert api_base == "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"


def test_mantle_messages_validate_environment_without_project_id():
    config = AmazonMantleMessagesConfig()
    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
    )
    assert "anthropic-workspace" not in headers


def test_mantle_completion_sends_workspace_header_and_clean_body():
    import litellm

    requests = []

    def mock_post(self, url, data=None, headers=None, **kwargs):
        requests.append(_capture_request(url=url, headers=headers or {}, data=data))
        return _anthropic_response(url)

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", mock_post):
        response = litellm.completion(
            model="bedrock/mantle/anthropic.claude-mythos-preview",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
            aws_bedrock_project_id="proj_abc123def456",
            aws_access_key_id="fake-key",
            aws_secret_access_key="fake-secret",
            aws_region_name="us-east-1",
        )

    assert response.choices[0].message.content == "ok"
    assert len(requests) == 1
    assert requests[0]["path"] == "/anthropic/v1/messages"
    assert requests[0]["headers"]["anthropic-workspace"] == "proj_abc123def456"
    assert "aws_bedrock_project_id" not in requests[0]["body"]


@pytest.mark.asyncio
async def test_mantle_anthropic_messages_sends_workspace_header_and_clean_body():
    import litellm

    requests = []

    async def mock_post(self, url, data=None, headers=None, **kwargs):
        requests.append(_capture_request(url=url, headers=headers or {}, data=data))
        return _anthropic_response(url)

    try:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new=mock_post,
        ):
            response = await litellm.anthropic_messages(
                model="bedrock/mantle/anthropic.claude-mythos-preview",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                aws_bedrock_project_id="proj_abc123def456",
                aws_access_key_id="fake-key",
                aws_secret_access_key="fake-secret",
                aws_region_name="us-east-1",
            )
    finally:
        await litellm.close_litellm_async_clients()

    assert response["content"][0]["text"] == "ok"
    assert len(requests) == 1
    assert requests[0]["path"] == "/anthropic/v1/messages"
    assert requests[0]["headers"]["anthropic-workspace"] == "proj_abc123def456"
    assert "aws_bedrock_project_id" not in requests[0]["body"]


@pytest.mark.asyncio
async def test_mantle_anthropic_messages_routes_to_vpc_api_base():
    import litellm

    urls = []

    async def mock_post(self, url, data=None, headers=None, **kwargs):
        urls.append(str(url))
        return _anthropic_response(str(url))

    try:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new=mock_post,
        ):
            await litellm.anthropic_messages(
                model="bedrock/mantle/anthropic.claude-mythos-preview",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                api_base=_VPC_ENDPOINT,
                aws_access_key_id="fake-key",
                aws_secret_access_key="fake-secret",
                aws_region_name="us-gov-west-1",
            )
    finally:
        await litellm.close_litellm_async_clients()

    assert len(urls) == 1
    assert urls[0] == f"{_VPC_ENDPOINT}/anthropic/v1/messages"
    assert "api.aws" not in urls[0]


def test_mantle_chat_config_signs_with_bedrock_mantle_service():
    """The bedrock/mantle/ chat route must SigV4-sign with service 'bedrock-mantle'."""
    config = AmazonMantleConfig()
    signed_service = None

    def capture_sign(self, *, service_name, **kwargs):
        nonlocal signed_service
        signed_service = service_name
        return {"Authorization": "AWS4-HMAC-SHA256 ..."}, b"{}"

    with patch(
        "litellm.llms.bedrock.base_aws_llm.BaseAWSLLM._sign_request",
        capture_sign,
    ):
        config.sign_request(
            headers={"Content-Type": "application/json"},
            optional_params={
                "aws_access_key_id": "fake",
                "aws_secret_access_key": "fake",
                "aws_region_name": "us-east-1",
            },
            request_data={"model": "anthropic.claude-opus-4-8", "messages": []},
            api_base="https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
        )

    assert signed_service == "bedrock-mantle"


def test_mantle_messages_config_signs_with_bedrock_mantle_service():
    """The bedrock/mantle/ messages route must SigV4-sign with service 'bedrock-mantle'."""
    config = AmazonMantleMessagesConfig()
    signed_service = None

    def capture_sign(self, *, service_name, **kwargs):
        nonlocal signed_service
        signed_service = service_name
        return {"Authorization": "AWS4-HMAC-SHA256 ..."}, b"{}"

    with patch(
        "litellm.llms.bedrock.base_aws_llm.BaseAWSLLM._sign_request",
        capture_sign,
    ):
        config.sign_request(
            headers={"Content-Type": "application/json"},
            optional_params={
                "aws_access_key_id": "fake",
                "aws_secret_access_key": "fake",
                "aws_region_name": "us-east-1",
            },
            request_data={"model": "anthropic.claude-opus-4-8", "messages": []},
            api_base="https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
        )

    assert signed_service == "bedrock-mantle"


def test_mantle_configs_sign_real_sigv4_scope_is_bedrock_mantle(monkeypatch):
    """Both bedrock/mantle/ configs must produce a *real* SigV4 signature whose
    credential scope names service 'bedrock-mantle' (not 'bedrock').

    Unlike the two tests above this does NOT mock _sign_request: it drives the
    real botocore SigV4Auth with fake static creds (offline, no network), so a
    regression in the signed service name would change the credential scope and
    fail here. Cred setup mirrors test_no_bearer_signs_with_sigv4 in
    tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_transformation.py.
    """
    # No Bearer token -> _sign_request must take the SigV4 path, not Bearer auth.
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    for config in (AmazonMantleConfig(), AmazonMantleMessagesConfig()):
        headers, _ = config.sign_request(
            headers={"Content-Type": "application/json"},
            optional_params={
                "aws_access_key_id": "AKIAEXAMPLE",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_region_name": "us-east-1",
            },
            request_data={"model": "anthropic.claude-opus-4-8", "messages": []},
            api_base="https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
        )

        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "/us-east-1/bedrock-mantle/aws4_request" in headers["Authorization"]
