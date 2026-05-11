import json
from unittest.mock import patch

import httpx
import pytest


def _anthropic_response(url: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        request=httpx.Request("POST", url),
    )


def _capture_request(url: str, headers: dict, data: bytes | str | None) -> dict:
    raw_body = data.decode("utf-8") if isinstance(data, bytes) else data or "{}"
    return {
        "path": httpx.URL(url).path,
        "headers": headers,
        "body": json.loads(raw_body),
    }


def test_claude_platform_builds_default_messages_url_from_region():
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    config = BedrockClaudePlatformConfig()

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="claude-sonnet-4-6",
            optional_params={"aws_region_name": "us-west-2"},
            litellm_params={},
        )
        == "https://aws-external-anthropic.us-west-2.api.aws/v1/messages"
    )


def test_claude_platform_ignores_standard_anthropic_base_url(monkeypatch):
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.example")
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://api.anthropic-api.example")

    config = BedrockClaudePlatformConfig()

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="claude-sonnet-4-6",
            optional_params={"aws_region_name": "us-west-2"},
            litellm_params={},
        )
        == "https://aws-external-anthropic.us-west-2.api.aws/v1/messages"
    )


def test_claude_platform_uses_bedrock_subroute():
    import litellm
    from litellm.llms.bedrock.common_utils import BedrockModelInfo

    model, provider, _, _ = litellm.get_llm_provider(
        model="bedrock/claude_platform/claude-sonnet-4-6"
    )

    assert provider == "bedrock"
    assert model == "claude_platform/claude-sonnet-4-6"
    assert BedrockModelInfo.get_bedrock_route(model) == "claude_platform"
    assert BedrockModelInfo.get_claude_platform_model(model) == "claude-sonnet-4-6"


def test_claude_platform_requires_workspace_header():
    from litellm import AuthenticationError
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    config = BedrockClaudePlatformConfig()

    with pytest.raises(AuthenticationError) as exc_info:
        config.validate_environment(
            api_key="fake-platform-key",
            headers={},
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            optional_params={},
            litellm_params={},
        )

    assert "workspace" in str(exc_info.value).lower()


def test_claude_platform_api_key_auth_sets_workspace_and_key_headers():
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    config = BedrockClaudePlatformConfig()
    headers = config.validate_environment(
        api_key="fake-platform-key",
        headers={"anthropic-beta": "skills-2025-10-02"},
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"workspace_id": "wrkspc_test"},
        litellm_params={},
    )

    assert headers["x-api-key"] == "fake-platform-key"
    assert headers["anthropic-workspace-id"] == "wrkspc_test"
    assert headers["anthropic-beta"] == "skills-2025-10-02"


def test_claude_platform_does_not_use_standard_anthropic_api_key(monkeypatch):
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "standard-anthropic-key")

    config = BedrockClaudePlatformConfig()
    headers = config.validate_environment(
        api_key=None,
        headers={},
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"workspace_id": "wrkspc_test"},
        litellm_params={},
    )

    assert "x-api-key" not in headers


def test_claude_platform_sigv4_signs_transformed_request_body():
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    config = BedrockClaudePlatformConfig()
    request_body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }

    with patch.object(
        config,
        "_sign_request",
        return_value=({"Authorization": "signed"}, json.dumps(request_body).encode()),
    ) as mock_sign_request:
        headers, signed_body = config.sign_request(
            headers={"anthropic-workspace-id": "wrkspc_test"},
            optional_params={"aws_region_name": "us-west-2"},
            request_data=request_body,
            api_base="https://aws-external-anthropic.us-west-2.api.aws/v1/messages",
            api_key=None,
            model="claude-sonnet-4-6",
        )

    assert signed_body == json.dumps(request_body).encode()
    assert headers["Authorization"] == "signed"
    mock_sign_request.assert_called_once()
    assert (
        mock_sign_request.call_args.kwargs["service_name"] == "aws-external-anthropic"
    )
    assert mock_sign_request.call_args.kwargs["request_data"] == request_body


def test_claude_platform_standard_anthropic_api_key_does_not_skip_sigv4(monkeypatch):
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "standard-anthropic-key")
    config = BedrockClaudePlatformConfig()
    request_body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }

    with patch.object(
        config,
        "_sign_request",
        return_value=({"Authorization": "signed"}, json.dumps(request_body).encode()),
    ) as mock_sign_request:
        headers, signed_body = config.sign_request(
            headers={"anthropic-workspace-id": "wrkspc_test"},
            optional_params={"aws_region_name": "us-west-2"},
            request_data=request_body,
            api_base="https://aws-external-anthropic.us-west-2.api.aws/v1/messages",
            api_key=None,
            model="claude-sonnet-4-6",
        )

    assert signed_body == json.dumps(request_body).encode()
    assert headers["Authorization"] == "signed"
    mock_sign_request.assert_called_once()


def test_bedrock_claude_platform_messages_config_round_trips_native_body():
    import litellm
    from litellm.types.utils import LlmProviders

    config = litellm.ProviderConfigManager.get_provider_anthropic_messages_config(
        model="claude_platform/claude-sonnet-4-6",
        provider=LlmProviders.BEDROCK,
    )

    assert config is not None
    headers, _ = config.validate_anthropic_messages_environment(
        api_key="fake-platform-key",
        headers={},
        model="claude_platform/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"max_tokens": 10},
        litellm_params={"workspace_id": "wrkspc_test"},
    )
    request_body = config.transform_anthropic_messages_request(
        model="claude_platform/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 10},
        litellm_params={},
        headers=headers,
    )

    assert headers["anthropic-workspace-id"] == "wrkspc_test"
    assert headers["x-api-key"] == "fake-platform-key"
    assert request_body == {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }


def test_chat_completion_routes_bedrock_claude_platform_to_messages_api():
    import litellm

    requests = []

    def mock_post(self, url, data=None, headers=None, **kwargs):
        requests.append(_capture_request(url=url, headers=headers or {}, data=data))
        return _anthropic_response(url)

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", mock_post):
        response = litellm.completion(
            model="bedrock/claude_platform/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
            api_base="https://aws-external-anthropic.us-west-2.api.aws",
            api_key="fake-platform-key",
            workspace_id="wrkspc_test",
        )

    assert response.choices[0].message.content == "ok"
    assert len(requests) == 1
    assert requests[0]["path"] == "/v1/messages"
    assert requests[0]["headers"]["x-api-key"] == "fake-platform-key"
    assert requests[0]["headers"]["anthropic-workspace-id"] == "wrkspc_test"
    assert requests[0]["body"]["model"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_anthropic_messages_routes_bedrock_claude_platform_to_messages_api():
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
                model="bedrock/claude_platform/claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                api_base="https://aws-external-anthropic.us-west-2.api.aws",
                api_key="fake-platform-key",
                workspace_id="wrkspc_test",
            )
    finally:
        await litellm.close_litellm_async_clients()

    assert response["content"][0]["text"] == "ok"
    assert len(requests) == 1
    assert requests[0]["path"] == "/v1/messages"
    assert requests[0]["headers"]["x-api-key"] == "fake-platform-key"
    assert requests[0]["headers"]["anthropic-workspace-id"] == "wrkspc_test"
    assert requests[0]["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert requests[0]["body"]["max_tokens"] == 10
    assert requests[0]["body"]["model"] == "claude-sonnet-4-6"
