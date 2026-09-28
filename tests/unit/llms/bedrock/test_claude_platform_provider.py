import json
from collections.abc import Callable, Mapping
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
from botocore.credentials import Credentials
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.router import GenericLiteLLMParams

WORKSPACE_ALIASES: Final = ("workspace_id", "aws_workspace_id", "anthropic_workspace_id", "anthropic-workspace-id")


class ClaudePlatformMessagesBody(BaseModel):
    """Messages API fields per https://docs.anthropic.com/en/api/messages (2026-09) minus context_management, which
    the AWS endpoint rejects with 400"""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[dict]
    max_tokens: int
    system: str | list[dict] | None = None
    metadata: dict | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    tools: list[dict] | None = None
    tool_choice: dict | None = None
    thinking: dict | None = None
    service_tier: str | None = None
    mcp_servers: list[dict] | None = None
    output_format: dict | None = None
    container: str | dict | None = None


def _gateway_reject(url: str, message: str) -> httpx.Response:
    return httpx.Response(
        status_code=400,
        json={"type": "error", "error": {"type": "invalid_request_error", "message": message}},
        request=httpx.Request("POST", url),
    )


def _fake_claude_platform_gateway(url: str, headers: Mapping[str, str], data: bytes | str | None) -> httpx.Response:
    if "anthropic-workspace-id" not in headers:
        return _gateway_reject(url, "missing anthropic-workspace-id header")
    if "x-api-key" not in headers and not headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256 "):
        return _gateway_reject(url, "missing x-api-key or SigV4 Authorization")
    try:
        ClaudePlatformMessagesBody.model_validate_json(data or "{}")
    except ValidationError as exc:
        return _gateway_reject(url, "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))
    return _anthropic_response(url)


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


def _capture_request(url: str, headers: Mapping[str, str], data: bytes | str | None) -> dict:
    raw_body = data.decode("utf-8") if isinstance(data, bytes) else data or "{}"
    return {
        "path": httpx.URL(url).path,
        "headers": httpx.Headers(dict(headers)),
        "body": json.loads(raw_body),
    }


GatewayResponder = Callable[[str, Mapping[str, str], bytes], httpx.Response]


def _gateway_transport(requests: list[dict], respond: GatewayResponder) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(_capture_request(url=str(request.url), headers=request.headers, data=request.content))
        return respond(str(request.url), request.headers, request.content)

    return httpx.MockTransport(handle)


def _sync_gateway_client(
    requests: list[dict],
    respond: GatewayResponder = _fake_claude_platform_gateway,
) -> HTTPHandler:
    return HTTPHandler(client=httpx.Client(transport=_gateway_transport(requests, respond)))


def _async_gateway_client(requests: list[dict]) -> AsyncHTTPHandler:
    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=_gateway_transport(requests, _fake_claude_platform_gateway))
    return handler


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

    model, provider, _, _ = litellm.get_llm_provider(model="bedrock/claude_platform/claude-sonnet-4-6")

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
    assert mock_sign_request.call_args.kwargs["service_name"] == "aws-external-anthropic"
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
        litellm_params=GenericLiteLLMParams(),
        headers=headers,
    )

    assert headers["anthropic-workspace-id"] == "wrkspc_test"
    assert headers["x-api-key"] == "fake-platform-key"
    assert request_body == {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }


@pytest.mark.asyncio
async def test_anthropic_messages_bedrock_claude_platform_forwards_anthropic_beta_verbatim():
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
            await litellm.anthropic_messages(
                model="bedrock/claude_platform/claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                mcp_servers=[{"type": "url", "url": "https://mcp.example.com/mcp", "name": "example"}],
                api_base="https://aws-external-anthropic.us-west-2.api.aws",
                api_key="fake-platform-key",
                workspace_id="wrkspc_test",
                extra_headers={"anthropic-beta": "prompt-caching-scope-2026-01-05,mcp-client-2025-11-20"},
            )
    finally:
        await litellm.close_litellm_async_clients()

    assert len(requests) == 1
    assert requests[0]["headers"]["anthropic-beta"] == "mcp-client-2025-11-20,prompt-caching-scope-2026-01-05"
    assert requests[0]["body"]["mcp_servers"] == [
        {"type": "url", "url": "https://mcp.example.com/mcp", "name": "example"}
    ]


def test_claude_platform_strips_auth_params_from_request_body():
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    config = BedrockClaudePlatformConfig()
    optional_params = {
        "workspace_id": "wrkspc_test",
        "aws_region_name": "us-west-2",
        "max_tokens": 10,
    }

    request_body = config.transform_request(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert request_body == EXPECTED_CHAT_BODY
    assert optional_params == {"workspace_id": "wrkspc_test", "aws_region_name": "us-west-2", "max_tokens": 10}


def test_claude_platform_messages_strips_auth_params_from_request_body():
    import litellm
    from litellm.types.utils import LlmProviders

    config = litellm.ProviderConfigManager.get_provider_anthropic_messages_config(
        model="claude_platform/claude-sonnet-4-6",
        provider=LlmProviders.BEDROCK,
    )
    assert config is not None

    input_params = {
        "workspace_id": "wrkspc_test",
        "aws_region_name": "us-west-2",
        "max_tokens": 10,
    }
    request_body = config.transform_anthropic_messages_request(
        model="claude_platform/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params=input_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request_body == {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }
    assert input_params == {"workspace_id": "wrkspc_test", "aws_region_name": "us-west-2", "max_tokens": 10}


def test_claude_platform_strips_unsupported_context_management_param(caplog):
    import logging

    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    config = BedrockClaudePlatformConfig()
    optional_params = {
        "workspace_id": "wrkspc_test",
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        "max_tokens": 10,
    }

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        request_body = config.transform_request(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

    assert "context_management" not in request_body
    assert request_body["max_tokens"] == 10
    assert "context_management" in optional_params
    assert any(
        "context_management" in record.message and record.levelno == logging.WARNING for record in caplog.records
    )


def test_claude_platform_messages_strips_unsupported_context_management_param():
    import litellm
    from litellm.types.utils import LlmProviders

    config = litellm.ProviderConfigManager.get_provider_anthropic_messages_config(
        model="claude_platform/claude-sonnet-4-6",
        provider=LlmProviders.BEDROCK,
    )
    assert config is not None

    input_params = {
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        "max_tokens": 10,
    }
    request_body = config.transform_anthropic_messages_request(
        model="claude_platform/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params=input_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "context_management" not in request_body
    assert request_body["max_tokens"] == 10
    assert "context_management" in input_params


@pytest.mark.parametrize("override_in", ["litellm_params", "optional_params"])
def test_claude_platform_unsupported_override_allows_context_management(override_in):
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    config = BedrockClaudePlatformConfig()
    override = {"claude_platform_unsupported_params": []}
    context_management = {"edits": [{"type": "clear_tool_uses_20250919"}]}

    request_body = config.transform_request(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={
            "context_management": context_management,
            "max_tokens": 10,
            **(override if override_in == "optional_params" else {}),
        },
        litellm_params=override if override_in == "litellm_params" else {},
        headers={},
    )

    assert request_body == {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "max_tokens": 10,
        "context_management": context_management,
    }


def test_claude_platform_unsupported_override_ignores_invalid_type():
    from litellm.llms.bedrock.claude_platform import common_utils
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    config = BedrockClaudePlatformConfig()
    with patch.object(common_utils.verbose_logger, "warning") as mock_warning:
        request_body = config.transform_request(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            optional_params={
                "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
                "max_tokens": 10,
            },
            litellm_params={"claude_platform_unsupported_params": "not_a_list"},
            headers={},
        )

    assert "context_management" not in request_body
    assert request_body["max_tokens"] == 10
    warned = [call.args[0] for call in mock_warning.call_args_list]
    assert any("claude_platform_unsupported_params" in message for message in warned)


def test_claude_platform_messages_does_not_advertise_beta_for_stripped_context_management():
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
        optional_params={
            "max_tokens": 10,
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        },
        litellm_params={"workspace_id": "wrkspc_test"},
    )

    assert "context-management-2025-06-27" not in headers.get("anthropic-beta", "")


def test_claude_platform_messages_override_keeps_beta_for_context_management():
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
        optional_params={
            "max_tokens": 10,
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        },
        litellm_params={
            "workspace_id": "wrkspc_test",
            "claude_platform_unsupported_params": [],
        },
    )

    assert "context-management-2025-06-27" in headers.get("anthropic-beta", "")


def test_claude_platform_messages_unsupported_override_allows_context_management():
    import litellm
    from litellm.types.utils import LlmProviders

    config = litellm.ProviderConfigManager.get_provider_anthropic_messages_config(
        model="claude_platform/claude-sonnet-4-6",
        provider=LlmProviders.BEDROCK,
    )
    assert config is not None

    request_body = config.transform_anthropic_messages_request(
        model="claude_platform/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params={
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
            "max_tokens": 10,
        },
        litellm_params=GenericLiteLLMParams.model_validate({"claude_platform_unsupported_params": []}),
        headers={},
    )

    assert "context_management" in request_body
    assert request_body["max_tokens"] == 10


SIGV4_KWARGS: Final = {
    "aws_region_name": "us-west-2",
    "aws_access_key_id": "AKIATEST",
    "aws_secret_access_key": "test-secret",
    "aws_session_token": "test-token",
}
API_KEY_KWARGS: Final = {"api_key": "fake-platform-key", "aws_region_name": "us-west-2"}
EXPECTED_CHAT_BODY: Final = {
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    "max_tokens": 10,
}
EXPECTED_NATIVE_BODY: Final = {
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 10,
    "stream": False,
}


def _assert_gateway_accepted(request: dict, auth_kwargs: dict) -> None:
    assert request["path"] == "/v1/messages"
    assert request["headers"]["anthropic-workspace-id"] == "wrkspc_test"
    if "api_key" in auth_kwargs:
        assert request["headers"]["x-api-key"] == auth_kwargs["api_key"]
        assert "Authorization" not in request["headers"]
    else:
        assert request["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIATEST/")
        assert "/us-west-2/aws-external-anthropic/aws4_request" in request["headers"]["Authorization"]


@pytest.mark.parametrize("auth_kwargs", [API_KEY_KWARGS, SIGV4_KWARGS], ids=["api_key", "sigv4"])
@pytest.mark.parametrize("workspace_alias", WORKSPACE_ALIASES)
def test_chat_completion_claude_platform_sends_exact_body_through_strict_gateway(auth_kwargs, workspace_alias):
    import litellm

    requests: list[dict] = []

    response = litellm.completion(
        model="bedrock/claude_platform/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        client=_sync_gateway_client(requests),
        **{workspace_alias: "wrkspc_test"},
        **auth_kwargs,
    )

    assert response.choices[0].message.content == "ok"
    assert len(requests) == 1, requests
    assert requests[0]["body"] == EXPECTED_CHAT_BODY
    _assert_gateway_accepted(requests[0], auth_kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_kwargs", [API_KEY_KWARGS, SIGV4_KWARGS], ids=["api_key", "sigv4"])
@pytest.mark.parametrize("workspace_alias", WORKSPACE_ALIASES)
async def test_anthropic_messages_claude_platform_sends_exact_body_through_strict_gateway(auth_kwargs, workspace_alias):
    import litellm

    requests: list[dict] = []

    response = await litellm.anthropic_messages(
        model="bedrock/claude_platform/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        client=_async_gateway_client(requests),
        **{workspace_alias: "wrkspc_test"},
        **auth_kwargs,
    )

    assert response["content"][0]["text"] == "ok"
    assert len(requests) == 1, requests
    assert requests[0]["body"] == EXPECTED_NATIVE_BODY
    _assert_gateway_accepted(requests[0], auth_kwargs)


def test_chat_completion_claude_platform_drops_context_management_and_gateway_accepts():
    import litellm

    requests: list[dict] = []

    litellm.completion(
        model="bedrock/claude_platform/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        workspace_id="wrkspc_test",
        context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
        client=_sync_gateway_client(requests),
        **API_KEY_KWARGS,
    )

    assert requests[0]["body"] == EXPECTED_CHAT_BODY


def test_chat_completion_claude_platform_override_kwarg_is_honoured_and_not_sent():
    import litellm

    requests: list[dict] = []
    context_management = {"edits": [{"type": "clear_tool_uses_20250919"}]}

    litellm.completion(
        model="bedrock/claude_platform/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        workspace_id="wrkspc_test",
        context_management=context_management,
        claude_platform_unsupported_params=[],
        client=_sync_gateway_client(requests, respond=lambda url, headers, data: _anthropic_response(url)),
        **API_KEY_KWARGS,
    )

    assert requests[0]["body"] == {**EXPECTED_CHAT_BODY, "context_management": context_management}


def test_fake_claude_platform_gateway_rejects_leaked_internal_fields():
    leaked = json.dumps({**EXPECTED_NATIVE_BODY, "workspace_id": "wrkspc_test", "aws_region_name": "us-west-2"})
    response = _fake_claude_platform_gateway(
        url="https://aws-external-anthropic.us-west-2.api.aws/v1/messages",
        headers={"anthropic-workspace-id": "wrkspc_test", "x-api-key": "k"},
        data=leaked,
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "workspace_id: Extra inputs are not permitted; aws_region_name: Extra inputs are not permitted"
    )


def test_sigv4_no_duplicate_content_type_when_caller_sets_lowercase():
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    llm = BaseAWSLLM()
    mock_credentials = Credentials("key", "secret", "token")
    mock_sigv4 = MagicMock()
    captured: list[dict] = []

    def fake_aws_request(method, url, data, headers):
        captured.append(dict(headers))
        req = MagicMock()
        req.headers = {"Authorization": "AWS4-HMAC-SHA256 Credential=test"}
        req.body = data.encode() if isinstance(data, str) else data
        return req

    with (
        patch("botocore.auth.SigV4Auth", return_value=mock_sigv4),
        patch("botocore.awsrequest.AWSRequest", side_effect=fake_aws_request),
        patch.object(llm, "get_credentials", return_value=mock_credentials),
        patch.object(llm, "_get_aws_region_name", return_value="us-east-1"),
    ):
        llm._sign_request(
            service_name="aws-external-anthropic",
            headers={"content-type": "application/json"},
            optional_params={"aws_region_name": "us-east-1"},
            request_data={
                "model": "claude-sonnet-4-6",
                "messages": [],
                "max_tokens": 10,
            },
            api_base="https://aws-external-anthropic.us-east-1.api.aws/v1/messages",
        )

    signed = captured[0]
    ct_keys = [k for k in signed if k.lower() == "content-type"]
    assert ct_keys == ["content-type"], (
        f"Expected exactly one 'content-type' key, got {ct_keys}. "
        "Duplicate keys produce 'application/json, application/json' in the "
        "SigV4 canonical string and cause a 401."
    )
