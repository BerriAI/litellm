import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from botocore.credentials import Credentials

from litellm.types.router import GenericLiteLLMParams


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


def test_claude_platform_strips_auth_params_from_request_body():
    """
    Regression: workspace_id is consumed by validate_environment (sent as the
    anthropic-workspace-id header) and aws_* params by sign_request, but
    transform_request used to forward them into the Messages API body, which
    rejects unknown fields: "workspace_id: Extra inputs are not permitted".
    """
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

    assert "workspace_id" not in request_body
    assert "aws_region_name" not in request_body
    assert request_body["max_tokens"] == 10
    # sign_request still needs the aws_* params — the original dict must not
    # be mutated by the body transformation.
    assert optional_params["aws_region_name"] == "us-west-2"
    assert optional_params["workspace_id"] == "wrkspc_test"


def test_claude_platform_messages_strips_auth_params_from_request_body():
    """
    Same regression as above for the native /v1/messages path.
    """
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

    assert "workspace_id" not in request_body
    assert "aws_region_name" not in request_body
    assert request_body["max_tokens"] == 10
    assert input_params["aws_region_name"] == "us-west-2"
    assert input_params["workspace_id"] == "wrkspc_test"


def test_claude_platform_strips_unsupported_context_management_param(caplog):
    """
    Regression: context_management is a valid first-party Anthropic Messages
    API field, but the Claude Platform on AWS (aws-external-anthropic)
    endpoint does not support it and rejects it with
    "context_management: Extra inputs are not permitted". It must be dropped
    from the body, and — unlike the silent auth-param strip — the drop is
    logged at WARNING because it reflects user intent that won't be applied.
    """
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
    # original dict is not mutated
    assert "context_management" in optional_params
    # the drop is surfaced to the user
    assert any(
        "context_management" in record.message and record.levelno == logging.WARNING
        for record in caplog.records
    )


def test_claude_platform_messages_strips_unsupported_context_management_param():
    """
    Same context_management strip on the native /v1/messages path.
    """
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


def test_claude_platform_unsupported_override_allows_context_management():
    """
    Operators can pass claude_platform_unsupported_params=[] in litellm_params
    to stop filtering context_management once the AWS endpoint supports it.
    """
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    config = BedrockClaudePlatformConfig()
    optional_params = {
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        "max_tokens": 10,
    }

    request_body = config.transform_request(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        optional_params=optional_params,
        litellm_params={"claude_platform_unsupported_params": []},
        headers={},
    )

    assert request_body["max_tokens"] == 10
    assert "context_management" in request_body


def test_claude_platform_unsupported_override_ignores_invalid_type():
    """
    If claude_platform_unsupported_params is set to a non-collection type
    (e.g. a string), the override is ignored, defaults apply, and a warning
    naming the param is logged so the operator sees it had no effect.
    """
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
    """
    Regression: validate_anthropic_messages_environment used to compute
    anthropic-beta from unfiltered optional_params, while
    transform_anthropic_messages_request strips context_management from the
    body. The resulting request claimed a beta the body did not use, which
    can trip strict gateway validation. Headers must be derived from the
    same filtered view as the body.
    """
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
    """
    When the operator opts back into context_management via
    claude_platform_unsupported_params=[], the body keeps the field and the
    header must still advertise the matching beta.
    """
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
    """
    Messages-path: operators can pass claude_platform_unsupported_params=[]
    in litellm_params to allow context_management through.
    """
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
        litellm_params=GenericLiteLLMParams.model_validate(
            {"claude_platform_unsupported_params": []}
        ),
        headers={},
    )

    assert "context_management" in request_body
    assert request_body["max_tokens"] == 10


def test_chat_completion_claude_platform_sigv4_body_has_no_auth_params():
    """
    End-to-end (mocked transport): a config-driven SigV4 call with
    workspace_id + aws_region_name must not leak either param into the wire
    body. Uses SigV4 (no api_key) since that is how proxy configs pass
    aws_region_name.
    """
    import litellm

    requests = []

    def mock_post(self, url, data=None, headers=None, **kwargs):
        requests.append(_capture_request(url=url, headers=headers or {}, data=data))
        return _anthropic_response(url)

    mock_credentials = Credentials("test-key", "test-secret", "test-token")

    with (
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", mock_post),
        patch(
            "litellm.llms.bedrock.base_aws_llm.BaseAWSLLM.get_credentials",
            return_value=mock_credentials,
        ),
    ):
        response = litellm.completion(
            model="bedrock/claude_platform/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
            aws_region_name="us-west-2",
            workspace_id="wrkspc_test",
        )

    assert response.choices[0].message.content == "ok"
    assert len(requests) == 1
    body = requests[0]["body"]
    assert "workspace_id" not in body
    assert "aws_region_name" not in body
    assert requests[0]["headers"]["anthropic-workspace-id"] == "wrkspc_test"


def test_sigv4_no_duplicate_content_type_when_caller_sets_lowercase():
    """
    Regression: get_anthropic_headers() supplies "content-type" (lowercase).
    _sign_request() used to prepend "Content-Type" (uppercase), leaving both
    keys in the dict.  botocore joins them into "application/json, application/json"
    in the canonical string, while the wire request sends only one value → 401.

    Fix: prepend with lowercase "content-type" so **headers overwrites it when
    the caller already set it.
    """
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
