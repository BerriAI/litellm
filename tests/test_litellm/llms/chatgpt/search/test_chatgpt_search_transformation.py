import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.exceptions import AuthenticationError, BadRequestError
from litellm.llms.chatgpt.common_utils import GetAccessTokenError
from litellm.llms.chatgpt.search.transformation import (
    ChatGPTSearchPassthroughConfig,
    ChatGPTSearchRequest,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.pass_through.guardrail_translation.handler import LlmPassthroughRouteHandler
from litellm.passthrough.main import allm_passthrough_route
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class StubAuthenticator:
    def get_access_token(self) -> str:
        return "oauth-token"

    def get_account_id(self) -> str:
        return "account-123"

    def get_api_base(self) -> str:
        return "https://default.chatgpt.test/backend-api/codex"


def test_search_request_preserves_unknown_fields() -> None:
    request = ChatGPTSearchRequest.model_validate(
        {
            "id": "session-123",
            "model": " gpt-5.6-sol ",
            "commands": {"search_query": [{"q": "LiteLLM"}]},
            "future": {"preserved": True},
        }
    )

    assert request.model_dump(exclude_unset=True) == {
        "id": "session-123",
        "model": "gpt-5.6-sol",
        "commands": {"search_query": [{"q": "LiteLLM"}]},
        "future": {"preserved": True},
    }


def test_search_passthrough_config_builds_url_and_oauth_headers() -> None:
    config = ChatGPTSearchPassthroughConfig(authenticator=StubAuthenticator())
    url, api_base = config.get_complete_url(
        api_base="https://custom.chatgpt.test/backend-api/codex/",
        api_key=None,
        model="gpt-5.6-sol",
        endpoint="alpha/search",
        request_query_params=None,
        litellm_params={},
    )
    headers = config.validate_environment(
        headers={},
        model="gpt-5.6-sol",
        messages=[],
        optional_params={},
        litellm_params={"litellm_session_id": "generated-session"},
    )
    headers, body = config.sign_request(
        headers=headers,
        litellm_params={
            "proxy_server_request": {
                "headers": {
                    "authorization": "Bearer untrusted-token",
                    "originator": "codex_vscode",
                    "x-codex-turn-metadata": '{"turn_id":"turn-123"}',
                    "x-untrusted": "not-forwarded",
                }
            }
        },
        request_data={"id": "session-123"},
        api_base=str(url),
        model="gpt-5.6-sol",
    )

    assert str(url) == "https://custom.chatgpt.test/backend-api/codex/alpha/search"
    assert api_base == "https://custom.chatgpt.test/backend-api/codex/"
    assert headers["Authorization"] == "Bearer oauth-token"
    assert headers["ChatGPT-Account-Id"] == "account-123"
    assert headers["session_id"] == "session-123"
    assert headers["accept"] == "application/json"
    assert headers["originator"] == "codex_vscode"
    assert json.loads(headers["x-codex-turn-metadata"]) == {"turn_id": "turn-123"}
    assert "x-untrusted" not in headers
    assert body is None


def test_search_passthrough_config_rejects_other_endpoints() -> None:
    config = ChatGPTSearchPassthroughConfig(authenticator=StubAuthenticator())

    with pytest.raises(ValueError, match="Unsupported ChatGPT passthrough endpoint"):
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gpt-5.6-sol",
            endpoint="responses",
            request_query_params=None,
            litellm_params={},
        )


def test_search_passthrough_config_maps_device_login_failure() -> None:
    class FailingAuthenticator(StubAuthenticator):
        def get_access_token(self) -> str:
            raise GetAccessTokenError(message="device login required", status_code=401)

    config = ChatGPTSearchPassthroughConfig(authenticator=FailingAuthenticator())

    with pytest.raises(AuthenticationError, match="device login required"):
        config.validate_environment(
            headers={},
            model="gpt-5.6-sol",
            messages=[],
            optional_params={},
            litellm_params={},
        )


def test_search_passthrough_config_drops_malformed_proxy_headers() -> None:
    config = ChatGPTSearchPassthroughConfig(authenticator=StubAuthenticator())

    headers, body = config.sign_request(
        headers={"Authorization": "Bearer oauth-token"},
        litellm_params={"proxy_server_request": {"headers": {"originator": ["codex_vscode"]}}},
        request_data=None,
        api_base="https://chatgpt.test/backend-api/codex/alpha/search",
    )

    assert headers == {"Authorization": "Bearer oauth-token"}
    assert "originator" not in headers
    assert body is None


def test_search_passthrough_config_maps_transport_errors() -> None:
    config = ChatGPTSearchPassthroughConfig(authenticator=StubAuthenticator())

    error = config.get_error_class(
        error_message="upstream unavailable",
        status_code=503,
        headers={"retry-after": "5"},
    )

    assert isinstance(error, OpenAIError)
    assert error.status_code == 503
    assert error.message == "upstream unavailable"
    assert error.headers == {"retry-after": "5"}


def test_search_passthrough_config_model_info_contract() -> None:
    config = ChatGPTSearchPassthroughConfig(authenticator=StubAuthenticator())

    assert config.get_api_base("https://custom.chatgpt.test") == "https://custom.chatgpt.test"
    assert config.get_base_model("gpt-5.6-sol") == "gpt-5.6-sol"
    assert config.get_models() == []


def test_provider_manager_returns_chatgpt_search_passthrough_config() -> None:
    config = ProviderConfigManager.get_provider_passthrough_config(
        model="gpt-5.6-sol",
        provider=LlmProviders.CHATGPT,
    )

    assert isinstance(config, ChatGPTSearchPassthroughConfig)


@pytest.mark.asyncio
async def test_search_uses_shared_passthrough_transport() -> None:
    config = ChatGPTSearchPassthroughConfig(authenticator=StubAuthenticator())
    client = AsyncHTTPHandler()
    requests: list[httpx.Request] = []

    async def send(request: httpx.Request, stream: bool = False) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            json={"results": []},
            request=request,
        )

    logging_obj = MagicMock()
    try:
        with (
            patch(
                "litellm.passthrough.main.get_llm_provider",
                return_value=("gpt-5.6-sol", "chatgpt", None, None),
            ),
            patch(
                "litellm.litellm_core_utils.llm_response_utils.get_api_base.get_llm_provider",
                return_value=("gpt-5.6-sol", "chatgpt", None, None),
            ),
            patch.object(client.client, "send", new=send),
        ):
            response = await allm_passthrough_route(
                method="POST",
                endpoint="alpha/search",
                model="chatgpt/gpt-5.6-sol",
                json={
                    "id": "session-123",
                    "model": "sol",
                    "commands": {"search_query": [{"q": "LiteLLM"}]},
                    "future": {"preserved": True},
                },
                client=client,
                provider_config=config,
                litellm_logging_obj=logging_obj,
                proxy_server_request={
                    "headers": {
                        "authorization": "Bearer proxy-token",
                        "originator": "codex_vscode",
                    }
                },
                required_custom_llm_provider="chatgpt",
            )
    finally:
        await client.close()

    assert response.status_code == 200
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://default.chatgpt.test/backend-api/codex/alpha/search"
    assert request.headers["authorization"] == "Bearer oauth-token"
    assert request.headers["originator"] == "codex_vscode"
    assert json.loads(request.content) == {
        "id": "session-123",
        "model": "gpt-5.6-sol",
        "commands": {"search_query": [{"q": "LiteLLM"}]},
        "future": {"preserved": True},
    }


@pytest.mark.asyncio
async def test_search_required_provider_rejects_non_chatgpt_model() -> None:
    with patch(
        "litellm.passthrough.main.get_llm_provider",
        return_value=("gpt-5.6-sol", "openai", "sk-openai", "https://api.openai.com/v1"),
    ):
        with pytest.raises(BadRequestError, match="requires provider `chatgpt`"):
            await allm_passthrough_route(
                method="POST",
                endpoint="alpha/search",
                model="openai/gpt-5.6-sol",
                json={"model": "gpt-5.6-sol"},
                required_custom_llm_provider="chatgpt",
            )


@pytest.mark.asyncio
async def test_chatgpt_passthrough_runs_generic_guardrails() -> None:
    guardrail = MagicMock()
    guardrail.guardrail_name = "search-policy"
    guardrail.apply_guardrail = AsyncMock(return_value={"texts": []})
    data = {
        "model": "sol",
        "custom_llm_provider": "chatgpt",
        "required_custom_llm_provider": "chatgpt",
        "json": {"commands": {"search_query": [{"q": "LiteLLM"}]}},
    }

    result = await LlmPassthroughRouteHandler().process_input_messages(
        data=data,
        guardrail_to_apply=guardrail,
    )

    assert result is data
    guardrail.apply_guardrail.assert_awaited_once()
