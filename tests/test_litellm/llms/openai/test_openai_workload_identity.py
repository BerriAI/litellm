import json
import sys
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.llms.litellm_proxy.responses.transformation import LiteLLMProxyResponsesAPIConfig
from litellm.llms.openai.common_utils import BaseOpenAILLM, OpenAIError
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.openai.workload_identity import (
    OpenAIWorkloadIdentityConfig,
    _workload_identity_auth,
    get_workload_identity_bearer_token,
    resolve_openai_workload_identity_config,
)
from litellm.types.router import GenericLiteLLMParams

TOKEN_EXCHANGE_URL: Final = "https://auth.openai.com/oauth/token"


@pytest.fixture
def wif_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> OpenAIWorkloadIdentityConfig:
    token_file: Final = tmp_path / "subject_token.jwt"
    token_file.write_text("subject-token-from-file")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setenv("OPENAI_IDENTITY_PROVIDER_ID", "idp_test123")
    monkeypatch.setenv("OPENAI_SERVICE_ACCOUNT_ID", "user-test456")
    monkeypatch.setenv("OPENAI_IDENTITY_TOKEN_FILE", str(token_file))
    _workload_identity_auth.cache_clear()
    litellm.in_memory_llm_clients_cache.flush_cache()
    return OpenAIWorkloadIdentityConfig(
        identity_provider_id="idp_test123",
        service_account_id="user-test456",
        token_file=str(token_file),
    )


def mock_token_exchange(access_token: str = "exchanged-bearer-token") -> respx.Route:
    return respx.post(TOKEN_EXCHANGE_URL).mock(
        return_value=httpx.Response(200, json={"access_token": access_token, "expires_in": 3600})
    )


class TestResolveConfig:
    def test_resolves_from_env(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        assert resolve_openai_workload_identity_config(api_key=None, api_base=None) == wif_env

    def test_static_api_key_wins(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        assert resolve_openai_workload_identity_config(api_key="sk-static", api_base=None) is None

    def test_env_openai_api_key_wins(
        self, wif_env: OpenAIWorkloadIdentityConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        assert resolve_openai_workload_identity_config(api_key=None, api_base=None) is None

    @pytest.mark.parametrize("empty_key", ["", "   "])
    def test_empty_api_key_arg_does_not_disable_wif(
        self, wif_env: OpenAIWorkloadIdentityConfig, empty_key: str
    ) -> None:
        assert resolve_openai_workload_identity_config(api_key=empty_key, api_base=None) == wif_env

    @pytest.mark.parametrize("empty_key", ["", "   "])
    def test_empty_env_openai_api_key_does_not_disable_wif(
        self, wif_env: OpenAIWorkloadIdentityConfig, monkeypatch: pytest.MonkeyPatch, empty_key: str
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", empty_key)
        assert resolve_openai_workload_identity_config(api_key=None, api_base=None) == wif_env

    def test_foreign_api_base_disables(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        assert resolve_openai_workload_identity_config(api_key=None, api_base="https://my-vllm.internal/v1") is None

    def test_openai_api_base_allows(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        assert resolve_openai_workload_identity_config(api_key=None, api_base="https://api.openai.com/v1") == wif_env

    def test_plaintext_http_api_base_disables(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        assert resolve_openai_workload_identity_config(api_key=None, api_base="http://api.openai.com/v1") is None

    def test_foreign_env_base_url_disables(
        self, wif_env: OpenAIWorkloadIdentityConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://my-vllm.internal/v1")
        assert resolve_openai_workload_identity_config(api_key=None, api_base=None) is None

    def test_openai_env_base_url_allows(
        self, wif_env: OpenAIWorkloadIdentityConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        assert resolve_openai_workload_identity_config(api_key=None, api_base=None) == wif_env

    def test_foreign_litellm_api_base_disables(
        self, wif_env: OpenAIWorkloadIdentityConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "api_base", "https://my-vllm.internal/v1")
        assert resolve_openai_workload_identity_config(api_key=None, api_base=None) is None

    @pytest.mark.parametrize(
        "missing_var",
        ["OPENAI_IDENTITY_PROVIDER_ID", "OPENAI_SERVICE_ACCOUNT_ID", "OPENAI_IDENTITY_TOKEN_FILE"],
    )
    def test_partial_env_disables(
        self, wif_env: OpenAIWorkloadIdentityConfig, monkeypatch: pytest.MonkeyPatch, missing_var: str
    ) -> None:
        monkeypatch.delenv(missing_var)
        assert resolve_openai_workload_identity_config(api_key=None, api_base=None) is None


class TestTokenExchange:
    @respx.mock
    def test_exchanges_subject_token_for_bearer(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        route: Final = mock_token_exchange()
        assert get_workload_identity_bearer_token(wif_env) == "exchanged-bearer-token"
        request_body: Final = json.loads(route.calls.last.request.content)
        assert request_body["grant_type"] == "urn:ietf:params:oauth:grant-type:token-exchange"
        assert request_body["subject_token"] == "subject-token-from-file"
        assert request_body["subject_token_type"] == "urn:ietf:params:oauth:token-type:jwt"
        assert request_body["identity_provider_id"] == "idp_test123"
        assert request_body["service_account_id"] == "user-test456"

    @respx.mock
    def test_token_cached_across_mints(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        route: Final = mock_token_exchange()
        first: Final = get_workload_identity_bearer_token(wif_env)
        second: Final = get_workload_identity_bearer_token(wif_env)
        assert first == second == "exchanged-bearer-token"
        assert route.call_count == 1

    def test_old_sdk_raises_upgrade_error(
        self, wif_env: OpenAIWorkloadIdentityConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import openai as openai_module

        monkeypatch.delattr(openai_module, "auth", raising=False)
        monkeypatch.setitem(sys.modules, "openai.auth", None)
        with pytest.raises(OpenAIError, match=r"openai>=2\.32\.0"):
            wif_env.to_sdk_workload_identity()


class TestClientConstruction:
    def test_sync_client_uses_workload_identity(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        client: Final = OpenAIChatCompletion()._get_openai_client(is_async=False, api_key=None, api_base=None)
        assert isinstance(client, OpenAI)
        assert client.api_key == "workload-identity-auth"
        assert client._workload_identity_auth is not None

    def test_async_client_uses_workload_identity(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        client: Final = OpenAIChatCompletion()._get_openai_client(is_async=True, api_key=None, api_base=None)
        assert isinstance(client, AsyncOpenAI)
        assert client.api_key == "workload-identity-auth"
        assert client._workload_identity_auth is not None

    def test_static_key_client_unaffected(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        client: Final = OpenAIChatCompletion()._get_openai_client(is_async=False, api_key="sk-static", api_base=None)
        assert isinstance(client, OpenAI)
        assert client.api_key == "sk-static"
        assert client._workload_identity_auth is None

    def test_cache_key_separates_wif_identities(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        other_config: Final = OpenAIWorkloadIdentityConfig(
            identity_provider_id="idp_other",
            service_account_id="user-other",
            token_file=wif_env.token_file,
        )
        keys: Final = tuple(
            BaseOpenAILLM.get_openai_client_cache_key(
                client_initialization_params={"api_key": None, "is_async": False, "workload_identity_config": config},
                client_type="openai",
            )
            for config in (wif_env, other_config, None)
        )
        assert len(set(keys)) == 3

    @respx.mock
    def test_request_carries_exchanged_bearer(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        mock_token_exchange()
        completion_route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-wif",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        )
        client = OpenAIChatCompletion()._get_openai_client(is_async=False, api_key=None, api_base=None)
        assert isinstance(client, OpenAI)
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        auth_header: Final = completion_route.calls.last.request.headers["Authorization"]
        assert auth_header == "Bearer exchanged-bearer-token"


class TestResponsesValidateEnvironment:
    @respx.mock
    def test_mints_bearer_when_wif_configured(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        mock_token_exchange()
        headers: Final = OpenAIResponsesAPIConfig().validate_environment(
            headers={}, model="gpt-4o-mini", litellm_params=GenericLiteLLMParams()
        )
        assert headers["Authorization"] == "Bearer exchanged-bearer-token"

    def test_static_key_wins(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        headers: Final = OpenAIResponsesAPIConfig().validate_environment(
            headers={}, model="gpt-4o-mini", litellm_params=GenericLiteLLMParams(api_key="sk-responses")
        )
        assert headers["Authorization"] == "Bearer sk-responses"

    def test_foreign_api_base_skips_wif(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        headers: Final = OpenAIResponsesAPIConfig().validate_environment(
            headers={},
            model="gpt-4o-mini",
            litellm_params=GenericLiteLLMParams(api_base="https://my-vllm.internal/v1"),
        )
        assert headers["Authorization"] == "Bearer None"

    def test_litellm_proxy_subclass_never_mints_wif(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        headers: Final = LiteLLMProxyResponsesAPIConfig().validate_environment(
            headers={}, model="gpt-4o-mini", litellm_params=GenericLiteLLMParams()
        )
        assert headers["Authorization"] == "Bearer None"
