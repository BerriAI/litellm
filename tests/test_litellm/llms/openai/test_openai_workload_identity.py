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
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai.common_utils import BaseOpenAILLM, OpenAIError
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.openai.workload_identity import (
    OpenAIWorkloadIdentityConfig,
    _workload_identity_auth,
    build_async_openai_client,
    build_openai_client,
    get_workload_identity_bearer_token,
    resolve_openai_workload_identity_config,
)
from litellm.types.router import GenericLiteLLMParams

TOKEN_EXCHANGE_URL: Final = "https://auth.openai.com/oauth/token"
CHAT_COMPLETIONS_URL: Final = "https://api.openai.com/v1/chat/completions"
EMBEDDINGS_URL: Final = "https://api.openai.com/v1/embeddings"
MODELS_URL: Final = "https://api.openai.com/v1/models"
CHAT_COMPLETION_BODY: Final = {
    "id": "chatcmpl-wif",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


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

    @pytest.mark.parametrize(
        "api_base",
        (
            "https://southcentralus.privatelink.api.openai.com/v1",
            "https://eu.api.openai.com/v1",
            "https://us.api.openai.com/v1",
        ),
    )
    def test_openai_backed_api_base_allows(self, wif_env: OpenAIWorkloadIdentityConfig, api_base: str) -> None:
        assert resolve_openai_workload_identity_config(api_key=None, api_base=api_base) == wif_env

    @pytest.mark.parametrize(
        "api_base",
        (
            "https://api.openai.com.evil.example/v1",
            "https://openai.com/v1",
            "https://euapi.openai.com/v1",
            "http://southcentralus.privatelink.api.openai.com/v1",
        ),
    )
    def test_lookalike_or_plaintext_api_base_disables(
        self, wif_env: OpenAIWorkloadIdentityConfig, api_base: str
    ) -> None:
        assert resolve_openai_workload_identity_config(api_key=None, api_base=api_base) is None

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

    def test_privatelink_client_uses_workload_identity(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        client: Final = OpenAIChatCompletion()._get_openai_client(
            is_async=False, api_key=None, api_base="https://southcentralus.privatelink.api.openai.com/v1"
        )
        assert isinstance(client, OpenAI)
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

    @respx.mock
    def test_privatelink_api_base_mints_bearer(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        mock_token_exchange()
        headers: Final = OpenAIResponsesAPIConfig().validate_environment(
            headers={},
            model="gpt-4o-mini",
            litellm_params=GenericLiteLLMParams(api_base="https://southcentralus.privatelink.api.openai.com/v1"),
        )
        assert headers["Authorization"] == "Bearer exchanged-bearer-token"

    def test_litellm_proxy_subclass_never_mints_wif(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        headers: Final = LiteLLMProxyResponsesAPIConfig().validate_environment(
            headers={}, model="gpt-4o-mini", litellm_params=GenericLiteLLMParams()
        )
        assert headers["Authorization"] == "Bearer None"


@pytest.fixture
def deployment_wif(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    token_file: Final = tmp_path / "deployment_subject_token.jwt"
    token_file.write_text("subject-token-from-deployment-file")
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_IDENTITY_PROVIDER_ID",
        "OPENAI_SERVICE_ACCOUNT_ID",
        "OPENAI_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _workload_identity_auth.cache_clear()
    litellm.in_memory_llm_clients_cache.flush_cache()
    return {
        "openai_identity_provider_id": "idp_deployment",
        "openai_service_account_id": "user-deployment",
        "openai_identity_token_file": str(token_file),
    }


def deployment_config(deployment_wif: dict[str, str]) -> OpenAIWorkloadIdentityConfig:
    return OpenAIWorkloadIdentityConfig(
        identity_provider_id="idp_deployment",
        service_account_id="user-deployment",
        token_file=deployment_wif["openai_identity_token_file"],
    )


def mock_chat_completions() -> respx.Route:
    return respx.post(CHAT_COMPLETIONS_URL).mock(return_value=httpx.Response(200, json=CHAT_COMPLETION_BODY))


def mock_streaming_chat_completions() -> respx.Route:
    chunk: Final = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    events: Final = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    body: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return respx.post(CHAT_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )


class TestResolveConfigFromDeployment:
    def test_resolves_from_litellm_params_without_env(self, deployment_wif: dict[str, str]) -> None:
        assert resolve_openai_workload_identity_config(
            api_key=None, api_base=None, litellm_params=deployment_wif
        ) == deployment_config(deployment_wif)

    def test_env_alone_disables_nothing_when_params_are_absent(self, deployment_wif: dict[str, str]) -> None:
        assert resolve_openai_workload_identity_config(api_key=None, api_base=None, litellm_params=None) is None

    def test_unrelated_litellm_params_do_not_resolve(self, deployment_wif: dict[str, str]) -> None:
        assert (
            resolve_openai_workload_identity_config(api_key=None, api_base=None, litellm_params={"model": "gpt-4o"})
            is None
        )

    def test_litellm_params_beat_env(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        config: Final = resolve_openai_workload_identity_config(
            api_key=None,
            api_base=None,
            litellm_params={
                "openai_identity_provider_id": "idp_deployment",
                "openai_service_account_id": "user-deployment",
                "openai_identity_token_file": wif_env.token_file,
            },
        )
        assert config == OpenAIWorkloadIdentityConfig(
            identity_provider_id="idp_deployment",
            service_account_id="user-deployment",
            token_file=wif_env.token_file,
        )

    def test_partial_litellm_params_fill_from_env_per_field(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        config: Final = resolve_openai_workload_identity_config(
            api_key=None, api_base=None, litellm_params={"openai_identity_provider_id": "idp_deployment"}
        )
        assert config == OpenAIWorkloadIdentityConfig(
            identity_provider_id="idp_deployment",
            service_account_id=wif_env.service_account_id,
            token_file=wif_env.token_file,
        )

    @pytest.mark.parametrize("blank", ["", None, 7])
    def test_blank_or_non_string_param_falls_back_to_env(
        self, wif_env: OpenAIWorkloadIdentityConfig, blank: object
    ) -> None:
        config: Final = resolve_openai_workload_identity_config(
            api_key=None, api_base=None, litellm_params={"openai_identity_provider_id": blank}
        )
        assert config == wif_env

    def test_partial_litellm_params_without_env_disable(self, deployment_wif: dict[str, str]) -> None:
        partial: Final = {key: value for key, value in deployment_wif.items() if key != "openai_identity_token_file"}
        assert resolve_openai_workload_identity_config(api_key=None, api_base=None, litellm_params=partial) is None

    def test_static_api_key_beats_litellm_params(self, deployment_wif: dict[str, str]) -> None:
        assert (
            resolve_openai_workload_identity_config(api_key="sk-static", api_base=None, litellm_params=deployment_wif)
            is None
        )

    def test_deployment_identity_beats_env_openai_api_key(
        self, deployment_wif: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        config: Final = resolve_openai_workload_identity_config(
            api_key="sk-from-env", api_base=None, litellm_params=deployment_wif
        )
        assert config is not None
        assert config.service_account_id == deployment_wif["openai_service_account_id"]

    @pytest.mark.parametrize("global_attr", ["api_key", "openai_key"])
    def test_deployment_identity_beats_litellm_module_key(
        self, deployment_wif: dict[str, str], monkeypatch: pytest.MonkeyPatch, global_attr: str
    ) -> None:
        monkeypatch.setattr(litellm, global_attr, "sk-module-global")
        config: Final = resolve_openai_workload_identity_config(
            api_key="sk-module-global", api_base=None, litellm_params=deployment_wif
        )
        assert config is not None
        assert config.identity_provider_id == deployment_wif["openai_identity_provider_id"]

    def test_env_openai_api_key_beats_partial_deployment_identity(
        self, deployment_wif: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        unrelated: Final = {key: value for key, value in deployment_wif.items() if not key.startswith("openai_")}
        assert (
            resolve_openai_workload_identity_config(api_key="sk-from-env", api_base=None, litellm_params=unrelated)
            is None
        )

    def test_foreign_api_base_disables_deployment_wif(self, deployment_wif: dict[str, str]) -> None:
        assert (
            resolve_openai_workload_identity_config(
                api_key=None, api_base="https://my-vllm.internal/v1", litellm_params=deployment_wif
            )
            is None
        )


class TestDeploymentClientConstruction:
    def test_sync_client_from_deployment_params(self, deployment_wif: dict[str, str]) -> None:
        client: Final = OpenAIChatCompletion()._get_openai_client(
            is_async=False, api_key=None, api_base=None, litellm_params=deployment_wif
        )
        assert isinstance(client, OpenAI)
        assert client.api_key == "workload-identity-auth"
        assert client._workload_identity_auth is not None

    def test_async_client_from_deployment_params(self, deployment_wif: dict[str, str]) -> None:
        client: Final = OpenAIChatCompletion()._get_openai_client(
            is_async=True, api_key=None, api_base=None, litellm_params=deployment_wif
        )
        assert isinstance(client, AsyncOpenAI)
        assert client._workload_identity_auth is not None

    def test_distinct_deployments_get_distinct_cached_clients(self, deployment_wif: dict[str, str]) -> None:
        other_deployment: Final = {**deployment_wif, "openai_service_account_id": "user-other"}
        handler: Final = OpenAIChatCompletion()
        first: Final = handler._get_openai_client(
            is_async=False, api_key=None, api_base=None, litellm_params=deployment_wif
        )
        second: Final = handler._get_openai_client(
            is_async=False, api_key=None, api_base=None, litellm_params=other_deployment
        )
        again: Final = handler._get_openai_client(
            is_async=False, api_key=None, api_base=None, litellm_params=dict(deployment_wif)
        )
        assert first is not second
        assert again is first

    def test_builders_reuse_cached_client_per_identity(self, deployment_wif: dict[str, str]) -> None:
        other_deployment: Final = {**deployment_wif, "openai_service_account_id": "user-other"}
        first: Final = build_openai_client(api_key=None, api_base=None, litellm_params=deployment_wif)
        again: Final = build_openai_client(api_key=None, api_base=None, litellm_params=dict(deployment_wif))
        other: Final = build_openai_client(api_key=None, api_base=None, litellm_params=other_deployment)
        async_first: Final = build_async_openai_client(api_key=None, api_base=None, litellm_params=deployment_wif)
        async_again: Final = build_async_openai_client(api_key=None, api_base=None, litellm_params=deployment_wif)
        assert again is first
        assert other is not first
        assert async_again is async_first
        assert isinstance(async_first, AsyncOpenAI)

    def test_builders_never_cache_static_key_clients(self) -> None:
        first: Final = build_openai_client(api_key="sk-static", api_base=None, litellm_params=None)
        again: Final = build_openai_client(api_key="sk-static", api_base=None, litellm_params=None)
        assert again is not first

    @respx.mock
    def test_completion_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("deployment-bearer")
        completion_route: Final = mock_chat_completions()

        response: Final = litellm.completion(
            model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], **deployment_wif
        )

        assert response.choices[0].message.content == "ok"
        request: Final = completion_route.calls.last.request
        assert request.headers["Authorization"] == "Bearer deployment-bearer"
        assert not any(key.startswith("openai_") for key in json.loads(request.content))

    @respx.mock
    def test_streaming_completion_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("stream-bearer")
        stream_route: Final = mock_streaming_chat_completions()

        chunks: Final = tuple(
            litellm.completion(
                model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], stream=True, **deployment_wif
            )
        )

        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "ok"
        assert stream_route.calls.last.request.headers["Authorization"] == "Bearer stream-bearer"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_streaming_completion_kwargs_carry_exchanged_bearer(
        self, deployment_wif: dict[str, str]
    ) -> None:
        mock_token_exchange("async-stream-bearer")
        stream_route: Final = mock_streaming_chat_completions()

        stream: Final = await litellm.acompletion(
            model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], stream=True, **deployment_wif
        )
        chunks: Final = tuple([chunk async for chunk in stream])

        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "ok"
        assert stream_route.calls.last.request.headers["Authorization"] == "Bearer async-stream-bearer"

    @respx.mock
    def test_router_deployment_without_api_key_authenticates_via_token_exchange(
        self, deployment_wif: dict[str, str]
    ) -> None:
        exchange_route: Final = mock_token_exchange("router-bearer")
        completion_route: Final = mock_chat_completions()
        router: Final = litellm.Router(
            model_list=[{"model_name": "wif-gpt", "litellm_params": {"model": "openai/gpt-4o-mini", **deployment_wif}}]
        )

        response: Final = router.completion(model="wif-gpt", messages=[{"role": "user", "content": "hi"}])

        assert response.choices[0].message.content == "ok"
        assert exchange_route.called
        assert completion_route.calls.last.request.headers["Authorization"] == "Bearer router-bearer"

    @respx.mock
    def test_embedding_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("embedding-bearer")
        embeddings_route: Final = respx.post(EMBEDDINGS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        )

        litellm.embedding(model="openai/text-embedding-3-small", input=["hi"], **deployment_wif)

        assert embeddings_route.calls.last.request.headers["Authorization"] == "Bearer embedding-bearer"


class TestResponsesValidateEnvironmentFromDeployment:
    @respx.mock
    def test_mints_bearer_from_litellm_params(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("responses-bearer")
        headers: Final = OpenAIResponsesAPIConfig().validate_environment(
            headers={}, model="gpt-4o-mini", litellm_params=GenericLiteLLMParams(**deployment_wif)
        )
        assert headers["Authorization"] == "Bearer responses-bearer"

    def test_static_key_in_litellm_params_wins(self, deployment_wif: dict[str, str]) -> None:
        headers: Final = OpenAIResponsesAPIConfig().validate_environment(
            headers={},
            model="gpt-4o-mini",
            litellm_params=GenericLiteLLMParams(api_key="sk-responses", **deployment_wif),
        )
        assert headers["Authorization"] == "Bearer sk-responses"


class TestDiscoverModels:
    @staticmethod
    def mock_models() -> respx.Route:
        return respx.get(MODELS_URL).mock(
            return_value=httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4.1"}]})
        )

    @respx.mock
    def test_discovers_with_exchanged_bearer_from_litellm_params(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("discovery-bearer")
        models_route: Final = self.mock_models()

        assert OpenAIGPTConfig().discover_models(deployment_wif) == ["gpt-4o-mini", "gpt-4.1"]
        assert models_route.calls.last.request.headers["Authorization"] == "Bearer discovery-bearer"

    @respx.mock
    def test_discovers_with_env_wif_when_params_carry_no_key(self, wif_env: OpenAIWorkloadIdentityConfig) -> None:
        mock_token_exchange("env-discovery-bearer")
        models_route: Final = self.mock_models()

        OpenAIGPTConfig().discover_models({})

        assert models_route.calls.last.request.headers["Authorization"] == "Bearer env-discovery-bearer"

    @respx.mock
    def test_static_api_key_in_params_skips_token_exchange(self, deployment_wif: dict[str, str]) -> None:
        exchange_route: Final = mock_token_exchange()
        models_route: Final = self.mock_models()

        OpenAIGPTConfig().discover_models({**deployment_wif, "api_key": "sk-discovery"})

        assert models_route.calls.last.request.headers["Authorization"] == "Bearer sk-discovery"
        assert not exchange_route.called

    @respx.mock
    def test_blank_api_base_in_params_discovers_from_openai(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("blank-base-bearer")
        models_route: Final = self.mock_models()

        assert OpenAIGPTConfig().discover_models({**deployment_wif, "api_base": ""}) == ["gpt-4o-mini", "gpt-4.1"]
        assert models_route.calls.last.request.headers["Authorization"] == "Bearer blank-base-bearer"

    @respx.mock
    def test_openai_compatible_subclass_never_mints_wif(self, deployment_wif: dict[str, str]) -> None:
        exchange_route: Final = mock_token_exchange()
        models_route: Final = self.mock_models()

        class CompatibleConfig(OpenAIGPTConfig):
            pass

        CompatibleConfig().discover_models(deployment_wif)

        assert models_route.calls.last.request.headers["Authorization"] == "Bearer None"
        assert not exchange_route.called

    @respx.mock
    def test_empty_static_key_never_borrows_the_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-that-must-stay-home")
        foreign_models: Final = respx.get("https://third-party.example/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "other-model"}]})
        )

        assert OpenAIGPTConfig().get_models(api_key="", api_base="https://third-party.example") == ["other-model"]
        assert foreign_models.calls.last.request.headers["Authorization"] == "Bearer "


class TestClientsideBaseOverride:
    def test_client_api_base_override_clears_deployment_wif(self, deployment_wif: dict[str, str]) -> None:
        from litellm.router_utils.clientside_credential_handler import get_dynamic_litellm_params

        redirected: Final = get_dynamic_litellm_params(
            litellm_params={"model": "openai/gpt-4o-mini", **deployment_wif},
            request_kwargs={"api_base": "https://not-openai.example/v1"},
        )

        assert not any(key in redirected for key in deployment_wif)
        assert (
            resolve_openai_workload_identity_config(
                api_key=None, api_base=redirected["api_base"], litellm_params=redirected
            )
            is None
        )


IMAGES_URL: Final = "https://api.openai.com/v1/images/generations"
SPEECH_URL: Final = "https://api.openai.com/v1/audio/speech"
TRANSCRIPTIONS_URL: Final = "https://api.openai.com/v1/audio/transcriptions"
MODERATIONS_URL: Final = "https://api.openai.com/v1/moderations"
COMPLETIONS_URL: Final = "https://api.openai.com/v1/completions"
FILES_URL: Final = "https://api.openai.com/v1/files"
BATCHES_URL: Final = "https://api.openai.com/v1/batches"
FINE_TUNING_JOBS_URL: Final = "https://api.openai.com/v1/fine_tuning/jobs"
IMAGE_BODY: Final = {"created": 1, "data": [{"b64_json": "aGk="}]}
MODERATION_BODY: Final = {
    "id": "modr-1",
    "model": "omni-moderation-latest",
    "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
}
TEXT_COMPLETION_BODY: Final = {
    "id": "cmpl-1",
    "object": "text_completion",
    "created": 1,
    "model": "gpt-3.5-turbo-instruct",
    "choices": [{"text": "ok", "index": 0, "finish_reason": "stop", "logprobs": None}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
FILE_BODY: Final = {
    "id": "file-1",
    "object": "file",
    "bytes": 2,
    "created_at": 1,
    "filename": "in.jsonl",
    "purpose": "batch",
    "status": "processed",
}
FILE_DELETED_BODY: Final = {"id": "file-1", "object": "file", "deleted": True}
BATCH_BODY: Final = {
    "id": "batch-1",
    "object": "batch",
    "endpoint": "/v1/chat/completions",
    "input_file_id": "file-1",
    "completion_window": "24h",
    "status": "validating",
    "created_at": 1,
}
FINE_TUNING_JOB_BODY: Final = {
    "id": "ftjob-1",
    "object": "fine_tuning.job",
    "created_at": 1,
    "error": None,
    "fine_tuned_model": None,
    "finished_at": None,
    "hyperparameters": {"n_epochs": "auto"},
    "model": "gpt-5.4-nano",
    "organization_id": "org-1",
    "result_files": [],
    "seed": 0,
    "status": "queued",
    "trained_tokens": None,
    "training_file": "file-1",
    "validation_file": None,
}


def mock_streaming_text_completions() -> respx.Route:
    events: Final = (
        {
            "id": "cmpl-1",
            "object": "text_completion",
            "created": 1,
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{"text": "ok", "index": 0, "finish_reason": None, "logprobs": None}],
        },
        {
            "id": "cmpl-1",
            "object": "text_completion",
            "created": 1,
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{"text": "", "index": 0, "finish_reason": "stop", "logprobs": None}],
        },
    )
    body: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return respx.post(COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    )


def bearer_of(route: respx.Route) -> str:
    return route.calls.last.request.headers["Authorization"]


class TestDeploymentNonChatSurfaces:
    @respx.mock
    def test_image_generation_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("image-bearer")
        route: Final = respx.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=IMAGE_BODY))

        response: Final = litellm.image_generation(model="openai/gpt-image-2", prompt="a cat", **deployment_wif)

        assert response.data[0].b64_json == "aGk="
        assert bearer_of(route) == "Bearer image-bearer"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_image_generation_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("aimage-bearer")
        route: Final = respx.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=IMAGE_BODY))

        response: Final = await litellm.aimage_generation(model="openai/gpt-image-2", prompt="a cat", **deployment_wif)

        assert response.data[0].b64_json == "aGk="
        assert bearer_of(route) == "Bearer aimage-bearer"

    @respx.mock
    def test_speech_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("speech-bearer")
        route: Final = respx.post(SPEECH_URL).mock(
            return_value=httpx.Response(200, content=b"audio", headers={"content-type": "audio/mpeg"})
        )

        response: Final = litellm.speech(model="openai/gpt-4o-mini-tts", input="hi", voice="alloy", **deployment_wif)

        assert response.content == b"audio"
        assert bearer_of(route) == "Bearer speech-bearer"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_speech_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("aspeech-bearer")
        route: Final = respx.post(SPEECH_URL).mock(
            return_value=httpx.Response(200, content=b"audio", headers={"content-type": "audio/mpeg"})
        )

        response: Final = await litellm.aspeech(
            model="openai/gpt-4o-mini-tts", input="hi", voice="alloy", **deployment_wif
        )

        assert response.content == b"audio"
        assert bearer_of(route) == "Bearer aspeech-bearer"

    @respx.mock
    def test_transcription_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str], tmp_path: Path) -> None:
        mock_token_exchange("transcribe-bearer")
        route: Final = respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json={"text": "hi"}))
        audio: Final = tmp_path / "tone.wav"
        audio.write_bytes(b"RIFF....WAVE")

        with audio.open("rb") as audio_file:
            response: Final = litellm.transcription(
                model="openai/gpt-4o-mini-transcribe", file=audio_file, **deployment_wif
            )

        assert response.text == "hi"
        assert bearer_of(route) == "Bearer transcribe-bearer"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_transcription_kwargs_carry_exchanged_bearer(
        self, deployment_wif: dict[str, str], tmp_path: Path
    ) -> None:
        mock_token_exchange("atranscribe-bearer")
        route: Final = respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json={"text": "hi"}))
        audio: Final = tmp_path / "tone.wav"
        audio.write_bytes(b"RIFF....WAVE")

        with audio.open("rb") as audio_file:
            response: Final = await litellm.atranscription(
                model="openai/gpt-4o-mini-transcribe", file=audio_file, **deployment_wif
            )

        assert response.text == "hi"
        assert bearer_of(route) == "Bearer atranscribe-bearer"

    @respx.mock
    def test_moderation_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("moderation-bearer")
        route: Final = respx.post(MODERATIONS_URL).mock(return_value=httpx.Response(200, json=MODERATION_BODY))

        response: Final = litellm.moderation(input="hi", model="omni-moderation-latest", **deployment_wif)

        assert response.results[0].flagged is False
        assert bearer_of(route) == "Bearer moderation-bearer"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_moderation_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("amoderation-bearer")
        route: Final = respx.post(MODERATIONS_URL).mock(return_value=httpx.Response(200, json=MODERATION_BODY))

        response: Final = await litellm.amoderation(input="hi", model="omni-moderation-latest", **deployment_wif)

        assert response.results[0].flagged is False
        assert bearer_of(route) == "Bearer amoderation-bearer"

    @respx.mock
    def test_text_completion_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("text-bearer")
        route: Final = respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(200, json=TEXT_COMPLETION_BODY))

        response: Final = litellm.text_completion(
            model="text-completion-openai/gpt-3.5-turbo-instruct", prompt="hi", **deployment_wif
        )

        assert response.choices[0].text == "ok"
        assert bearer_of(route) == "Bearer text-bearer"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_text_completion_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("atext-bearer")
        route: Final = respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(200, json=TEXT_COMPLETION_BODY))

        response: Final = await litellm.atext_completion(
            model="text-completion-openai/gpt-3.5-turbo-instruct", prompt="hi", **deployment_wif
        )

        assert response.choices[0].text == "ok"
        assert bearer_of(route) == "Bearer atext-bearer"

    @respx.mock
    def test_streaming_text_completion_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("text-stream-bearer")
        route: Final = mock_streaming_text_completions()

        chunks: Final = tuple(
            litellm.text_completion(
                model="text-completion-openai/gpt-3.5-turbo-instruct", prompt="hi", stream=True, **deployment_wif
            )
        )

        assert chunks[0].choices[0].text == "ok"
        assert bearer_of(route) == "Bearer text-stream-bearer"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_streaming_text_completion_kwargs_carry_exchanged_bearer(
        self, deployment_wif: dict[str, str]
    ) -> None:
        mock_token_exchange("atext-stream-bearer")
        route: Final = mock_streaming_text_completions()

        stream: Final = await litellm.atext_completion(
            model="text-completion-openai/gpt-3.5-turbo-instruct", prompt="hi", stream=True, **deployment_wif
        )
        chunks: Final = [chunk async for chunk in stream]

        assert chunks[0].choices[0].text == "ok"
        assert bearer_of(route) == "Bearer atext-stream-bearer"

    @respx.mock
    @pytest.mark.parametrize(
        ("method", "url", "response", "invoke"),
        [
            pytest.param(
                "POST",
                FILES_URL,
                httpx.Response(200, json=FILE_BODY),
                lambda p: litellm.create_file(file=("in.jsonl", b"{}"), purpose="batch", **p),
                id="create_file",
            ),
            pytest.param(
                "GET",
                f"{FILES_URL}/file-1",
                httpx.Response(200, json=FILE_BODY),
                lambda p: litellm.file_retrieve(file_id="file-1", **p),
                id="file_retrieve",
            ),
            pytest.param(
                "DELETE",
                f"{FILES_URL}/file-1",
                httpx.Response(200, json=FILE_DELETED_BODY),
                lambda p: litellm.file_delete(file_id="file-1", **p),
                id="file_delete",
            ),
            pytest.param(
                "GET",
                FILES_URL,
                httpx.Response(200, json={"object": "list", "data": [FILE_BODY]}),
                lambda p: litellm.file_list(**p),
                id="file_list",
            ),
            pytest.param(
                "GET",
                f"{FILES_URL}/file-1/content",
                httpx.Response(200, content=b"{}"),
                lambda p: litellm.file_content(file_id="file-1", **p),
                id="file_content",
            ),
            pytest.param(
                "POST",
                BATCHES_URL,
                httpx.Response(200, json=BATCH_BODY),
                lambda p: litellm.create_batch(
                    completion_window="24h", endpoint="/v1/chat/completions", input_file_id="file-1", **p
                ),
                id="create_batch",
            ),
            pytest.param(
                "GET",
                f"{BATCHES_URL}/batch-1",
                httpx.Response(200, json=BATCH_BODY),
                lambda p: litellm.retrieve_batch(batch_id="batch-1", **p),
                id="retrieve_batch",
            ),
            pytest.param(
                "POST",
                f"{BATCHES_URL}/batch-1/cancel",
                httpx.Response(200, json=BATCH_BODY),
                lambda p: litellm.cancel_batch(batch_id="batch-1", **p),
                id="cancel_batch",
            ),
            pytest.param(
                "GET",
                BATCHES_URL,
                httpx.Response(200, json={"object": "list", "data": [BATCH_BODY], "has_more": False}),
                lambda p: litellm.list_batches(**p),
                id="list_batches",
            ),
            pytest.param(
                "POST",
                FINE_TUNING_JOBS_URL,
                httpx.Response(200, json=FINE_TUNING_JOB_BODY),
                lambda p: litellm.create_fine_tuning_job(model="gpt-5.4-nano", training_file="file-1", **p),
                id="create_fine_tuning_job",
            ),
            pytest.param(
                "POST",
                f"{FINE_TUNING_JOBS_URL}/ftjob-1/cancel",
                httpx.Response(200, json=FINE_TUNING_JOB_BODY),
                lambda p: litellm.cancel_fine_tuning_job(fine_tuning_job_id="ftjob-1", **p),
                id="cancel_fine_tuning_job",
            ),
            pytest.param(
                "GET",
                FINE_TUNING_JOBS_URL,
                httpx.Response(200, json={"object": "list", "data": [FINE_TUNING_JOB_BODY], "has_more": False}),
                lambda p: litellm.list_fine_tuning_jobs(**p),
                id="list_fine_tuning_jobs",
            ),
            pytest.param(
                "GET",
                f"{FINE_TUNING_JOBS_URL}/ftjob-1",
                httpx.Response(200, json=FINE_TUNING_JOB_BODY),
                lambda p: litellm.retrieve_fine_tuning_job(fine_tuning_job_id="ftjob-1", **p),
                id="retrieve_fine_tuning_job",
            ),
        ],
    )
    def test_managed_object_kwargs_carry_exchanged_bearer(
        self, deployment_wif: dict[str, str], method: str, url: str, response: httpx.Response, invoke
    ) -> None:
        mock_token_exchange("managed-bearer")
        route: Final = respx.route(method=method, url__startswith=url).mock(return_value=response)

        invoke({"custom_llm_provider": "openai", **deployment_wif})

        assert bearer_of(route) == "Bearer managed-bearer"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_managed_object_kwargs_carry_exchanged_bearer(self, deployment_wif: dict[str, str]) -> None:
        mock_token_exchange("amanaged-bearer")
        files_route: Final = respx.get(url__startswith=FILES_URL).mock(
            return_value=httpx.Response(200, json={"object": "list", "data": [FILE_BODY]})
        )
        batches_route: Final = respx.get(url__startswith=BATCHES_URL).mock(
            return_value=httpx.Response(200, json={"object": "list", "data": [BATCH_BODY], "has_more": False})
        )
        jobs_route: Final = respx.get(url__startswith=FINE_TUNING_JOBS_URL).mock(
            return_value=httpx.Response(200, json={"object": "list", "data": [FINE_TUNING_JOB_BODY], "has_more": False})
        )

        await litellm.afile_list(custom_llm_provider="openai", **deployment_wif)
        await litellm.alist_batches(custom_llm_provider="openai", **deployment_wif)
        await litellm.alist_fine_tuning_jobs(custom_llm_provider="openai", **deployment_wif)

        assert bearer_of(files_route) == "Bearer amanaged-bearer"
        assert bearer_of(batches_route) == "Bearer amanaged-bearer"
        assert bearer_of(jobs_route) == "Bearer amanaged-bearer"

    @respx.mock
    @pytest.mark.asyncio
    async def test_repeated_managed_object_calls_exchange_the_token_once(self, deployment_wif: dict[str, str]) -> None:
        exchange_route: Final = mock_token_exchange("once-bearer")
        files_route: Final = respx.get(url__startswith=FILES_URL).mock(
            return_value=httpx.Response(200, json={"object": "list", "data": [FILE_BODY]})
        )

        for _ in range(3):
            await litellm.afile_list(custom_llm_provider="openai", **deployment_wif)

        assert files_route.call_count == 3
        assert exchange_route.call_count == 1
        assert bearer_of(files_route) == "Bearer once-bearer"

    @respx.mock
    def test_repeated_moderation_calls_exchange_the_token_once(self, deployment_wif: dict[str, str]) -> None:
        exchange_route: Final = mock_token_exchange("once-bearer")
        moderation_route: Final = respx.post(MODERATIONS_URL).mock(return_value=httpx.Response(200, json=MODERATION_BODY))

        for _ in range(3):
            litellm.moderation(input="hi", model="omni-moderation-latest", **deployment_wif)

        assert moderation_route.call_count == 3
        assert exchange_route.call_count == 1
