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

    def test_env_openai_api_key_beats_litellm_params(
        self, deployment_wif: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        assert (
            resolve_openai_workload_identity_config(api_key=None, api_base=None, litellm_params=deployment_wif) is None
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
