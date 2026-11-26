import json
import sys
from collections.abc import Callable
from functools import partial
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Final
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
import responses
import respx
from fastapi.testclient import TestClient

import litellm
from litellm._version import version
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.llms.databricks.chat.transformation import DatabricksConfig
from litellm.llms.databricks.common_utils import (
    DatabricksBase,
    DatabricksException,
)
from litellm.llms.databricks.messages.transformation import DatabricksAnthropicMessagesConfig


def test_databricks_validate_environment():
    databricks_base = DatabricksBase()

    with patch.object(databricks_base, "_get_databricks_credentials") as mock_get_credentials:
        try:
            databricks_base.databricks_validate_environment(
                api_key=None,
                api_base="my_api_base",
                endpoint_type="chat_completions",
                custom_endpoint=False,
                headers=None,
            )
        except Exception:
            pass
        mock_get_credentials.assert_called_once()


SDK_HOST: Final = "https://my.workspace.cloud.databricks.com"
SDK_TOKEN: Final = "sdk-test-token"
MISSING_SDK_MESSAGE: Final = (
    "If the Databricks base URL and API key are not set, the databricks-sdk Python library must be installed."
)


def _install_fake_databricks_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    config: Final = SimpleNamespace(
        host=SDK_HOST,
        authenticate=lambda: {"Authorization": f"Bearer {SDK_TOKEN}"},
    )
    sdk_module: Final = ModuleType("databricks.sdk")
    package_module: Final = ModuleType("databricks")
    package_module.__path__ = []
    setattr(sdk_module, "WorkspaceClient", lambda: SimpleNamespace(config=config))
    setattr(sdk_module, "useragent", SimpleNamespace(with_partner=lambda _: None))
    setattr(package_module, "sdk", sdk_module)
    monkeypatch.setitem(sys.modules, "databricks", package_module)
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk_module)


@pytest.fixture
def _sdk_only_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATABRICKS_API_BASE", "DATABRICKS_API_KEY", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    _install_fake_databricks_sdk(monkeypatch)


def _assert_sdk_request(request: httpx.Request, expected_url: str, expected_body: dict[str, object]) -> None:
    assert str(request.url) == expected_url
    assert request.headers["Authorization"] == f"Bearer {SDK_TOKEN}"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["User-Agent"] == f"litellm/{version}"
    assert json.loads(request.content) == expected_body


@pytest.mark.parametrize("set_base", [True, False])
def test_throws_if_api_base_or_api_key_not_set_without_databricks_sdk(
    monkeypatch: pytest.MonkeyPatch, set_base: bool
) -> None:
    monkeypatch.setitem(sys.modules, "databricks.sdk", None)
    if set_base:
        monkeypatch.setenv("DATABRICKS_API_BASE", f"{SDK_HOST}/serving-endpoints")
        monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    else:
        monkeypatch.setenv("DATABRICKS_API_KEY", "dapimykey")
        monkeypatch.delenv("DATABRICKS_API_BASE", raising=False)

    with pytest.raises(BadRequestError) as completion_error:
        litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages=[{"role": "user", "content": "How are you?"}],
        )
    with pytest.raises(BadRequestError) as embedding_error:
        litellm.embedding(model="databricks/bge-12312", input=["Hello", "World"])

    for error in (completion_error.value, embedding_error.value):
        assert error.status_code == 400
        assert error.llm_provider == "databricks"
        assert MISSING_SDK_MESSAGE in error.message


@pytest.mark.respx(assert_all_called=True)
def test_completions_uses_databricks_sdk_if_api_key_and_base_not_specified(
    _sdk_only_credentials: None, respx_mock: respx.MockRouter
) -> None:
    upstream_response: Final = {
        "id": "chatcmpl_3f78f09a-489c-4b8d-a587-f162c7497891",
        "object": "chat.completion",
        "created": 1726285449,
        "model": "dbrx-instruct-071224",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm an AI assistant. I'm doing well. How can I help?",
                    "function_call": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 230,
            "completion_tokens": 38,
            "completion_tokens_details": None,
            "total_tokens": 268,
            "prompt_tokens_details": None,
        },
        "system_fingerprint": None,
    }
    url: Final = f"{SDK_HOST}/serving-endpoints/chat/completions"
    route: Final = respx_mock.post(url).mock(return_value=httpx.Response(200, json=upstream_response))
    messages: Final = [{"role": "user", "content": "How are you?"}]

    response: Final = litellm.completion(
        model="databricks/dbrx-instruct-071224",
        messages=messages,
        temperature=0.5,
        extraparam="testpassingextraparam",
    )

    _assert_sdk_request(
        route.calls.last.request,
        url,
        {
            "model": "dbrx-instruct-071224",
            "messages": messages,
            "temperature": 0.5,
            "extraparam": "testpassingextraparam",
        },
    )
    assert response.to_dict() == {**upstream_response, "model": "databricks/dbrx-instruct-071224"}


@pytest.mark.respx(assert_all_called=True)
def test_embeddings_uses_databricks_sdk_if_api_key_and_base_not_specified(
    _sdk_only_credentials: None, respx_mock: respx.MockRouter
) -> None:
    upstream_response: Final = {
        "object": "list",
        "model": "bge-large-en-v1.5",
        "data": [
            {
                "index": 0,
                "object": "embedding",
                "embedding": [0.06768798828125, -0.01291656494140625, -0.0501708984375],
            },
            {
                "index": 1,
                "object": "embedding",
                "embedding": [0.0245361328125, -0.030364990234375, 0.0137939453125],
            },
        ],
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8,
            "completion_tokens": 0,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }
    url: Final = f"{SDK_HOST}/serving-endpoints/embeddings"
    route: Final = respx_mock.post(url).mock(return_value=httpx.Response(200, json=upstream_response))
    inputs: Final = ["Hello", "World"]

    response: Final = litellm.embedding(
        model="databricks/bge-large-en-v1.5",
        input=inputs,
        extraparam="testpassingextraparam",
    )

    _assert_sdk_request(
        route.calls.last.request,
        url,
        {"model": "bge-large-en-v1.5", "input": inputs, "extraparam": "testpassingextraparam"},
    )
    assert response.to_dict() == upstream_response


@responses.activate
def test_should_reuse_the_token_until_its_refresh_margin() -> None:
    clock: Final = Mock(return_value=1000.0)
    get_token: Final = partial(
        DatabricksBase().get_cached_oauth_m2m_token,
        client_id="synthetic-client",
        client_secret="synthetic-secret",
        token_cache=InMemoryCache(clock=clock),
    )
    responses.post(f"{SDK_HOST}/oidc/v1/token", json={"access_token": "first-token", "expires_in": 3600})
    responses.post(f"{SDK_HOST}/oidc/v1/token", json={"access_token": "refreshed-token", "expires_in": 3600})

    assert get_token(SDK_HOST) == "first-token"
    clock.return_value = 4539.0
    assert get_token(f"{SDK_HOST}/serving-endpoints/") == "first-token"
    assert len(responses.calls) == 1
    clock.return_value = 4541.0
    assert get_token(SDK_HOST) == "refreshed-token"
    assert len(responses.calls) == 2


@responses.activate
@pytest.mark.parametrize("expires_in", (0, 30, 60))
def test_should_not_cache_a_token_within_the_refresh_margin(expires_in: int) -> None:
    get_token: Final = partial(
        DatabricksBase().get_cached_oauth_m2m_token,
        SDK_HOST,
        "synthetic-client",
        "synthetic-secret",
        token_cache=InMemoryCache(),
    )
    token: Final = responses.post(
        f"{SDK_HOST}/oidc/v1/token", json={"access_token": "short-lived-token", "expires_in": expires_in}
    )

    assert get_token() == "short-lived-token"
    assert get_token() == "short-lived-token"
    assert token.call_count == 2


@responses.activate
@pytest.mark.parametrize("changed", ("workspace", "client_id", "client_secret"))
def test_should_not_reuse_a_token_for_a_different_workspace_or_credentials(changed: str) -> None:
    base: Final = DatabricksBase()
    cache: Final = InMemoryCache()
    responses.post(f"{SDK_HOST}/oidc/v1/token", json={"access_token": "original-token", "expires_in": 3600})
    assert (
        base.get_cached_oauth_m2m_token(SDK_HOST, "synthetic-client", "synthetic-secret", token_cache=cache)
        == "original-token"
    )

    next_host: Final = "https://other-workspace.example" if changed == "workspace" else SDK_HOST
    client_id: Final = "rotated-client" if changed == "client_id" else "synthetic-client"
    client_secret: Final = "rotated-secret" if changed == "client_secret" else "synthetic-secret"
    responses.post(f"{next_host}/oidc/v1/token", json={"access_token": "new-token", "expires_in": 3600})
    assert base.get_cached_oauth_m2m_token(next_host, client_id, client_secret, token_cache=cache) == "new-token"
    assert len(responses.calls) == 2


@responses.activate
def test_should_propagate_oauth_failure_and_retry_the_next_request() -> None:
    get_token: Final = partial(
        DatabricksBase().get_cached_oauth_m2m_token,
        SDK_HOST,
        "synthetic-client",
        "synthetic-secret",
        token_cache=InMemoryCache(),
    )
    responses.post(f"{SDK_HOST}/oidc/v1/token", status=401, json={"error": "invalid_client"})
    responses.post(f"{SDK_HOST}/oidc/v1/token", json={"access_token": "valid-token", "expires_in": 3600})

    with pytest.raises(DatabricksException, match="OAuth M2M token request failed"):
        get_token()
    assert get_token() == "valid-token"


@pytest.mark.parametrize(
    ("deployment_credentials", "missing"),
    [
        ({"client_id": "only-id"}, "client_secret"),
        ({"client_secret": "only-secret"}, "client_id"),
        ({"client_id": "only-id", "client_secret": ""}, "client_secret"),
    ],
)
@responses.activate
def test_chat_validate_environment_rejects_half_deployment_pair_instead_of_mixing_with_global(
    monkeypatch: pytest.MonkeyPatch, deployment_credentials: dict[str, str], missing: str
) -> None:
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "global-id")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "global-secret")

    with pytest.raises(
        DatabricksException, match=rf"Incomplete Databricks OAuth credentials: missing {missing}\."
    ) as exc_info:
        DatabricksConfig().validate_environment(
            headers={},
            model="databricks/dbrx-instruct",
            messages=[],
            optional_params={},
            litellm_params=deployment_credentials,
            api_base=f"{SDK_HOST}/serving-endpoints",
        )
    assert exc_info.value.status_code == 400
    assert redact_string(exc_info.value.message) == exc_info.value.message
    assert len(responses.calls) == 0


_DEPLOYMENT_PAIR: Final = MappingProxyType({"client_id": "deployment-id", "client_secret": "deployment-secret"})


@pytest.mark.parametrize(
    "validate",
    [
        pytest.param(
            lambda: DatabricksConfig().validate_environment(
                headers={},
                model="databricks/dbrx-instruct",
                messages=[],
                optional_params={},
                litellm_params=dict(_DEPLOYMENT_PAIR),
                api_base=None,
            ),
            id="chat",
        ),
        pytest.param(
            lambda: DatabricksBase().databricks_validate_environment(
                api_key=None,
                api_base=None,
                endpoint_type="embeddings",
                custom_endpoint=False,
                headers={},
                **_DEPLOYMENT_PAIR,
            ),
            id="embeddings",
        ),
        pytest.param(
            lambda: DatabricksAnthropicMessagesConfig().validate_anthropic_messages_environment(
                headers={},
                model="databricks-claude-3-5-sonnet",
                messages=[],
                optional_params={},
                litellm_params=dict(_DEPLOYMENT_PAIR),
                api_base=None,
            ),
            id="messages",
        ),
    ],
)
@responses.activate
def test_deployment_pair_without_api_base_is_rejected_instead_of_authenticating_as_the_ambient_identity(
    monkeypatch: pytest.MonkeyPatch, validate: Callable[[], object]
) -> None:
    monkeypatch.delenv("DATABRICKS_API_BASE", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", SDK_HOST)
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "global-id")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "global-secret")

    with pytest.raises(DatabricksException, match="Missing api_base") as exc_info:
        validate()
    assert exc_info.value.status_code == 400
    assert redact_string(exc_info.value.message) == exc_info.value.message
    assert len(responses.calls) == 0


def test_resolve_api_key_skips_ambient_keys_only_when_the_deployment_sets_oauth_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_API_KEY", "ambient-pat")

    assert DatabricksBase.resolve_api_key(None) == "ambient-pat"
    assert DatabricksBase.resolve_api_key(None, "deployment-id", "deployment-secret") is None
    assert DatabricksBase.resolve_api_key("deployment-pat", "deployment-id", "deployment-secret") == "deployment-pat"
    assert DatabricksBase.resolve_api_key("deployment-pat", "only-id") == "deployment-pat"


@responses.activate
def test_chat_validate_environment_deployment_oauth() -> None:
    config: Final = DatabricksConfig()
    responses.post(
        f"{SDK_HOST}/oidc/v1/token",
        json={"access_token": "chat-oauth-token", "expires_in": 3600},
    )

    headers: Final = config.validate_environment(
        headers={},
        model="databricks/dbrx-instruct",
        messages=[],
        optional_params={},
        litellm_params={
            "client_id": "chat-client-id",
            "client_secret": "chat-client-secret",
        },
        api_base=f"{SDK_HOST}/serving-endpoints",
    )
    assert headers["Authorization"] == "Bearer chat-oauth-token"
    assert headers["Content-Type"] == "application/json"


@responses.activate
def test_chat_validate_environment_static_api_key_precedence() -> None:
    config: Final = DatabricksConfig()
    headers: Final = config.validate_environment(
        headers={},
        model="databricks/dbrx-instruct",
        messages=[],
        optional_params={},
        litellm_params={
            "client_id": "chat-client-id",
            "client_secret": "chat-client-secret",
        },
        api_key="static-test-api-key",
        api_base=f"{SDK_HOST}/serving-endpoints",
    )
    assert headers["Authorization"] == "Bearer static-test-api-key"
    assert len(responses.calls) == 0


@responses.activate
def test_chat_validate_environment_global_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "global-chat-client")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "global-chat-secret")

    config: Final = DatabricksConfig()
    responses.post(
        f"{SDK_HOST}/oidc/v1/token",
        json={"access_token": "global-chat-token", "expires_in": 3600},
    )

    headers: Final = config.validate_environment(
        headers={},
        model="databricks/dbrx-instruct",
        messages=[],
        optional_params={},
        litellm_params={},
        api_base=f"{SDK_HOST}/serving-endpoints",
    )
    assert headers["Authorization"] == "Bearer global-chat-token"


@responses.activate
def test_embeddings_validate_environment_deployment_oauth() -> None:
    base: Final = DatabricksBase()
    responses.post(
        f"{SDK_HOST}/oidc/v1/token",
        json={"access_token": "embed-oauth-token", "expires_in": 3600},
    )

    url, headers = base.databricks_validate_environment(
        api_key=None,
        api_base=f"{SDK_HOST}/serving-endpoints",
        endpoint_type="embeddings",
        custom_endpoint=False,
        headers={},
        client_id="embed-client",
        client_secret="embed-secret",
    )
    assert headers["Authorization"] == "Bearer embed-oauth-token"
    assert url == f"{SDK_HOST}/serving-endpoints/embeddings"


@responses.activate
def test_embeddings_validate_environment_static_api_key_precedence() -> None:
    base: Final = DatabricksBase()
    url, headers = base.databricks_validate_environment(
        api_key="embed-static-key",
        api_base=f"{SDK_HOST}/serving-endpoints",
        endpoint_type="embeddings",
        custom_endpoint=False,
        headers={},
        client_id="embed-client",
        client_secret="embed-secret",
    )
    assert headers["Authorization"] == "Bearer embed-static-key"
    assert len(responses.calls) == 0


@responses.activate
def test_messages_validate_environment_deployment_oauth() -> None:
    config: Final = DatabricksAnthropicMessagesConfig()
    responses.post(
        f"{SDK_HOST}/oidc/v1/token",
        json={"access_token": "messages-oauth-token", "expires_in": 3600},
    )

    headers, workspace = config.validate_anthropic_messages_environment(
        headers={},
        model="databricks-claude-3-5-sonnet",
        messages=[],
        optional_params={},
        litellm_params={
            "client_id": "messages-client",
            "client_secret": "messages-secret",
        },
        api_base=f"{SDK_HOST}/serving-endpoints",
    )
    assert headers["Authorization"] == "Bearer messages-oauth-token"
    assert workspace == f"{SDK_HOST}/serving-endpoints"


@responses.activate
def test_messages_validate_environment_static_api_key_precedence() -> None:
    config: Final = DatabricksAnthropicMessagesConfig()
    headers, workspace = config.validate_anthropic_messages_environment(
        headers={},
        model="databricks-claude-3-5-sonnet",
        messages=[],
        optional_params={},
        litellm_params={
            "client_id": "messages-client",
            "client_secret": "messages-secret",
        },
        api_key="messages-static-key",
        api_base=f"{SDK_HOST}/serving-endpoints",
    )
    assert headers["Authorization"] == "Bearer messages-static-key"
    assert len(responses.calls) == 0
