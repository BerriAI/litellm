import json
import sys
from functools import partial
from types import ModuleType, SimpleNamespace
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
from litellm.llms.databricks.common_utils import DatabricksBase, DatabricksException


def test_databricks_validate_environment():
    databricks_base = DatabricksBase()

    with patch.object(
        databricks_base, "_get_databricks_credentials"
    ) as mock_get_credentials:
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
