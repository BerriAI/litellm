from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Final

import pytest

from e2e_config import unique_marker
from e2e_http import Result, Success, UnauthorizedError, unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, KeyGenerateBody, LiteLLMParamsBody
from proxy_client import ProxyClient
from secret_store import SecretStore

pytestmark = [pytest.mark.e2e, pytest.mark.secret_manager]

BACKEND_MODEL: Final = "openai/gpt-4o-mini"
VIRTUAL_KEY_PREFIX: Final = "litellm-e2e/virtual-keys/"
PROVIDER_KEY_ENV: Final = "OPENAI_API_KEY"


# The proxy's env never holds OPENAI_API_KEY and each test seeds it under a fresh name, so a passing
# call proves the key came from the manager and not get_secret's os.environ fallback.
def _provider_key() -> str:
    key: Final = os.environ.get(PROVIDER_KEY_ENV, "").strip()
    if not key:
        pytest.fail(f"The secret manager suite seeds the manager with the runner's {PROVIDER_KEY_ENV}, which is unset")
    return key


def _seed(store: SecretStore, resources: ResourceManager, value: str) -> str:
    name: Final = f"litellm-e2e-openai-{unique_marker()}"
    store.write(name, value)
    resources.defer(lambda: store.destroy(name))
    return name


def _deploy(proxy: ProxyClient, resources: ResourceManager, secret_name: str) -> str:
    model_name: Final = f"secret-manager-backed-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model_name,
        LiteLLMParamsBody(model=BACKEND_MODEL, api_key=f"os.environ/{secret_name}"),
        provider_live=True,
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model_name


def _chat(proxy: ProxyClient, key: str, model: str) -> Result[ChatResponse]:
    return proxy.chat(
        key,
        ChatBody(
            model=model,
            messages=[ChatMessage(role="user", content=f"reply with one word {unique_marker()}")],
            max_tokens=16,
        ),
    )


def _eventually(proxy: ProxyClient, read: Callable[[], str | None], expected: str | None, context: str) -> None:
    deadline: Final = time.monotonic() + proxy.poll_timeout
    last: str | None = read()
    while last != expected and time.monotonic() < deadline:
        time.sleep(proxy.poll_interval)
        last = read()
    if last != expected:
        pytest.fail(
            f"{context}: the secret manager still holds {'a value' if last is not None else 'nothing'} after the deadline"
        )


class TestSecretManager:
    @pytest.mark.covers("other.config.secret_resolution.kms_integration")
    def test_deployment_key_resolves_from_the_manager(
        self, proxy: ProxyClient, resources: ResourceManager, store: SecretStore, scoped_key: str
    ) -> None:
        model: Final = _deploy(proxy, resources, _seed(store, resources, _provider_key()))

        response: Final = unwrap(_chat(proxy, scoped_key, model))

        assert response.choices, f"the manager-backed deployment answered with no choices: {response}"

    @pytest.mark.covers("other.config.secret_resolution.manager_value_used")
    def test_deployment_uses_the_value_the_manager_holds(
        self, proxy: ProxyClient, resources: ResourceManager, store: SecretStore, scoped_key: str
    ) -> None:
        bogus: Final = f"sk-litellm-e2e-not-a-key-{unique_marker()}"
        model: Final = _deploy(proxy, resources, _seed(store, resources, bogus))

        result: Final = _chat(proxy, scoped_key, model)

        match result:
            case UnauthorizedError(body=body):
                assert "AuthenticationError" in body, f"the 401 did not come from the provider: {body[:300]}"
            case Success():
                pytest.fail("a deployment whose managed secret is not a real key still reached the provider")
            case _:
                pytest.fail(f"expected the provider to reject the manager-held key with 401, got {result}")

    @pytest.mark.covers("other.config.secret_manager.virtual_key_stored")
    def test_generated_key_is_written_to_the_manager(
        self, proxy: ProxyClient, resources: ResourceManager, store: SecretStore
    ) -> None:
        alias: Final = f"litellm-e2e-vk-{unique_marker()}"
        secret_name: Final = f"{VIRTUAL_KEY_PREFIX}{alias}"
        resources.defer(lambda: store.destroy(secret_name))
        key: Final = proxy.generate_key(KeyGenerateBody(key_alias=alias))
        resources.defer(lambda: proxy.delete_key(key))

        _eventually(proxy, lambda: store.read(secret_name), key, f"the generated key {alias}")

    @pytest.mark.requires_capability("deletes_stored_keys")
    @pytest.mark.covers("other.config.secret_manager.virtual_key_deleted")
    def test_deleted_key_is_removed_from_the_manager(
        self, proxy: ProxyClient, resources: ResourceManager, store: SecretStore
    ) -> None:
        alias: Final = f"litellm-e2e-vk-{unique_marker()}"
        secret_name: Final = f"{VIRTUAL_KEY_PREFIX}{alias}"
        resources.defer(lambda: store.destroy(secret_name))
        key: Final = proxy.generate_key(KeyGenerateBody(key_alias=alias))
        resources.defer(lambda: proxy.delete_key(key))
        _eventually(proxy, lambda: store.read(secret_name), key, f"the generated key {alias}")

        proxy.delete_key(key)

        _eventually(proxy, lambda: store.read(secret_name), None, f"the deleted key {alias}")
