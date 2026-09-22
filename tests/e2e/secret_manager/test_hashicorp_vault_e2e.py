"""Live e2e: the proxy reads deployment credentials from, and stores virtual keys in,
a real HashiCorp Vault.

Runs against a proxy booted from tests/e2e/gateway/secret_manager_vault_ci_config.yml,
which sets `key_management_system: hashicorp_vault` with read-and-write access and
virtual-key storage under VIRTUAL_KEY_PREFIX. Deselected unless
E2E_SECRET_MANAGER_VAULT is set, because the default stack runs no secret manager.

Every secret a test seeds is named with a fresh marker, and the proxy's own
environment never holds it, so a deployment can only get its key from Vault:
get_secret falls back to os.environ when the manager errors, and a name the proxy
has never seen cannot be rescued by that fallback. The runner, not the proxy,
holds OPENAI_API_KEY; the tests copy it into Vault, so a passing call proves the
value travelled through the manager.
"""

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
from vault_client import Vault

pytestmark = [pytest.mark.e2e, pytest.mark.secret_manager_vault]

BACKEND_MODEL: Final = "openai/gpt-4o-mini"
# Mirrors key_management_settings.prefix_for_stored_virtual_keys in the lane's config.
VIRTUAL_KEY_PREFIX: Final = "litellm-e2e/virtual-keys/"
PROVIDER_KEY_ENV: Final = "OPENAI_API_KEY"


def _provider_key() -> str:
    key: Final = os.environ.get(PROVIDER_KEY_ENV, "").strip()
    if not key:
        pytest.fail(f"The Vault suite seeds Vault with the runner's {PROVIDER_KEY_ENV}, which is unset")
    return key


def _seed(vault: Vault, resources: ResourceManager, value: str) -> str:
    """Write `value` under a fresh secret name, destroyed on teardown, and return the name."""
    name: Final = f"litellm-e2e-openai-{unique_marker()}"
    vault.write(name, value)
    resources.defer(lambda: vault.destroy(name))
    return name


def _deploy(proxy: ProxyClient, resources: ResourceManager, secret_name: str) -> str:
    """Register a deployment whose api_key is the Vault secret, and return its model name.
    provider_live keeps it off the provider cache, which would answer without the key."""
    model_name: Final = f"vault-backed-{unique_marker()}"
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
    """Poll `read` until it returns `expected`: the proxy writes to Vault from its key
    management hooks, which need not have finished when the key route answers."""
    deadline: Final = time.monotonic() + proxy.poll_timeout
    last: str | None = read()
    while last != expected and time.monotonic() < deadline:
        time.sleep(proxy.poll_interval)
        last = read()
    if last != expected:
        pytest.fail(f"{context}: Vault still holds {'a value' if last is not None else 'nothing'} after the deadline")


class TestHashicorpVaultSecretManager:
    @pytest.mark.covers("other.config.secret_resolution.kms_integration")
    def test_deployment_key_resolves_from_vault(
        self, proxy: ProxyClient, resources: ResourceManager, vault: Vault, scoped_key: str
    ) -> None:
        model: Final = _deploy(proxy, resources, _seed(vault, resources, _provider_key()))

        response: Final = unwrap(_chat(proxy, scoped_key, model))

        assert response.choices, f"the Vault-backed deployment answered with no choices: {response}"

    @pytest.mark.covers("other.config.secret_resolution.manager_value_used")
    def test_deployment_uses_the_value_vault_holds(
        self, proxy: ProxyClient, resources: ResourceManager, vault: Vault, scoped_key: str
    ) -> None:
        bogus: Final = f"sk-litellm-e2e-not-a-key-{unique_marker()}"
        model: Final = _deploy(proxy, resources, _seed(vault, resources, bogus))

        result: Final = _chat(proxy, scoped_key, model)

        match result:
            case UnauthorizedError(body=body):
                assert "AuthenticationError" in body, f"the 401 did not come from the provider: {body[:300]}"
            case Success():
                pytest.fail("a deployment whose Vault secret is not a real key still reached the provider")
            case _:
                pytest.fail(f"expected the provider to reject the Vault-held key with 401, got {result}")

    @pytest.mark.covers("other.config.secret_manager.virtual_key_stored")
    def test_generated_key_is_written_to_vault(
        self, proxy: ProxyClient, resources: ResourceManager, vault: Vault
    ) -> None:
        alias: Final = f"litellm-e2e-vk-{unique_marker()}"
        secret_name: Final = f"{VIRTUAL_KEY_PREFIX}{alias}"
        resources.defer(lambda: vault.destroy(secret_name))
        key: Final = proxy.generate_key(KeyGenerateBody(key_alias=alias))
        resources.defer(lambda: proxy.delete_key(key))

        _eventually(proxy, lambda: vault.read(secret_name), key, f"the generated key {alias}")

    @pytest.mark.covers("other.config.secret_manager.virtual_key_deleted")
    def test_deleted_key_is_removed_from_vault(
        self, proxy: ProxyClient, resources: ResourceManager, vault: Vault
    ) -> None:
        alias: Final = f"litellm-e2e-vk-{unique_marker()}"
        secret_name: Final = f"{VIRTUAL_KEY_PREFIX}{alias}"
        resources.defer(lambda: vault.destroy(secret_name))
        key: Final = proxy.generate_key(KeyGenerateBody(key_alias=alias))
        resources.defer(lambda: proxy.delete_key(key))
        _eventually(proxy, lambda: vault.read(secret_name), key, f"the generated key {alias}")

        proxy.delete_key(key)

        _eventually(proxy, lambda: vault.read(secret_name), None, f"the deleted key {alias}")
