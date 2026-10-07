import asyncio
import importlib
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest

import litellm
from litellm.proxy._types import KeyManagementSystem
from litellm.secret_managers.main import _should_read_secret_from_secret_manager, get_secret
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


class MockSecretClient:
    def get_secret(self, secret_name):
        return Mock(value="mocked_secret_value")


@pytest.mark.asyncio
async def test_azure_kms():
    """
    Basic asserts that the value from get secret is from Azure Key Vault when Key Management System is Azure Key Vault
    """
    with patch("litellm.secret_manager_client", new=MockSecretClient()):
        litellm._key_management_system = KeyManagementSystem.AZURE_KEY_VAULT
        secret = get_secret(secret_name="ishaan-test-key")
        assert secret == "mocked_secret_value"


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    importlib.reload(litellm)
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)


def redact_oidc_signature(secret_val: str) -> list[str]:
    return secret_val.split(".")[:-1] + ["SIGNATURE_REMOVED"]


def test_oidc_env_variable(monkeypatch):
    env_var_name = "OIDC_TEST_PATH_" + uuid4().hex
    monkeypatch.setenv(env_var_name, "secret-" + uuid4().hex)
    secret_val = get_secret(f"oidc/env/{env_var_name}")

    print(f"secret_val: {redact_oidc_signature(secret_val)}")

    assert secret_val == os.environ[env_var_name]


def test_oidc_file(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", temp_dir)
        temp_file_path = os.path.join(temp_dir, "token.txt")
        secret_value = "secret-" + uuid4().hex
        with open(temp_file_path, "w") as temp_file:
            temp_file.write(secret_value)

        secret_val = get_secret(f"oidc/file/{temp_file_path}")

        print(f"secret_val: {redact_oidc_signature(secret_val)}")

        assert secret_val == secret_value


def test_oidc_env_path(monkeypatch):
    with tempfile.NamedTemporaryFile(mode="w+") as temp_file:
        secret_value = "secret-" + uuid4().hex
        temp_file.write(secret_value)
        temp_file.flush()
        temp_file_path = temp_file.name

        env_var_name = "OIDC_TEST_PATH_" + uuid4().hex

        monkeypatch.setenv(env_var_name, temp_file_path)

        secret_val = get_secret(f"oidc/env_path/{env_var_name}")

        print(f"secret_val: {redact_oidc_signature(secret_val)}")

        assert secret_val == secret_value


def test_should_read_secret_from_secret_manager(monkeypatch):
    """
    Test that _should_read_secret_from_secret_manager returns correct values based on access mode
    """
    from litellm.types.secret_managers.main import KeyManagementSettings

    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings())
    assert _should_read_secret_from_secret_manager() is False

    monkeypatch.setattr(litellm, "secret_manager_client", "dummy_client")
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))
    assert _should_read_secret_from_secret_manager() is True

    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_and_write"))
    assert _should_read_secret_from_secret_manager() is True

    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="write_only"))
    assert _should_read_secret_from_secret_manager() is False

    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings())


def test_get_secret_with_access_mode(monkeypatch):
    """
    Test that get_secret respects access mode settings
    """
    from litellm.types.secret_managers.main import KeyManagementSettings

    test_secret_name = "TEST_SECRET_KEY"
    test_secret_value = "test_secret_value"
    monkeypatch.setenv(test_secret_name, test_secret_value)

    monkeypatch.setattr(litellm, "secret_manager_client", "dummy_client")
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="write_only"))
    assert get_secret(test_secret_name) == test_secret_value

    monkeypatch.setattr(litellm, "secret_manager_client", "dummy_client")
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings())
    assert _should_read_secret_from_secret_manager() is True

    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))
    assert _should_read_secret_from_secret_manager() is True

    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_and_write"))
    assert _should_read_secret_from_secret_manager() is True

    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings())


def test_key_management_settings_defaults():
    """
    Test that KeyManagementSettings initializes with correct default values.
    """
    from litellm.types.secret_managers.main import KeyManagementSettings

    settings = KeyManagementSettings()

    assert settings.store_virtual_keys is False
    assert settings.prefix_for_stored_virtual_keys == "litellm/"
    assert settings.access_mode == "read_only"
    assert settings.description is None
    assert settings.tags is None
    assert settings.primary_secret_name is None


def test_key_management_settings_custom_values():
    """
    Test that KeyManagementSettings correctly stores custom description and tags.
    """
    from litellm.types.secret_managers.main import KeyManagementSettings

    custom_tags = {"Environment": "Dev", "Team": "Intelligence"}
    custom_description = "LiteLLM-managed API key for development"

    settings = KeyManagementSettings(
        store_virtual_keys=True,
        prefix_for_stored_virtual_keys="litellm/custom/",
        access_mode="read_and_write",
        primary_secret_name="primary/litellm/keys",
        description=custom_description,
        tags=custom_tags,
    )

    assert settings.store_virtual_keys is True
    assert settings.prefix_for_stored_virtual_keys == "litellm/custom/"
    assert settings.access_mode == "read_and_write"
    assert settings.primary_secret_name == "primary/litellm/keys"
    assert settings.description == custom_description
    assert settings.tags == custom_tags


@pytest.mark.asyncio
async def test_async_write_secret_receives_description_and_tags(monkeypatch):
    """
    Test that AWSSecretsManagerV2.async_write_secret receives description and tags when KeyManagementSettings is set.
    """
    from litellm import litellm
    from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
    from litellm.types.secret_managers.main import KeyManagementSettings

    mock_async_write = AsyncMock(return_value={"Name": "litellm/test_secret"})
    secret_manager_client = MagicMock(spec=AWSSecretsManagerV2)
    secret_manager_client.async_write_secret = mock_async_write

    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(
            store_virtual_keys=True,
            description="LiteLLM Unit Test Secret",
            tags={"Owner": "UnitTest", "Purpose": "Validation"},
        ),
    )

    monkeypatch.setattr(litellm, "secret_manager_client", secret_manager_client)

    from litellm.proxy.hooks.key_management_event_hooks import (
        KeyManagementEventHooks,
    )

    await KeyManagementEventHooks._store_virtual_key_in_secret_manager(
        secret_name="test_secret", secret_token="test_value"
    )

    mock_async_write.assert_called_once()
    args, kwargs = mock_async_write.call_args

    assert kwargs["secret_name"].endswith("test_secret")
    assert kwargs["secret_value"] == "test_value"
    assert kwargs["description"] == "LiteLLM Unit Test Secret"
    assert kwargs["tags"] == {"Owner": "UnitTest", "Purpose": "Validation"}


def test_key_management_settings_serialization_roundtrip():
    """
    Test that KeyManagementSettings serializes and deserializes consistently (Pydantic behavior).
    """
    from litellm.types.secret_managers.main import KeyManagementSettings

    original = KeyManagementSettings(
        store_virtual_keys=True,
        prefix_for_stored_virtual_keys="litellm/dev/",
        access_mode="read_and_write",
        description="Roundtrip test",
        tags={"Env": "QA"},
    )

    as_dict = original.model_dump()
    reloaded = KeyManagementSettings(**as_dict)

    assert reloaded.store_virtual_keys is True
    assert reloaded.prefix_for_stored_virtual_keys == "litellm/dev/"
    assert reloaded.access_mode == "read_and_write"
    assert reloaded.description == "Roundtrip test"
    assert reloaded.tags == {"Env": "QA"}
