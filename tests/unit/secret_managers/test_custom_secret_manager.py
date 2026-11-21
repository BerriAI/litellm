"""
Test custom secret manager implementation
"""

import os
from typing import Optional, Union
from unittest.mock import MagicMock

import httpx
import pytest


import litellm
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.secret_managers.custom_secret_manager_loader import load_custom_secret_manager
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem


class TestCustomSecretManager(CustomSecretManager):
    """
    Test implementation of a custom secret manager.

    This simulates a simple in-memory secret store for testing purposes.
    """

    __test__ = False

    def __init__(self, secrets: Optional[dict] = None):
        super().__init__(secret_manager_name="test_custom_secret_manager")
        self.secrets = secrets or {
            "TEST_API_KEY": "test-api-key-12345",
            "TEST_SECRET": "super-secret-value",
            "OPENAI_API_KEY": "sk-test-openai-key",
        }

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Read a secret from the in-memory store asynchronously.
        """
        return self.secrets.get(secret_name)

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Read a secret from the in-memory store synchronously.
        """
        return self.secrets.get(secret_name)

    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: Optional[str] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        tags: Optional[Union[dict, list]] = None,
    ) -> dict:
        """
        Write a secret to the in-memory store.
        """
        self.secrets[secret_name] = secret_value
        return {
            "secret_name": secret_name,
            "status": "success",
            "description": description,
        }

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """
        Delete a secret from the in-memory store.
        """
        if secret_name in self.secrets:
            del self.secrets[secret_name]
            return {"secret_name": secret_name, "status": "deleted"}
        return {"secret_name": secret_name, "status": "not_found"}


def test_custom_secret_manager_initialization():
    """
    Test that a custom secret manager can be initialized correctly.
    """
    secret_manager = TestCustomSecretManager()
    assert secret_manager.secret_manager_name == "test_custom_secret_manager"
    assert "TEST_API_KEY" in secret_manager.secrets


def test_custom_secret_manager_sync_read():
    """
    Test synchronous secret reading.
    """
    secret_manager = TestCustomSecretManager()

    # Test reading an existing secret
    api_key = secret_manager.sync_read_secret("TEST_API_KEY")
    assert api_key == "test-api-key-12345"

    # Test reading a non-existent secret
    missing_key = secret_manager.sync_read_secret("NON_EXISTENT_KEY")
    assert missing_key is None


@pytest.mark.asyncio
async def test_custom_secret_manager_async_read():
    """
    Test asynchronous secret reading.
    """
    secret_manager = TestCustomSecretManager()

    # Test reading an existing secret
    api_key = await secret_manager.async_read_secret("TEST_SECRET")
    assert api_key == "super-secret-value"

    # Test reading a non-existent secret
    missing_key = await secret_manager.async_read_secret("NON_EXISTENT_KEY")
    assert missing_key is None


@pytest.mark.asyncio
async def test_custom_secret_manager_async_write():
    """
    Test asynchronous secret writing.
    """
    secret_manager = TestCustomSecretManager()

    # Write a new secret
    result = await secret_manager.async_write_secret(
        secret_name="NEW_SECRET",
        secret_value="new-secret-value",
        description="A new test secret",
    )

    assert result["status"] == "success"
    assert result["secret_name"] == "NEW_SECRET"

    # Verify the secret was written
    secret_value = secret_manager.sync_read_secret("NEW_SECRET")
    assert secret_value == "new-secret-value"


@pytest.mark.asyncio
async def test_custom_secret_manager_async_delete():
    """
    Test asynchronous secret deletion.
    """
    secret_manager = TestCustomSecretManager()

    # Verify secret exists
    assert secret_manager.sync_read_secret("TEST_API_KEY") is not None

    # Delete the secret
    result = await secret_manager.async_delete_secret("TEST_API_KEY")
    assert result["status"] == "deleted"

    # Verify secret is deleted
    assert secret_manager.sync_read_secret("TEST_API_KEY") is None


def test_custom_secret_manager_integration_with_litellm(monkeypatch):
    """
    Test that the custom secret manager integrates with LiteLLM's secret management system.
    """
    secret_manager = TestCustomSecretManager()
    monkeypatch.setattr(litellm, "secret_manager_client", secret_manager)
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.CUSTOM)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))

    from litellm.secret_managers.main import get_secret

    api_key = get_secret("TEST_API_KEY")
    assert api_key == "test-api-key-12345"

    openai_key = get_secret("OPENAI_API_KEY")
    assert openai_key == "sk-test-openai-key"


class MinimalCustomSecretManager(CustomSecretManager):
    """
    Minimal implementation that only implements required methods.
    """

    def __init__(self):
        super().__init__(secret_manager_name="minimal_custom")

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Minimal async read implementation.
        """
        return f"async-{secret_name}-value"

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Minimal sync read implementation.
        """
        return f"sync-{secret_name}-value"


def test_minimal_custom_secret_manager():
    """
    Test that a minimal implementation without write/delete methods works.
    """
    secret_manager = MinimalCustomSecretManager()

    # Read should work
    value = secret_manager.sync_read_secret("TEST_KEY")
    assert value == "sync-TEST_KEY-value"

    # Write should raise NotImplementedError
    import asyncio

    with pytest.raises(NotImplementedError) as exc_info:
        asyncio.run(secret_manager.async_write_secret("KEY", "value"))

    assert "Write operations are not implemented" in str(exc_info.value)

    # Delete should raise NotImplementedError
    import asyncio

    with pytest.raises(NotImplementedError) as exc_info:
        asyncio.run(secret_manager.async_delete_secret("KEY"))

    assert "Delete operations are not implemented" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests for load_custom_secret_manager
# ---------------------------------------------------------------------------


def test_load_custom_secret_manager_config_validation(monkeypatch):
    """
    Test that missing config_file_path or key_management_settings raises ValueError.
    """
    with pytest.raises(ValueError, match="config_file_path is required to load custom secret manager"):
        load_custom_secret_manager(None)

    monkeypatch.setattr(litellm, "_key_management_settings", None)
    with pytest.raises(ValueError, match="key_management_settings is required with custom_secret_manager field"):
        load_custom_secret_manager("config.yaml")

    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(custom_secret_manager=None))
    with pytest.raises(ValueError, match="custom_secret_manager field is required in key_management_settings"):
        load_custom_secret_manager("config.yaml")

    monkeypatch.setattr(
        litellm, "_key_management_settings", KeyManagementSettings(custom_secret_manager="InvalidModuleNoDot")
    )
    with pytest.raises(ValueError, match="CustomSecretManagerException - Invalid module path"):
        load_custom_secret_manager("config.yaml")


def test_load_custom_secret_manager_multi_segment_module(tmp_path, monkeypatch):
    """
    Test that module paths with multiple segments (e.g. foo.bar.MySecretManager)
    are handled properly without unpack error.
    """
    module_code = """
from litellm.integrations.custom_secret_manager import CustomSecretManager

class MultiSegmentSecretManager(CustomSecretManager):
    def __init__(self):
        super().__init__(secret_manager_name="multi_segment_manager")

    async def async_read_secret(self, secret_name: str, **kwargs):
        return "secret"

    def sync_read_secret(self, secret_name: str, **kwargs):
        return "secret"
"""
    module_file = tmp_path / "foo.bar.py"
    module_file.write_text(module_code)

    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(custom_secret_manager="foo.bar.MultiSegmentSecretManager"),
    )
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_system", None)

    load_custom_secret_manager(config_file_path=config_path)
    assert litellm.secret_manager_client is not None
    assert litellm.secret_manager_client.secret_manager_name == "multi_segment_manager"
    assert litellm._key_management_system == KeyManagementSystem.CUSTOM
    assert isinstance(litellm.secret_manager_client, CustomSecretManager)


def test_load_custom_secret_manager_package_directory_layout(tmp_path, monkeypatch):
    """
    Test that real package layout (pkg/submodule.py) resolves and loads properly.
    """
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    module_code = """
from litellm.integrations.custom_secret_manager import CustomSecretManager

class PackageSecretManager(CustomSecretManager):
    def __init__(self):
        super().__init__(secret_manager_name="pkg_manager")

    async def async_read_secret(self, secret_name: str, **kwargs):
        return "pkg_secret"

    def sync_read_secret(self, secret_name: str, **kwargs):
        return "pkg_secret"
"""
    (pkg_dir / "submodule.py").write_text(module_code)

    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(custom_secret_manager="pkg.submodule.PackageSecretManager"),
    )
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_system", None)

    load_custom_secret_manager(config_file_path=config_path)
    assert litellm.secret_manager_client is not None
    assert litellm.secret_manager_client.secret_manager_name == "pkg_manager"
    assert litellm._key_management_system == KeyManagementSystem.CUSTOM
    assert isinstance(litellm.secret_manager_client, CustomSecretManager)


def test_load_custom_secret_manager_package_init_layout(tmp_path, monkeypatch):
    """
    Test that package __init__.py layout (pkg/submodule/__init__.py) resolves and loads properly.
    """
    sub_dir = tmp_path / "pkg" / "submodule"
    sub_dir.mkdir(parents=True)
    module_code = """
from litellm.integrations.custom_secret_manager import CustomSecretManager

class InitSecretManager(CustomSecretManager):
    def __init__(self):
        super().__init__(secret_manager_name="init_manager")

    async def async_read_secret(self, secret_name: str, **kwargs):
        return "init_secret"

    def sync_read_secret(self, secret_name: str, **kwargs):
        return "init_secret"
"""
    (sub_dir / "__init__.py").write_text(module_code)

    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(custom_secret_manager="pkg.submodule.InitSecretManager"),
    )
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_system", None)

    load_custom_secret_manager(config_file_path=config_path)
    assert litellm.secret_manager_client is not None
    assert litellm.secret_manager_client.secret_manager_name == "init_manager"
    assert litellm._key_management_system == KeyManagementSystem.CUSTOM
    assert isinstance(litellm.secret_manager_client, CustomSecretManager)


@pytest.mark.parametrize(
    "bad_path",
    [
        "../../bad.MyClass",
        "../bad.MyClass",
        "foo/../../bad.MyClass",
        r"..\..\bad.MyClass",
        r"foo\..\bad.MyClass",
        "/etc/passwd.MyClass",
        r"\windows\system32\calc.MyClass",
        ".MyClass",
        "my_module.",
        "pkg..submodule.MyClass",
        "foo...bar.MyClass",
        "..pkg.MyClass",
    ],
)
def test_load_custom_secret_manager_rejects_path_traversal(tmp_path, monkeypatch, bad_path):
    """
    Test that path traversal attempts and invalid module paths are blocked with ValueError.
    """
    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(custom_secret_manager=bad_path),
    )
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_system", None)

    with pytest.raises(ValueError, match="CustomSecretManagerException - Invalid module path"):
        load_custom_secret_manager(config_file_path=config_path)


def test_load_custom_secret_manager_missing_class(tmp_path, monkeypatch):
    """
    Test that requesting a class that does not exist in the loaded module raises AttributeError.
    """
    module_code = """
from litellm.integrations.custom_secret_manager import CustomSecretManager

class RealSecretManager(CustomSecretManager):
    def __init__(self):
        super().__init__(secret_manager_name="real")
"""
    module_file = tmp_path / "secret_mod.py"
    module_file.write_text(module_code)

    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(custom_secret_manager="secret_mod.NonExistentClass"),
    )
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_system", None)

    with pytest.raises(AttributeError, match="CustomSecretManagerException - Class NonExistentClass not found"):
        load_custom_secret_manager(config_file_path=config_path)


def test_load_custom_secret_manager_missing_loader(tmp_path, monkeypatch):
    """
    Test that when importlib spec has no loader, an ImportError is raised.
    """
    module_file = tmp_path / "dummy_loader.py"
    module_file.write_text("# dummy")

    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(custom_secret_manager="dummy_loader.SomeClass"),
    )
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_system", None)

    import importlib.util

    mock_spec = MagicMock()
    mock_spec.loader = None
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *args, **kwargs: mock_spec)

    with pytest.raises(ImportError, match="Could not find a module specification"):
        load_custom_secret_manager(config_file_path=config_path)


def test_load_custom_secret_manager_missing_spec(tmp_path, monkeypatch):
    """
    Test that when spec_from_file_location returns None, an ImportError is raised.
    """
    module_file = tmp_path / "dummy_spec.py"
    module_file.write_text("# dummy")

    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(custom_secret_manager="dummy_spec.SomeClass"),
    )
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_system", None)

    import importlib.util

    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *args, **kwargs: None)

    with pytest.raises(ImportError, match="Could not find a module specification"):
        load_custom_secret_manager(config_file_path=config_path)


def test_load_custom_secret_manager_not_a_type(tmp_path, monkeypatch):
    """
    Test that when the target attribute in the module is not a type (e.g. string, object),
    TypeError is raised before calling issubclass.
    """
    module_code = """
NotAClass = "I am a string, not a class"
"""
    module_file = tmp_path / "not_type_mod.py"
    module_file.write_text(module_code)

    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(custom_secret_manager="not_type_mod.NotAClass"),
    )
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_system", None)

    with pytest.raises(TypeError, match="must be a subclass of CustomSecretManager"):
        load_custom_secret_manager(config_file_path=config_path)


def test_load_custom_secret_manager_invalid_subclass(tmp_path, monkeypatch):
    """
    Test that when the class does not inherit from CustomSecretManager, TypeError is raised.
    """
    module_code = """
class UnrelatedClass:
    pass
"""
    module_file = tmp_path / "unrelated_mod.py"
    module_file.write_text(module_code)

    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(custom_secret_manager="unrelated_mod.UnrelatedClass"),
    )
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_system", None)

    with pytest.raises(TypeError, match="must be a subclass of CustomSecretManager"):
        load_custom_secret_manager(config_file_path=config_path)
