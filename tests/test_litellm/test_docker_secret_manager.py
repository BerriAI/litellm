"""
Unit tests for DockerSecretsManager.

Uses pytest's `tmp_path` fixture to create real temporary files so no file-I/O
mocking is required.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.secret_managers import docker_secret_manager
from litellm.secret_managers.docker_secret_manager import DockerSecretsManager
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_manager(tmp_path) -> DockerSecretsManager:
    """Return a DockerSecretsManager pointed at *tmp_path*."""
    return DockerSecretsManager(secrets_dir=str(tmp_path))


def write_secret(tmp_path, name: str, value: str) -> None:
    """Write *value* to a file named *name* inside *tmp_path*."""
    secret_file = tmp_path / name
    secret_file.write_text(value)


# ---------------------------------------------------------------------------
# sync_read_secret
# ---------------------------------------------------------------------------


def test_sync_read_secret_existing_file(tmp_path):
    """sync_read_secret returns the file content when the secret exists."""
    write_secret(tmp_path, "MY_API_KEY", "sk-abc123")
    manager = make_manager(tmp_path)

    result = manager.sync_read_secret("MY_API_KEY")

    assert result == "sk-abc123"


def test_sync_read_secret_missing_file(tmp_path):
    """sync_read_secret returns None when the secret file does not exist."""
    manager = make_manager(tmp_path)

    result = manager.sync_read_secret("NONEXISTENT_SECRET")

    assert result is None


def test_sync_read_secret_strips_trailing_newline(tmp_path):
    """sync_read_secret strips the trailing newline Docker appends to secret files."""
    # Docker always adds a trailing newline; verify we strip it.
    write_secret(tmp_path, "DB_PASSWORD", "p@ssw0rd\n")
    manager = make_manager(tmp_path)

    result = manager.sync_read_secret("DB_PASSWORD")

    assert result == "p@ssw0rd"


def test_sync_read_secret_strips_surrounding_whitespace(tmp_path):
    """sync_read_secret strips all surrounding whitespace, not just newlines."""
    write_secret(tmp_path, "TOKEN", "  mytoken  \n")
    manager = make_manager(tmp_path)

    result = manager.sync_read_secret("TOKEN")

    assert result == "mytoken"


# ---------------------------------------------------------------------------
# async_read_secret
# ---------------------------------------------------------------------------


def test_async_read_secret_existing_file(tmp_path):
    """async_read_secret returns the file content when the secret exists."""
    write_secret(tmp_path, "ASYNC_KEY", "async-value-xyz")
    manager = make_manager(tmp_path)

    result = asyncio.run(manager.async_read_secret("ASYNC_KEY"))

    assert result == "async-value-xyz"


def test_async_read_secret_missing_file(tmp_path):
    """async_read_secret returns None when the secret file does not exist."""
    manager = make_manager(tmp_path)

    result = asyncio.run(manager.async_read_secret("NO_SUCH_SECRET"))

    assert result is None


def test_async_read_secret_strips_whitespace(tmp_path):
    """async_read_secret strips trailing whitespace just like the sync variant."""
    write_secret(tmp_path, "SECRET_WITH_NEWLINE", "value\n")
    manager = make_manager(tmp_path)

    result = asyncio.run(manager.async_read_secret("SECRET_WITH_NEWLINE"))

    assert result == "value"


# ---------------------------------------------------------------------------
# Custom secrets_dir
# ---------------------------------------------------------------------------


def test_custom_secrets_dir(tmp_path):
    """DockerSecretsManager respects a custom secrets_dir."""
    custom_dir = tmp_path / "custom_secrets"
    custom_dir.mkdir()
    (custom_dir / "CUSTOM_SECRET").write_text("custom-value")

    manager = DockerSecretsManager(secrets_dir=str(custom_dir))

    assert manager.sync_read_secret("CUSTOM_SECRET") == "custom-value"
    # A file in the default (non-custom) directory should NOT be found.
    assert manager.sync_read_secret("MISSING") is None


def test_secrets_dir_from_key_management_settings(tmp_path):
    """secrets_dir in KeyManagementSettings is honoured when constructing DockerSecretsManager."""
    (tmp_path / "MY_KEY").write_text("from-settings")
    settings = KeyManagementSettings(secrets_dir=str(tmp_path))

    manager = DockerSecretsManager(secrets_dir=settings.secrets_dir)

    assert manager.sync_read_secret("MY_KEY") == "from-settings"
    assert manager.secrets_dir == str(tmp_path)


def test_default_secrets_dir_non_windows(monkeypatch):
    """Without an explicit directory, non-Windows defaults to /run/secrets."""
    monkeypatch.setattr(docker_secret_manager.os, "name", "posix")

    manager = DockerSecretsManager()

    assert manager.secrets_dir == "/run/secrets"


def test_default_secrets_dir_windows(monkeypatch):
    """Without an explicit directory, Windows defaults to ProgramData Docker secrets."""
    monkeypatch.setattr(docker_secret_manager.os, "name", "nt")

    manager = DockerSecretsManager()

    assert manager.secrets_dir == r"C:\ProgramData\Docker\secrets"


# ---------------------------------------------------------------------------
# Registration: __init__ wires up litellm globals
# ---------------------------------------------------------------------------


def test_init_registers_as_secret_manager_client(tmp_path):
    """Constructing DockerSecretsManager sets litellm.secret_manager_client."""
    manager = make_manager(tmp_path)

    assert litellm.secret_manager_client is manager
    assert litellm._key_management_system == KeyManagementSystem.DOCKER_SECRET_MANAGER


# ---------------------------------------------------------------------------
# Unsupported operations return "not_supported" status
# ---------------------------------------------------------------------------


def test_async_write_secret_not_supported(tmp_path):
    """async_write_secret returns a not_supported status dict."""
    manager = make_manager(tmp_path)

    result = asyncio.run(manager.async_write_secret("KEY", "value"))

    assert result["status"] == "not_supported"


def test_async_delete_secret_not_supported(tmp_path):
    """async_delete_secret returns a not_supported status dict."""
    manager = make_manager(tmp_path)

    result = asyncio.run(manager.async_delete_secret("KEY"))

    assert result["status"] == "not_supported"
