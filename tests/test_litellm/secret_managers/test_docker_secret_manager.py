"""
Unit + end-to-end tests for DockerSecretManager.

The tests simulate a Docker secrets mount by writing temporary files under a
temp directory, so no actual Docker daemon is required.  The E2E section at the
bottom tests the full get_secret() call path through LiteLLM globals, mirroring
how the proxy server would use the manager in production.
"""

import os
import sys
import tempfile
from typing import Optional, Union
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.secret_managers.docker_secret_manager import (
    DockerSecretManager,
    _name_candidates,
)
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def secrets_dir(tmp_path):
    """A temporary directory that mimics /run/secrets."""
    return tmp_path


@pytest.fixture()
def populated_secrets_dir(secrets_dir):
    """Temp secrets directory pre-populated with several secret files."""
    (secrets_dir / "openai_api_key").write_text("sk-test-openai-key")
    (secrets_dir / "anthropic_api_key").write_text("sk-ant-test-key")
    (secrets_dir / "db_password").write_text("super-secret-db-pass\n")  # trailing newline
    (secrets_dir / "multiline_secret").write_text("line1\nline2\nline3")
    (secrets_dir / "UPPER_CASE_SECRET").write_text("upper-value")
    return secrets_dir


@pytest.fixture()
def manager(populated_secrets_dir):
    m = DockerSecretManager(secrets_path=str(populated_secrets_dir))
    yield m
    # cleanup litellm globals
    litellm.secret_manager_client = None
    litellm._key_management_system = None
    litellm._key_management_settings = None


# ---------------------------------------------------------------------------
# _name_candidates
# ---------------------------------------------------------------------------


def test_name_candidates_uppercase_yields_lowercase_fallback():
    candidates = list(_name_candidates("OPENAI_API_KEY"))
    assert candidates == ["OPENAI_API_KEY", "openai_api_key"]


def test_name_candidates_already_lowercase_no_duplicate():
    candidates = list(_name_candidates("openai_api_key"))
    assert candidates == ["openai_api_key"]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_sets_litellm_globals(populated_secrets_dir):
    m = DockerSecretManager(secrets_path=str(populated_secrets_dir))
    assert litellm.secret_manager_client is m
    assert litellm._key_management_system == KeyManagementSystem.DOCKER
    # cleanup
    litellm.secret_manager_client = None
    litellm._key_management_system = None


def test_init_warns_when_dir_missing(tmp_path, caplog):
    missing = str(tmp_path / "nonexistent")
    import logging
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        DockerSecretManager(secrets_path=missing)
    assert any("does not exist" in r.message for r in caplog.records)
    # cleanup
    litellm.secret_manager_client = None
    litellm._key_management_system = None


# ---------------------------------------------------------------------------
# sync_read_secret
# ---------------------------------------------------------------------------


def test_sync_read_exact_name(manager):
    assert manager.sync_read_secret("openai_api_key") == "sk-test-openai-key"


def test_sync_read_uppercase_name_falls_back_to_lowercase(manager):
    # File is named openai_api_key (lower), lookup is OPENAI_API_KEY (upper)
    assert manager.sync_read_secret("OPENAI_API_KEY") == "sk-test-openai-key"


def test_sync_read_uppercase_file_exact_match(manager):
    # File is named UPPER_CASE_SECRET; exact match should be tried first
    assert manager.sync_read_secret("UPPER_CASE_SECRET") == "upper-value"


def test_sync_read_strips_trailing_newline(manager):
    # db_password file was written with a trailing \n
    assert manager.sync_read_secret("db_password") == "super-secret-db-pass"


def test_sync_read_multiline_secret_preserved(manager):
    # Internal newlines should be kept; only trailing whitespace stripped
    assert manager.sync_read_secret("multiline_secret") == "line1\nline2\nline3"


def test_sync_read_missing_secret_returns_none(manager):
    assert manager.sync_read_secret("DOES_NOT_EXIST") is None


def test_sync_read_missing_secret_no_raise(manager):
    """Missing secrets must NOT raise — get_secret() relies on None for fallback."""
    result = manager.sync_read_secret("TOTALLY_MISSING")
    assert result is None


# ---------------------------------------------------------------------------
# async_read_secret
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_read_delegates_to_sync(manager):
    result = await manager.async_read_secret("anthropic_api_key")
    assert result == "sk-ant-test-key"


@pytest.mark.asyncio
async def test_async_read_missing_returns_none(manager):
    result = await manager.async_read_secret("NOT_HERE")
    assert result is None


# ---------------------------------------------------------------------------
# write / delete raise NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_raises(manager):
    with pytest.raises(NotImplementedError, match="read-only"):
        await manager.async_write_secret("KEY", "value")


@pytest.mark.asyncio
async def test_delete_raises(manager):
    with pytest.raises(NotImplementedError, match="read-only"):
        await manager.async_delete_secret("KEY")


# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------


def test_path_traversal_via_dotdot_raises(manager):
    # '../../etc/passwd' contains '/' so the path-separator check fires first
    with pytest.raises(ValueError, match="path separator|path traversal"):
        manager.sync_read_secret("../../etc/passwd")


def test_path_separator_in_name_raises(manager):
    """Names with '/' are rejected before realpath resolution."""
    with pytest.raises(ValueError, match="path separator"):
        manager.sync_read_secret("subdir/MY_SECRET")


def test_path_traversal_via_absolute_path_raises(manager, tmp_path):
    # An absolute path as secret name should be caught
    outside_file = tmp_path.parent / "outside_secret"
    outside_file.write_text("sensitive")
    with pytest.raises(ValueError, match="path traversal|path separator"):
        manager.sync_read_secret(str(outside_file))


def test_path_traversal_via_symlink_is_safe(manager, tmp_path):
    """A symlink that points outside the secrets dir must be blocked."""
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside-value")
    symlink = tmp_path / "evil_link"
    symlink.symlink_to(outside)
    with pytest.raises(ValueError, match="path traversal"):
        manager.sync_read_secret("evil_link")


# ---------------------------------------------------------------------------
# Permission / IO errors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.getuid() == 0, reason="root can read chmod 000 files; test is meaningless as root")
def test_unreadable_file_returns_none(manager, populated_secrets_dir):
    secret_file = populated_secrets_dir / "openai_api_key"
    original_mode = secret_file.stat().st_mode
    try:
        secret_file.chmod(0o000)
        result = manager.sync_read_secret("openai_api_key")
        assert result is None
    finally:
        secret_file.chmod(original_mode)


def test_latin1_secret_decoded_correctly(manager, populated_secrets_dir):
    """Passwords with Latin-1 chars (é, ñ, ü) must be returned, not dropped."""
    latin1_file = populated_secrets_dir / "latin1_secret"
    latin1_file.write_bytes("pässwörd".encode("latin-1"))
    result = manager.sync_read_secret("latin1_secret")
    assert result == "pässwörd"


# ---------------------------------------------------------------------------
# load_docker_secret_manager factory
# ---------------------------------------------------------------------------


def test_load_factory_default_path(populated_secrets_dir):
    settings = KeyManagementSettings(
        access_mode="read_only",
        secrets_path=str(populated_secrets_dir),
    )
    m = DockerSecretManager.load_docker_secret_manager(key_management_settings=settings)
    assert m.secrets_path == str(populated_secrets_dir)
    assert isinstance(m, DockerSecretManager)
    # cleanup
    litellm.secret_manager_client = None
    litellm._key_management_system = None


def test_load_factory_no_settings(tmp_path):
    """Factory with no settings uses the default /run/secrets path."""
    m = DockerSecretManager.load_docker_secret_manager(key_management_settings=None)
    assert m.secrets_path == "/run/secrets"
    # cleanup
    litellm.secret_manager_client = None


def test_load_factory_secrets_path_none_uses_default():
    """
    Regression: KeyManagementSettings.secrets_path is Optional[str] defaulting
    to None. When an operator omits secrets_path from their config the attribute
    exists on the object but evaluates to None, so a bare getattr() returns None
    rather than the fallback string — causing os.path.realpath(None) to raise
    TypeError. The factory must guard against this.
    """
    settings = KeyManagementSettings(access_mode="read_only")  # secrets_path omitted
    assert settings.secrets_path is None  # confirm the field is actually None

    # Must not raise TypeError
    m = DockerSecretManager.load_docker_secret_manager(key_management_settings=settings)
    assert m.secrets_path == "/run/secrets"
    # cleanup
    litellm.secret_manager_client = None
    litellm._key_management_system = None
    litellm._key_management_system = None


# ---------------------------------------------------------------------------
# E2E: full get_secret() call path
# ---------------------------------------------------------------------------


def test_e2e_get_secret_reads_from_docker(populated_secrets_dir):
    """
    End-to-end: DockerSecretManager is set as the LiteLLM secret manager,
    and get_secret() retrieves a value from the simulated /run/secrets directory.
    """
    from litellm.secret_managers.main import get_secret

    manager = DockerSecretManager(secrets_path=str(populated_secrets_dir))
    litellm._key_management_settings = KeyManagementSettings(
        access_mode="read_only",
        secrets_path=str(populated_secrets_dir),
    )

    try:
        result = get_secret("openai_api_key")
        assert result == "sk-test-openai-key"

        result_upper = get_secret("ANTHROPIC_API_KEY")
        assert result_upper == "sk-ant-test-key"
    finally:
        litellm.secret_manager_client = None
        litellm._key_management_system = None
        litellm._key_management_settings = None


def test_e2e_get_secret_returns_none_when_not_in_docker(populated_secrets_dir, monkeypatch):
    """
    End-to-end: when a secret is not in the Docker secrets dir, get_secret()
    returns None.  get_secret() only falls back to env vars on exceptions, not
    on None returns — intentional so operators aren't silently served stale env vars.
    """
    from litellm.secret_managers.main import get_secret

    monkeypatch.setenv("MY_FALLBACK_KEY", "env-fallback-value")

    manager = DockerSecretManager(secrets_path=str(populated_secrets_dir))
    litellm._key_management_settings = KeyManagementSettings(
        access_mode="read_only",
        secrets_path=str(populated_secrets_dir),
    )

    try:
        result = get_secret("MY_FALLBACK_KEY")
        # Not in /run/secrets → None; env var is NOT silently substituted
        assert result is None
    finally:
        litellm.secret_manager_client = None
        litellm._key_management_system = None
        litellm._key_management_settings = None


def test_e2e_get_secret_trailing_newline_stripped(populated_secrets_dir):
    """
    End-to-end: secrets created with `echo` (which adds \\n) must have
    trailing whitespace stripped before being returned.
    """
    from litellm.secret_managers.main import get_secret

    manager = DockerSecretManager(secrets_path=str(populated_secrets_dir))
    litellm._key_management_settings = KeyManagementSettings(
        access_mode="read_only",
        secrets_path=str(populated_secrets_dir),
    )

    try:
        result = get_secret("db_password")
        assert result == "super-secret-db-pass"
        assert not result.endswith("\n")
    finally:
        litellm.secret_manager_client = None
        litellm._key_management_system = None
        litellm._key_management_settings = None


def test_e2e_get_secret_write_only_mode_skips_docker(populated_secrets_dir, monkeypatch):
    """
    End-to-end: when access_mode is write_only, get_secret() must skip the
    secret manager and read from the environment instead.
    """
    from litellm.secret_managers.main import get_secret

    monkeypatch.setenv("openai_api_key", "env-value-not-docker")

    manager = DockerSecretManager(secrets_path=str(populated_secrets_dir))
    litellm._key_management_settings = KeyManagementSettings(
        access_mode="write_only",
        secrets_path=str(populated_secrets_dir),
    )

    try:
        result = get_secret("openai_api_key")
        # write_only mode: secret manager is bypassed, env var returned
        assert result == "env-value-not-docker"
    finally:
        litellm.secret_manager_client = None
        litellm._key_management_system = None
        litellm._key_management_settings = None


def test_e2e_key_management_system_enum_registered():
    """Ensure the DOCKER enum value is accessible and correct."""
    assert KeyManagementSystem.DOCKER.value == "docker"


