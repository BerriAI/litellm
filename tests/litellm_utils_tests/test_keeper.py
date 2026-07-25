"""
Unit tests for the Keeper Secrets Manager (KSM) read-only integration.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.secret_managers.keeper_secret_manager import KeeperSecretManager
from litellm.secret_managers.secret_manager_handler import get_secret_from_manager
from litellm.types.secret_managers.main import KeyManagementSystem


def _make_manager(results):
    client = MagicMock()
    client.get_notation_results.return_value = results
    return KeeperSecretManager(client=client), client


def test_keeper_enum_value():
    assert KeyManagementSystem.KEEPER.value == "keeper"


def test_sync_read_returns_first_notation_value():
    manager, client = _make_manager(["sk-openai-123"])

    value = manager.sync_read_secret("RECORD_UID/field/password")

    assert value == "sk-openai-123"
    client.get_notation_results.assert_called_once_with("RECORD_UID/field/password")


def test_sync_read_caches_and_avoids_second_call():
    manager, client = _make_manager(["sk-openai-123"])

    first = manager.sync_read_secret("RECORD_UID/field/password")
    second = manager.sync_read_secret("RECORD_UID/field/password")

    assert first == second == "sk-openai-123"
    client.get_notation_results.assert_called_once()


@pytest.mark.asyncio
async def test_async_read_returns_value():
    manager, client = _make_manager(["sk-gemini-abc"])

    value = await manager.async_read_secret("RECORD_UID/field/password")

    assert value == "sk-gemini-abc"
    client.get_notation_results.assert_called_once_with("RECORD_UID/field/password")


def test_sync_read_missing_secret_returns_none():
    manager, client = _make_manager([])

    assert manager.sync_read_secret("RECORD_UID/field/password") is None


def test_sync_read_swallows_client_errors():
    client = MagicMock()
    client.get_notation_results.side_effect = RuntimeError("boom")
    manager = KeeperSecretManager(client=client)

    assert manager.sync_read_secret("RECORD_UID/field/password") is None


@pytest.mark.asyncio
async def test_write_and_delete_are_read_only():
    manager, _ = _make_manager(["v"])

    write = await manager.async_write_secret(secret_name="x", secret_value="y")
    delete = await manager.async_delete_secret(secret_name="x")

    assert write["status"] == "not_supported"
    assert delete["status"] == "not_supported"


def test_missing_credentials_raise():
    for var in ("KSM_CONFIG", "KSM_TOKEN"):
        os.environ.pop(var, None)

    with pytest.raises(ValueError, match="Missing Keeper Secrets Manager credentials"):
        KeeperSecretManager()


def test_handler_dispatches_keeper_read():
    manager, client = _make_manager(["sk-from-keeper"])

    secret = get_secret_from_manager(
        client=manager,
        key_manager=KeyManagementSystem.KEEPER.value,
        secret_name="RECORD_UID/field/password",
    )

    assert secret == "sk-from-keeper"
    client.get_notation_results.assert_called_once_with("RECORD_UID/field/password")


def test_handler_raises_when_keeper_secret_missing():
    manager, _ = _make_manager([])

    with pytest.raises(ValueError, match="No secret found in Keeper Secrets Manager"):
        get_secret_from_manager(
            client=manager,
            key_manager=KeyManagementSystem.KEEPER.value,
            secret_name="RECORD_UID/field/password",
        )
