"""Regression tests for credential CRUD vs concurrent reload race condition.

Issue: https://github.com/BerriAI/litellm/issues/50273

A credential create, update or delete that commits while the periodic reload is running
could be undone in that pod's memory until the next reload. The DB was correct but memory
was wrong. The fix wraps DB write + memory update in MODEL_RECONCILE_LOCK, the same way
model delete already does.
"""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response

import litellm
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.models.credentials import CreateCredentialItem, UpdateCredentialItem
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import ProxyConfig
from litellm.types.utils import CredentialItem

_SNAPSHOT_WAIT_SECONDS: Final = 2
_LOCK_YIELD_ITERS: Final = 50


def _as_admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="test-key",
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


def _mock_prisma() -> MagicMock:
    prisma = MagicMock()
    prisma.db = MagicMock()
    return prisma


def _request_response() -> tuple[MagicMock, MagicMock]:
    return MagicMock(spec=Request), MagicMock(spec=Response)


@pytest.fixture(autouse=True)
def _reset_credential_list():
    original = litellm.credential_list.copy()
    litellm.credential_list.clear()
    yield
    litellm.credential_list.clear()
    litellm.credential_list.extend(original)


@contextmanager
def _repository_with_gated_find_all(
    *,
    snapshot: asyncio.Event,
    gate: asyncio.Event,
    find_all_result: list[CredentialItem],
    find_by_name: CredentialItem | None = None,
    delete_by_name: CredentialItem | None = None,
) -> Iterator[asyncio.Lock]:
    async def mock_find_all() -> list[CredentialItem]:
        snapshot.set()
        await gate.wait()
        return find_all_result

    repository = MagicMock()
    repository.return_value.find_all = mock_find_all
    repository.return_value.find_by_name = AsyncMock(return_value=find_by_name)
    repository.return_value.create = AsyncMock(return_value=None)
    repository.return_value.update_by_name = AsyncMock(return_value=None)
    repository.return_value.delete_by_name = AsyncMock(return_value=delete_by_name)
    lock = asyncio.Lock()  # fresh per test: asyncio.Lock binds to the first loop that acquires it
    with (
        patch("litellm.proxy.proxy_server.prisma_client", _mock_prisma()),
        patch("litellm.proxy.proxy_server.master_key", "sk-test-master"),
        patch("litellm.proxy.proxy_server.MODEL_RECONCILE_LOCK", lock),
        patch("litellm.proxy.proxy_server.CredentialsRepository", repository),
        patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository", repository),
        patch("litellm.proxy.common_utils.credential_hydration.CredentialsRepository", repository),
        patch.object(ProxyConfig, "get_config", AsyncMock(return_value={})),
        patch.object(ProxyConfig, "decrypt_credentials", side_effect=lambda cred: cred),
    ):
        yield lock


async def _reload_until_snapshot(snapshot: asyncio.Event, lock: asyncio.Lock) -> asyncio.Task[object]:
    proxy_config: Final = ProxyConfig()

    async def concurrent_reload() -> object:
        async with lock:
            return await proxy_config.get_credentials(_mock_prisma())

    reload_task: Final = asyncio.create_task(concurrent_reload())
    await asyncio.wait_for(snapshot.wait(), timeout=_SNAPSHOT_WAIT_SECONDS)
    return reload_task


async def _await_write_blocked_on_lock(write_task: asyncio.Task[object]) -> None:
    for _ in range(_LOCK_YIELD_ITERS):
        if not write_task.done():
            await asyncio.sleep(0)
            continue
        await write_task
        raise AssertionError("credential write finished while the reload still held the lock")
    assert not write_task.done()


@pytest.mark.asyncio
async def test_create_credential_with_concurrent_reload_does_not_lose_credential():
    """Regression: credential create racing a reload's read-modify-write was undone in memory.

    Before the fix, this test would fail because the reload's upsert_credentials would overwrite
    the create's upsert_credentials, leaving the new credential missing from litellm.credential_list.
    """
    from litellm.proxy.credential_endpoints.endpoints import create_credential

    snapshot: Final = asyncio.Event()
    gate: Final = asyncio.Event()
    request, response = _request_response()
    credential: Final = CreateCredentialItem(
        credential_name="new-cred",
        credential_values={"api_key": "sk-new"},
        credential_info={"custom_llm_provider": "openai"},
    )

    with _repository_with_gated_find_all(snapshot=snapshot, gate=gate, find_all_result=[]) as lock:
        reload_task: Final = await _reload_until_snapshot(snapshot, lock)
        create_task: Final = asyncio.create_task(create_credential(request, response, credential, _as_admin(), None))
        await _await_write_blocked_on_lock(create_task)
        gate.set()
        await reload_task
        result: Final = await create_task

    assert result == {"success": True, "message": "Credential created successfully"}
    assert CredentialAccessor.find_credential("new-cred") is not None, (
        "Credential was created but lost after reload, indicating the race condition"
    )


@pytest.mark.asyncio
async def test_update_credential_with_concurrent_reload_preserves_update():
    """Regression: credential update racing a reload's read-modify-write was overwritten in memory.

    Before the fix, this test would fail because the reload's upsert_credentials would restore
    the old values, leaving the update missing from litellm.credential_list.
    """
    from litellm.proxy.credential_endpoints.endpoints import update_credential

    existing_cred: Final = CredentialItem(
        credential_name="existing-cred",
        display_name="Old Name",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai"},
    )
    litellm.credential_list.append(existing_cred)

    snapshot: Final = asyncio.Event()
    gate: Final = asyncio.Event()
    request, response = _request_response()
    credential: Final = UpdateCredentialItem(
        display_name="New Name",
        credential_values={"api_key": "sk-updated"},
        credential_info={},
    )

    with _repository_with_gated_find_all(
        snapshot=snapshot,
        gate=gate,
        find_all_result=[existing_cred],
        find_by_name=existing_cred,
    ) as lock:
        reload_task: Final = await _reload_until_snapshot(snapshot, lock)
        update_task: Final = asyncio.create_task(
            update_credential(request, response, credential, "existing-cred", _as_admin(), None)
        )
        await _await_write_blocked_on_lock(update_task)
        gate.set()
        await reload_task
        result: Final = await update_task

    assert result == {"success": True, "message": "Credential updated successfully"}
    found: Final = CredentialAccessor.find_credential("existing-cred")
    assert found is not None
    assert found.display_name == "New Name", (
        "Credential was updated but reverted after reload, indicating the race condition"
    )


@pytest.mark.asyncio
async def test_delete_credential_with_concurrent_reload_stays_deleted():
    """Regression: credential delete racing a reload's read-modify-write was restored in memory.

    Before the fix, this test would fail because the reload's upsert_credentials would restore
    the deleted credential back into litellm.credential_list.
    """
    from litellm.proxy.credential_endpoints.endpoints import delete_credential

    deleted_cred: Final = CredentialItem(
        credential_name="to-delete",
        display_name="Will Be Deleted",
        credential_values={"api_key": "sk-deleted"},
        credential_info={"custom_llm_provider": "openai"},
    )
    litellm.credential_list.append(deleted_cred)

    snapshot: Final = asyncio.Event()
    gate: Final = asyncio.Event()
    request, response = _request_response()

    with (
        _repository_with_gated_find_all(
            snapshot=snapshot,
            gate=gate,
            find_all_result=[deleted_cred],
            delete_by_name=deleted_cred,
        ) as lock,
        patch(
            "litellm.proxy.credential_endpoints.endpoints._purge_user_connections_for_credential",
            new_callable=AsyncMock,
        ),
    ):
        reload_task: Final = await _reload_until_snapshot(snapshot, lock)
        delete_task: Final = asyncio.create_task(delete_credential(request, response, "to-delete", _as_admin()))
        await _await_write_blocked_on_lock(delete_task)
        gate.set()
        await reload_task
        result: Final = await delete_task

    assert result == {"success": True, "message": "Credential deleted successfully"}
    assert CredentialAccessor.find_credential("to-delete") is None, (
        "Credential was deleted but restored after reload, indicating the race condition"
    )
