"""Regression tests for credential CRUD vs concurrent reload race condition.

Issue: https://github.com/BerriAI/litellm/issues/50273

A credential create, update or delete that commits while the periodic reload is running
could be undone in that pod's memory until the next reload. The DB was correct but memory
was wrong. The fix wraps DB write + memory update in MODEL_RECONCILE_LOCK, the same way
model delete already does.
"""

import asyncio
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app
from litellm.types.utils import CredentialItem


def _as_admin():
    return UserAPIKeyAuth(
        api_key="test-key",
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


def _mock_prisma():
    prisma = MagicMock()
    prisma.db = MagicMock()
    return prisma


@pytest.fixture(autouse=True)
def _reset_credential_list():
    """Reset litellm.credential_list before and after each test."""
    original = litellm.credential_list.copy()
    litellm.credential_list.clear()
    yield
    litellm.credential_list.clear()
    litellm.credential_list.extend(original)


@pytest.mark.asyncio
async def test_create_credential_with_concurrent_reload_does_not_lose_credential():
    """Regression: credential create racing a reload's read-modify-write was undone in memory.

    Before the fix, this test would fail because the reload's upsert_credentials would overwrite
    the create's upsert_credentials, leaving the new credential missing from litellm.credential_list.
    """
    from litellm.proxy.credential_endpoints.endpoints import create_credential
    from litellm.proxy.proxy_server import MODEL_RECONCILE_LOCK, ProxyConfig

    reload_snapshot_taken: Final = asyncio.Event()
    reload_can_complete: Final = asyncio.Event()

    async def mock_find_all():
        reload_snapshot_taken.set()
        await reload_can_complete.wait()
        return []

    credentials_repository = MagicMock()
    credentials_repository.return_value.find_all = mock_find_all
    credentials_repository.return_value.create = AsyncMock(return_value=None)

    proxy_config = ProxyConfig()
    prisma_client = _mock_prisma()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
        patch("litellm.proxy.proxy_server.master_key", "sk-test-master"),
        patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository", credentials_repository),
        patch("litellm.proxy.common_utils.credential_hydration.CredentialsRepository", credentials_repository),
    ):
        app.dependency_overrides[user_api_key_auth] = _as_admin

        async def concurrent_reload():
            async with MODEL_RECONCILE_LOCK:
                await proxy_config.get_credentials(prisma_client)

        reload_task: Final = asyncio.create_task(concurrent_reload())
        await reload_snapshot_taken.wait()

        request = MagicMock()
        response = MagicMock()
        credential = MagicMock()
        credential.credential_name = "new-cred"
        credential.display_name = None
        credential.model_id = None
        credential.credential_values = {"api_key": "sk-new"}
        credential.credential_info = {"custom_llm_provider": "openai"}

        create_task: Final = asyncio.create_task(create_credential(request, response, credential, _as_admin(), None))

        await asyncio.sleep(0.01)
        reload_can_complete.set()
        await reload_task
        result: Final = await create_task

        app.dependency_overrides.pop(user_api_key_auth, None)

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
    from litellm.proxy.proxy_server import MODEL_RECONCILE_LOCK, ProxyConfig

    existing_cred: Final = CredentialItem(
        credential_name="existing-cred",
        display_name="Old Name",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai"},
    )
    litellm.credential_list.append(existing_cred)

    reload_snapshot_taken: Final = asyncio.Event()
    reload_can_complete: Final = asyncio.Event()

    async def mock_find_all():
        reload_snapshot_taken.set()
        await reload_can_complete.wait()
        return [existing_cred]

    credentials_repository = MagicMock()
    credentials_repository.return_value.find_all = mock_find_all
    credentials_repository.return_value.find_by_name = AsyncMock(
        return_value=MagicMock(
            credential_name="existing-cred",
            display_name="Old Name",
            credential_values={"api_key": "encrypted-old"},
            credential_info={"custom_llm_provider": "openai"},
        )
    )
    credentials_repository.return_value.update_by_name = AsyncMock(return_value=None)

    proxy_config = ProxyConfig()
    prisma_client = _mock_prisma()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
        patch("litellm.proxy.proxy_server.master_key", "sk-test-master"),
        patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository", credentials_repository),
        patch("litellm.proxy.common_utils.credential_hydration.CredentialsRepository", credentials_repository),
        patch(
            "litellm.proxy.credential_endpoints.endpoints.decrypt_value_helper",
            side_effect=lambda x, **kw: x.replace("encrypted-", ""),
        ),
        patch(
            "litellm.proxy.credential_endpoints.endpoints.encrypt_value_helper",
            side_effect=lambda x, *a, **kw: f"encrypted-{x}",
        ),
        patch("litellm.proxy.proxy_server.ProxyConfig.decrypt_credentials", side_effect=lambda x: x),
    ):
        app.dependency_overrides[user_api_key_auth] = _as_admin

        async def concurrent_reload():
            async with MODEL_RECONCILE_LOCK:
                await proxy_config.get_credentials(prisma_client)

        reload_task: Final = asyncio.create_task(concurrent_reload())
        await reload_snapshot_taken.wait()

        request = MagicMock()
        response = MagicMock()
        credential = MagicMock()
        credential.credential_name = None
        credential.display_name = "New Name"
        credential.model_id = None
        credential.credential_values = {"api_key": "sk-updated"}
        credential.credential_info = {}
        credential.credential_values_to_delete = None
        credential.model_fields_set = {"display_name", "credential_values"}

        update_task: Final = asyncio.create_task(
            update_credential(request, response, credential, "existing-cred", _as_admin(), None)
        )

        await asyncio.sleep(0.01)
        reload_can_complete.set()
        await reload_task
        result: Final = await update_task

        app.dependency_overrides.pop(user_api_key_auth, None)

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
    from litellm.proxy.proxy_server import MODEL_RECONCILE_LOCK, ProxyConfig

    deleted_cred: Final = CredentialItem(
        credential_name="to-delete",
        display_name="Will Be Deleted",
        credential_values={"api_key": "sk-deleted"},
        credential_info={"custom_llm_provider": "openai"},
    )
    litellm.credential_list.append(deleted_cred)

    reload_snapshot_taken: Final = asyncio.Event()
    reload_can_complete: Final = asyncio.Event()

    async def mock_find_all():
        reload_snapshot_taken.set()
        await reload_can_complete.wait()
        return [deleted_cred]

    credentials_repository = MagicMock()
    credentials_repository.return_value.find_all = mock_find_all
    credentials_repository.return_value.delete_by_name = AsyncMock(return_value=deleted_cred)

    proxy_config = ProxyConfig()
    prisma_client = _mock_prisma()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
        patch("litellm.proxy.proxy_server.master_key", "sk-test-master"),
        patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository", credentials_repository),
        patch("litellm.proxy.common_utils.credential_hydration.CredentialsRepository", credentials_repository),
        patch("litellm.proxy.proxy_server.ProxyConfig.decrypt_credentials", side_effect=lambda x: x),
        patch(
            "litellm.proxy.credential_endpoints.endpoints._purge_user_connections_for_credential",
            new_callable=AsyncMock,
        ),
    ):
        app.dependency_overrides[user_api_key_auth] = _as_admin

        async def concurrent_reload():
            async with MODEL_RECONCILE_LOCK:
                await proxy_config.get_credentials(prisma_client)

        reload_task: Final = asyncio.create_task(concurrent_reload())
        await reload_snapshot_taken.wait()

        request = MagicMock()
        response = MagicMock()

        delete_task: Final = asyncio.create_task(delete_credential(request, response, "to-delete", _as_admin()))

        await asyncio.sleep(0.01)
        reload_can_complete.set()
        await reload_task
        result: Final = await delete_task

        app.dependency_overrides.pop(user_api_key_auth, None)

    assert result == {"success": True, "message": "Credential deleted successfully"}
    assert CredentialAccessor.find_credential("to-delete") is None, (
        "Credential was deleted but restored after reload, indicating the race condition"
    )
