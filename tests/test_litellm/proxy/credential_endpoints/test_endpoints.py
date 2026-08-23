"""Tests for the credential management endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.credential_endpoints.endpoints import _create_credential_record, _get_credential_list
from litellm.proxy.proxy_server import app
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.types.utils import CredentialItem

client = TestClient(app)


def _as_admin():
    return UserAPIKeyAuth(api_key="test-key", user_role="proxy_admin")


def _patch_credential(name: str, body: dict):
    missing = object()
    previous_override = app.dependency_overrides.get(user_api_key_auth, missing)
    app.dependency_overrides[user_api_key_auth] = _as_admin
    try:
        return client.patch(
            f"/credentials/{name}",
            json=body,
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        if previous_override is missing:
            app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            app.dependency_overrides[user_api_key_auth] = previous_override


def _post_credential(body: dict, credential_list: tuple[CredentialItem, ...]):
    missing = object()
    previous_auth_override = app.dependency_overrides.get(user_api_key_auth, missing)
    previous_list_override = app.dependency_overrides.get(_get_credential_list, missing)
    app.dependency_overrides[user_api_key_auth] = _as_admin
    app.dependency_overrides[_get_credential_list] = lambda: credential_list
    try:
        return client.post(
            "/credentials",
            json=body,
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        if previous_auth_override is missing:
            app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            app.dependency_overrides[user_api_key_auth] = previous_auth_override
        if previous_list_override is missing:
            app.dependency_overrides.pop(_get_credential_list, None)
        else:
            app.dependency_overrides[_get_credential_list] = previous_list_override


def test_create_credential_rejects_config_defined_name_collision():
    existing = CredentialItem(
        credential_name="shared-credential",
        credential_values={"api_key": "config-secret"},
        credential_info={"description": "defined in config"},
    )
    response = _post_credential(
        {
            "credential_name": "shared-credential",
            "credential_values": {"api_key": "replacement-secret"},
            "credential_info": {},
        },
        (existing,),
    )

    assert response.status_code == 409, response.text


def test_get_credential_list_returns_the_live_list():
    assert _get_credential_list() is litellm.credential_list


@pytest.mark.asyncio
async def test_create_credential_maps_database_duplicate_race_to_409():
    from prisma.errors import UniqueViolationError

    repository = MagicMock(spec=CredentialsRepository)
    repository.create = AsyncMock(side_effect=UniqueViolationError({}, message="duplicate name"))
    data = {"credential_name": "shared-credential"}

    with pytest.raises(HTTPException) as exc_info:
        await _create_credential_record(
            credentials_repository=repository,
            data=data,
            credential_name="shared-credential",
        )

    assert exc_info.value.status_code == 409
    assert "already exists" in exc_info.value.detail
    repository.create.assert_awaited_once_with(data=data)


def test_update_credential_answers_404_when_the_credential_does_not_exist():
    """Regression: the handler used to ``return handle_exception_on_proxy(e)``, which makes
    the exception the response body and lets FastAPI answer 200, so a write the handler
    rejected read as a success to every caller that checks the status. The dashboard's API
    client branches on the status, so it reported a failed edit as applied."""
    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository") as repository,
    ):
        repository.return_value.find_by_name = AsyncMock(return_value=None)

        response = _patch_credential(
            "definitely-not-there",
            {
                "credential_name": "definitely-not-there",
                "credential_values": {"api_key": "sk-x"},
                "credential_info": {},
            },
        )

    assert response.status_code == 404, f"rejected write answered {response.status_code}: {response.text}"
    assert "error" in response.json()


def test_update_credential_answers_500_when_the_database_is_not_connected():
    """The other rejection this handler raises must carry its own status too."""
    with patch("litellm.proxy.proxy_server.prisma_client", None):
        response = _patch_credential(
            "any-name",
            {"credential_name": "any-name", "credential_values": {"api_key": "sk-x"}, "credential_info": {}},
        )

    assert response.status_code == 500, f"rejected write answered {response.status_code}: {response.text}"


def test_update_credential_still_answers_200_on_a_successful_write():
    """The fix must not turn a legitimate update into an error; the dashboard and the
    Playwright credentials spec both assert the success path."""
    stored = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai"},
    )
    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.master_key", "sk-test-master"),
        patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository") as repository,
    ):
        repository.return_value.find_by_name = AsyncMock(return_value=stored)
        repository.return_value.update_by_name = AsyncMock(return_value=None)

        response = _patch_credential(
            "existing",
            {"credential_name": "existing", "credential_values": {"api_key": "sk-new"}, "credential_info": {}},
        )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
