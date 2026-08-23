"""Tests for the credential management endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app
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


def test_update_credential_answers_404_when_the_credential_does_not_exist():
    """Regression: the handler used to ``return handle_exception_on_proxy(e)``, which makes
    the exception the response body and lets FastAPI answer 200, so a write the handler
    rejected read as a success to every caller that checks the status. The dashboard's API
    client branches on the status, so it reported a failed edit as applied."""
    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), patch(
        "litellm.proxy.credential_endpoints.endpoints.CredentialsRepository"
    ) as repository:
        repository.return_value.find_by_name = AsyncMock(return_value=None)

        response = _patch_credential(
            "definitely-not-there",
            {"credential_name": "definitely-not-there", "credential_values": {"api_key": "sk-x"}, "credential_info": {}},
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
    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), patch(
        "litellm.proxy.proxy_server.master_key", "sk-test-master"
    ), patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository") as repository:
        repository.return_value.find_by_name = AsyncMock(return_value=stored)
        repository.return_value.update_by_name = AsyncMock(return_value=None)

        response = _patch_credential(
            "existing",
            {"credential_name": "existing", "credential_values": {"api_key": "sk-new"}, "credential_info": {}},
        )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
