"""Tests for the credential management endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app
from litellm.types.utils import CredentialItem

client = TestClient(app)


def _as_admin():
    return UserAPIKeyAuth(api_key="test-key", user_role="proxy_admin")


def _call_as_admin(method: str, path: str, json_body: dict | None = None):
    missing = object()
    previous_override = app.dependency_overrides.get(user_api_key_auth, missing)
    app.dependency_overrides[user_api_key_auth] = _as_admin
    try:
        return client.request(method, path, json=json_body, headers={"Authorization": "Bearer test-key"})
    finally:
        if previous_override is missing:
            app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            app.dependency_overrides[user_api_key_auth] = previous_override


def _patch_credential(name: str, body: dict):
    return _call_as_admin("PATCH", f"/credentials/{name}", body)


def _delete_credential(name: str):
    return _call_as_admin("DELETE", f"/credentials/{name}")


def _list_credentials():
    return _call_as_admin("GET", "/credentials")


@pytest.fixture
def credential_store():
    """Stands the credential store up for one test: whether the database is reachable, what
    the proxy is already serving from memory, and what each repository call hands back."""

    def install(
        *,
        connected: bool = True,
        in_memory: tuple[object, ...] = (),
        **repository_calls: AsyncMock,
    ) -> None:
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock() if connected else None).start()
        patch("litellm.proxy.proxy_server.master_key", "sk-test-master").start()
        patch.object(litellm, "credential_list", list(in_memory)).start()
        repository = patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository").start()
        for call_name, result in repository_calls.items():
            setattr(repository.return_value, call_name, result)

    yield install
    patch.stopall()


def test_update_credential_answers_404_when_the_credential_does_not_exist(credential_store):
    """Regression: the handler used to ``return handle_exception_on_proxy(e)``, which makes
    the exception the response body and lets FastAPI answer 200, so a write the handler
    rejected read as a success to every caller that checks the status. The dashboard's API
    client branches on the status, so it reported a failed edit as applied."""
    credential_store(find_by_name=AsyncMock(return_value=None))

    response = _patch_credential(
        "definitely-not-there",
        {"credential_name": "definitely-not-there", "credential_values": {"api_key": "sk-x"}, "credential_info": {}},
    )

    assert response.status_code == 404, f"rejected write answered {response.status_code}: {response.text}"
    assert "error" in response.json()


def test_update_credential_answers_500_when_the_database_is_not_connected(credential_store):
    """The other rejection this handler raises must carry its own status too."""
    credential_store(connected=False)

    response = _patch_credential(
        "any-name",
        {"credential_name": "any-name", "credential_values": {"api_key": "sk-x"}, "credential_info": {}},
    )

    assert response.status_code == 500, f"rejected write answered {response.status_code}: {response.text}"


def test_update_credential_still_answers_200_on_a_successful_write(credential_store):
    """The fix must not turn a legitimate update into an error; the dashboard and the
    Playwright credentials spec both assert the success path."""
    stored = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai"},
    )
    credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=AsyncMock(return_value=None))

    response = _patch_credential(
        "existing",
        {"credential_name": "existing", "credential_values": {"api_key": "sk-new"}, "credential_info": {}},
    )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True


def test_delete_credential_answers_404_when_the_credential_does_not_exist(credential_store):
    """Regression: prisma's ``delete`` hands back None when the ``where`` clause matched no row
    instead of raising, and the handler never looked. Deleting a name that was never stored
    answered 200 "Credential deleted successfully", so an operator scripting cleanup could not
    tell a real deletion from a typo."""
    credential_store(delete_by_name=AsyncMock(return_value=None))

    response = _delete_credential("definitely-not-there")

    assert response.status_code == 404, f"delete of a missing credential answered {response.status_code}: {response.text}"
    assert "definitely-not-there" in response.text


def test_delete_credential_still_answers_200_and_drops_the_credential_from_memory(credential_store):
    """The fix must not turn a real deletion into an error, and the deleted credential must
    stop being served from the in-memory list the proxy routes on."""
    stored = CredentialItem(
        credential_name="doomed",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai"},
    )
    survivor = CredentialItem(
        credential_name="keeper",
        credential_values={"api_key": "sk-keep"},
        credential_info={},
    )
    credential_store(in_memory=(stored, survivor), delete_by_name=AsyncMock(return_value=MagicMock()))

    response = _delete_credential("doomed")

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    assert [credential.credential_name for credential in litellm.credential_list] == ["keeper"]


def test_delete_credential_leaves_a_credential_that_only_exists_in_memory_in_place(credential_store):
    """A credential declared in the config yaml is never written to the table, so the delete
    matches no row. Reporting success would be the same lie: it comes straight back on the next
    proxy boot. ``PATCH /credentials/{name}`` already answers 404 for that credential."""
    config_only = CredentialItem(
        credential_name="from-config-yaml",
        credential_values={"api_key": "sk-config"},
        credential_info={},
    )
    credential_store(in_memory=(config_only,), delete_by_name=AsyncMock(return_value=None))

    response = _delete_credential("from-config-yaml")

    assert response.status_code == 404, response.text
    assert [credential.credential_name for credential in litellm.credential_list] == ["from-config-yaml"]


def test_delete_credential_answers_500_when_the_database_is_not_connected(credential_store):
    """The handler used to ``return handle_exception_on_proxy(e)``, which makes the exception the
    response body and lets FastAPI answer 200. A DB-less proxy answered its own 500 as a success."""
    credential_store(connected=False)

    response = _delete_credential("any-name")

    assert response.status_code == 500, f"rejected delete answered {response.status_code}: {response.text}"


class _CredentialThatCannotBeMasked:
    """Stands in for anything that fails while ``GET /credentials`` builds its response."""

    credential_name = "unreadable"
    credential_info: dict = {}

    @property
    def credential_values(self):
        raise RuntimeError("credential store unreadable")


def test_get_credentials_answers_an_error_status_when_the_listing_fails(credential_store):
    """Same ``return`` instead of ``raise`` on the list route: a failed listing was serialized as
    a 200 whose body happened to be an error, so a caller reading the status saw an empty success."""
    credential_store(in_memory=(_CredentialThatCannotBeMasked(),))

    response = _list_credentials()

    assert response.status_code == 500, f"failed listing answered {response.status_code}: {response.text}"
    assert response.json().get("success") is not True
