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


def _create_credential(body: dict):
    return _call_as_admin("POST", "/credentials", body)


def test_create_credential_answers_409_when_the_name_is_already_taken(credential_store):
    """Regression: create used to let the insert hit the unique index and surface Prisma's
    ``Unique constraint failed on the fields: (credential_name)`` as a 500, so every caller
    had to string-match that message to tell a name collision from a real server fault. The
    Terraform provider did exactly that. A taken name is the caller's mistake, so it answers
    409 and names the route that updates the existing credential."""
    stored = CredentialItem(
        credential_name="aws_bedrock",
        credential_values={"aws_access_key_id": "old"},
        credential_info={"custom_llm_provider": "bedrock"},
    )
    create = AsyncMock()
    credential_store(find_by_name=AsyncMock(return_value=stored), create=create)

    response = _create_credential(
        {"credential_name": "aws_bedrock", "credential_values": {"aws_access_key_id": "new"}, "credential_info": {}},
    )

    assert response.status_code == 409, f"name collision answered {response.status_code}: {response.text}"
    detail = response.json()["error"]["message"]
    assert "aws_bedrock" in str(detail)
    assert "PATCH /credentials/aws_bedrock" in str(detail)
    assert "Unique constraint" not in response.text, f"the Prisma internals must not leak: {response.text}"
    create.assert_not_awaited(), "the guard must reject before writing"


def test_create_credential_answers_409_when_a_concurrent_create_wins_the_race(credential_store):
    """The existence check above is advisory: two creates of the same name can both pass it,
    and the loser's insert is what the unique index rejects. That loser must answer the same
    409 as the guard rather than falling through to a 500."""

    class _UniqueViolation(Exception):
        code = "P2002"

    credential_store(
        find_by_name=AsyncMock(return_value=None),
        create=AsyncMock(side_effect=_UniqueViolation("Unique constraint failed on the fields: (`credential_name`)")),
    )

    response = _create_credential(
        {"credential_name": "aws_bedrock", "credential_values": {"aws_access_key_id": "new"}, "credential_info": {}},
    )

    assert response.status_code == 409, f"the losing racer answered {response.status_code}: {response.text}"
    assert "Unique constraint" not in response.text, f"the Prisma internals must not leak: {response.text}"


def test_create_credential_still_answers_500_when_the_write_fails_for_another_reason(credential_store):
    """Only a unique-index collision becomes a 409; a genuine database fault must stay a 500
    so it is not mistaken for a caller error and quietly retried as an update."""
    credential_store(
        find_by_name=AsyncMock(return_value=None),
        create=AsyncMock(side_effect=Exception("connection reset by peer")),
    )

    response = _create_credential(
        {"credential_name": "aws_bedrock", "credential_values": {"aws_access_key_id": "new"}, "credential_info": {}},
    )

    assert response.status_code == 500, f"database fault answered {response.status_code}: {response.text}"


def test_create_credential_still_answers_200_for_a_name_that_is_free(credential_store):
    """The guard must not cost the happy path: a free name still creates."""
    credential_store(find_by_name=AsyncMock(return_value=None), create=AsyncMock(return_value=None))

    response = _create_credential(
        {"credential_name": "brand_new", "credential_values": {"aws_access_key_id": "new"}, "credential_info": {}},
    )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
