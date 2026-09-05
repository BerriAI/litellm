"""Tests for the credential management endpoints."""

import json
from functools import reduce
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import litellm
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.litellm_logging import _get_masked_values
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
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


def _encrypted_row(served: CredentialItem) -> CredentialItem:
    """The table row behind a credential the proxy serves: every value encrypted at rest."""
    return CredentialItem(
        credential_name=served.credential_name,
        credential_values={key: encrypt_value_helper(value) for key, value in served.credential_values.items()},
        credential_info=dict(served.credential_info),
    )


def _written_values(update_by_name: AsyncMock) -> dict:
    written = json.loads(update_by_name.await_args.kwargs["data"]["credential_values"])
    return {key: decrypt_value_helper(value, key) for key, value in written.items()}


def _serve_credential(
    credential_store, served: CredentialItem, in_memory: tuple[CredentialItem, ...] | None = None
) -> AsyncMock:
    """Serves ``served`` from the table; memory holds the same credential unless ``in_memory`` says
    otherwise (a replica that has not resynced yet, or one that never loaded the credential)."""
    update_by_name = AsyncMock(return_value=None)
    credential_store(
        in_memory=(served,) if in_memory is None else in_memory,
        find_by_name=AsyncMock(side_effect=lambda _name: _encrypted_row(served)),
        update_by_name=update_by_name,
    )
    return update_by_name


def test_update_credential_keeps_the_stored_secret_when_the_patch_echoes_its_masked_read_back(credential_store):
    """Regression for #28906: ``GET /credentials/by_name/{name}`` renders ``sk-real-secret-1234`` as
    ``sk****34``, and a caller editing an unrelated field sends that rendering back. The handler
    stored the placeholder as the secret, so every later provider call failed with 401 until the
    key was re-entered."""
    served = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-real-secret-1234"},
        credential_info={"custom_llm_provider": "openai", "description": "original"},
    )
    update_by_name = _serve_credential(credential_store, served)

    response = _patch_credential(
        "existing",
        {
            "credential_name": "existing",
            "credential_values": {"api_key": "sk****34"},
            "credential_info": {"custom_llm_provider": "openai", "description": "edited"},
        },
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == {"api_key": "sk-real-secret-1234"}
    assert json.loads(update_by_name.await_args.kwargs["data"]["credential_info"])["description"] == "edited"
    assert CredentialAccessor.get_credential_values("existing") == {"api_key": "sk-real-secret-1234"}
    assert litellm.credential_list[0].credential_info["description"] == "edited"


def test_update_credential_still_rotates_a_secret_sent_in_full(credential_store):
    """Dropping every sensitive field would also pass the regression above; a real new value
    must still replace the stored one in the table and in memory."""
    served = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-real-secret-1234"},
        credential_info={"custom_llm_provider": "openai"},
    )
    update_by_name = _serve_credential(credential_store, served)

    response = _patch_credential(
        "existing",
        {"credential_name": "existing", "credential_values": {"api_key": "sk-rotated-key-5678"}, "credential_info": {}},
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == {"api_key": "sk-rotated-key-5678"}
    assert CredentialAccessor.get_credential_values("existing") == {"api_key": "sk-rotated-key-5678"}


def test_update_credential_keeps_a_short_secret_whose_read_back_is_all_asterisks(credential_store):
    """Secrets of four characters or fewer render as ``*****`` with no prefix or suffix, so the
    echo check has to match that rendering too, not only the ``ab****yz`` shape."""
    served = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "abc"},
        credential_info={},
    )
    update_by_name = _serve_credential(credential_store, served)

    response = _patch_credential(
        "existing",
        {"credential_name": "existing", "credential_values": {"api_key": "*****"}, "credential_info": {}},
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == {"api_key": "abc"}
    assert CredentialAccessor.get_credential_values("existing") == {"api_key": "abc"}


@pytest.mark.parametrize(
    "rotated_values",
    (
        {"api_key": "new**secret"},
        {"vertex_credentials": {"private_key": "pk**rotated", "client_email": "svc@example.com"}},
    ),
)
def test_update_credential_rotates_a_secret_that_merely_contains_asterisks(credential_store, rotated_values):
    """Only the exact rendering GET serves is a placeholder: ``*****`` or two characters, four
    asterisks, two characters. A secret sent in full rotates even when it carries asterisks of
    its own, at the top level and inside a nested object alike."""
    served = CredentialItem(
        credential_name="existing",
        credential_values={
            "api_key": "sk-old-key-000011",
            "vertex_credentials": {"private_key": "pk-1234567890", "client_email": "svc@example.com"},
        },
        credential_info={},
    )
    update_by_name = _serve_credential(credential_store, served)

    response = _patch_credential(
        "existing",
        {"credential_name": "existing", "credential_values": rotated_values, "credential_info": {}},
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == {**served.credential_values, **rotated_values}
    assert CredentialAccessor.get_credential_values("existing") == {**served.credential_values, **rotated_values}


def test_update_credential_keeps_a_secret_stored_as_a_nested_object(credential_store):
    """Service-account style credentials are stored as an object whose sensitive fields are masked
    one level down, so the echo check compares the masked object, and the row's non-string value
    goes through untouched instead of being fed to the string decryptor."""
    served = CredentialItem(
        credential_name="existing",
        credential_values={"vertex_credentials": {"private_key": "pk-1234567890", "client_email": "svc@example.com"}},
        credential_info={},
    )
    update_by_name = _serve_credential(credential_store, served)

    response = _patch_credential(
        "existing",
        {
            "credential_name": "existing",
            "credential_values": {"vertex_credentials": {"private_key": "pk****90", "client_email": "svc@example.com"}},
            "credential_info": {},
        },
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == {
        "vertex_credentials": {"private_key": "pk-1234567890", "client_email": "svc@example.com"}
    }
    assert CredentialAccessor.get_credential_values("existing")["vertex_credentials"]["private_key"] == "pk-1234567890"


def test_update_credential_keeps_a_nested_secret_when_a_sibling_field_inside_the_object_changes(credential_store):
    """Editing ``client_email`` inside a service-account object sends the object back with its
    ``private_key`` still masked. The object replaces the stored one whole, so the masked leaf has
    to be swapped for the stored leaf instead of the object being compared as a unit."""
    served = CredentialItem(
        credential_name="existing",
        credential_values={"vertex_credentials": {"private_key": "pk-1234567890", "client_email": "svc@example.com"}},
        credential_info={},
    )
    update_by_name = _serve_credential(credential_store, served)

    response = _patch_credential(
        "existing",
        {
            "credential_name": "existing",
            "credential_values": {"vertex_credentials": {"private_key": "pk****90", "client_email": "new@example.com"}},
            "credential_info": {},
        },
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == {
        "vertex_credentials": {"private_key": "pk-1234567890", "client_email": "new@example.com"}
    }
    assert CredentialAccessor.get_credential_values("existing")["vertex_credentials"] == {
        "private_key": "pk-1234567890",
        "client_email": "new@example.com",
    }


def test_update_credential_keeps_the_table_secret_when_the_echo_came_from_a_stale_replica(credential_store):
    """Reads are served from memory, which lags the table after another replica rotates the key.
    An echo of that stale rendering is still a placeholder: the row must keep the rotated key
    rather than store the placeholder or revert to the stale one, and this replica's memory
    catches up to the row right away instead of serving the old key until the next resync."""
    served = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-rotated-key-000099"},
        credential_info={"description": "original"},
    )
    stale = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-old-key-000011"},
        credential_info={"description": "original"},
    )
    update_by_name = _serve_credential(credential_store, served, in_memory=(stale,))

    response = _patch_credential(
        "existing",
        {
            "credential_name": "existing",
            "credential_values": {"api_key": "sk****11"},
            "credential_info": {"description": "edited"},
        },
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == {"api_key": "sk-rotated-key-000099"}
    assert json.loads(update_by_name.await_args.kwargs["data"]["credential_info"])["description"] == "edited"
    assert CredentialAccessor.get_credential_values("existing") == {"api_key": "sk-rotated-key-000099"}


def test_update_credential_keeps_the_stored_secret_when_this_replica_never_loaded_it(credential_store):
    """The echo is judged against the table row, not the in-memory list: a replica that has not
    loaded the credential yet must still keep the real key in the row."""
    served = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-real-secret-1234"},
        credential_info={"description": "original"},
    )
    update_by_name = _serve_credential(credential_store, served, in_memory=())

    response = _patch_credential(
        "existing",
        {
            "credential_name": "existing",
            "credential_values": {"api_key": "sk****34"},
            "credential_info": {"description": "edited"},
        },
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == {"api_key": "sk-real-secret-1234"}
    assert litellm.credential_list == []


def test_update_credential_keeps_the_rotated_key_when_the_echo_is_the_mask_of_the_key_it_replaced(credential_store):
    """A read-back taken before someone rotated the key, whether a script cached it or a replica
    that had not resynced served it, renders the old key. Neither the row nor this replica renders
    that way any more, yet it is still a placeholder: the row keeps the rotated key."""
    served = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-rotated-key-000099"},
        credential_info={"description": "original"},
    )
    update_by_name = _serve_credential(credential_store, served)

    response = _patch_credential(
        "existing",
        {
            "credential_name": "existing",
            "credential_values": {"api_key": "sk****11"},
            "credential_info": {"description": "edited"},
        },
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == {"api_key": "sk-rotated-key-000099"}
    assert json.loads(update_by_name.await_args.kwargs["data"]["credential_info"])["description"] == "edited"
    assert CredentialAccessor.get_credential_values("existing") == {"api_key": "sk-rotated-key-000099"}


def test_update_credential_keeps_the_secret_when_the_patch_echoes_what_get_by_name_served(credential_store):
    """The #28906 flow through both endpoints: whatever ``GET /credentials/by_name`` renders today,
    sending it back with one field edited leaves the secret alone, so the echo rule stays tied to
    the masker's real output rather than to a rendering spelled out by hand."""
    served = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-real-secret-1234", "api_base": "https://api.example.com"},
        credential_info={"custom_llm_provider": "openai", "description": "original"},
    )
    update_by_name = _serve_credential(credential_store, served)
    read_back = _call_as_admin("GET", "/credentials/by_name/existing").json()
    assert read_back["credential_values"]["api_key"] != "sk-real-secret-1234"

    response = _patch_credential(
        "existing", {**read_back, "credential_info": {**read_back["credential_info"], "description": "edited"}}
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == {"api_key": "sk-real-secret-1234", "api_base": "https://api.example.com"}
    assert json.loads(update_by_name.await_args.kwargs["data"]["credential_info"])["description"] == "edited"


def test_update_credential_drops_a_masked_leaf_the_stored_object_no_longer_has(credential_store):
    """A read-back taken before a field was removed from a service-account object still carries
    that field masked. There is no stored leaf to put back, so the placeholder is dropped rather
    than written."""
    served = CredentialItem(
        credential_name="existing",
        credential_values={"vertex_credentials": {"private_key": "pk-1234567890", "client_email": "svc@example.com"}},
        credential_info={},
    )
    update_by_name = _serve_credential(credential_store, served)

    response = _patch_credential(
        "existing",
        {
            "credential_name": "existing",
            "credential_values": {
                "vertex_credentials": {
                    "private_key": "pk****90",
                    "refresh_token": "rt****89",
                    "client_email": "svc@example.com",
                }
            },
            "credential_info": {},
        },
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == {
        "vertex_credentials": {"private_key": "pk-1234567890", "client_email": "svc@example.com"}
    }


@pytest.mark.parametrize("levels", (20, 21))
def test_update_credential_keeps_a_secret_exactly_as_deep_as_the_masker_masks(credential_store, levels):
    """The masker stops masking at a fixed depth and the read-back check has to stop at the same
    one: at the last masked level the echo is a placeholder that must take the stored leaf, and one
    level past it the echo is the secret itself. Either way the write must hold the real secret."""
    nested = reduce(lambda inner, _level: {"credentials": inner}, range(levels), {"api_key": "leaf-secret-1234"})
    served = CredentialItem(credential_name="existing", credential_values=nested, credential_info={})
    update_by_name = _serve_credential(credential_store, served)
    read_back = _get_masked_values(served.credential_values, unmasked_length=4, number_of_asterisks=4)

    response = _patch_credential(
        "existing", {"credential_name": "existing", "credential_values": read_back, "credential_info": {}}
    )

    assert response.status_code == 200, response.text
    assert _written_values(update_by_name) == nested


def test_delete_credential_answers_404_when_the_credential_does_not_exist(credential_store):
    """Regression: prisma's ``delete`` hands back None when the ``where`` clause matched no row
    instead of raising, and the handler never looked. Deleting a name that was never stored
    answered 200 "Credential deleted successfully", so an operator scripting cleanup could not
    tell a real deletion from a typo."""
    credential_store(delete_by_name=AsyncMock(return_value=None))

    response = _delete_credential("definitely-not-there")

    assert response.status_code == 404, (
        f"delete of a missing credential answered {response.status_code}: {response.text}"
    )
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
