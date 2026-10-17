"""Tests for the credential management endpoints."""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

import litellm
from litellm.models.credentials import CredentialSource
from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.credential_endpoints.endpoints import get_llm_router
from litellm.proxy.proxy_server import app
from litellm.types.utils import CredentialItem

client = TestClient(app)


def _as_admin():
    return UserAPIKeyAuth(api_key="test-key", user_role="proxy_admin")


def _as_non_admin():
    return UserAPIKeyAuth(api_key="test-key", user_role="internal_user")


def _call_as(method: str, path: str, json_body: dict | None = None, auth=_as_admin):
    missing = object()
    previous_override = app.dependency_overrides.get(user_api_key_auth, missing)
    app.dependency_overrides[user_api_key_auth] = auth
    try:
        return client.request(method, path, json=json_body, headers={"Authorization": "Bearer test-key"})
    finally:
        if previous_override is missing:
            app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            app.dependency_overrides[user_api_key_auth] = previous_override


def _patch_credential(name: str, body: dict, auth=_as_admin):
    return _call_as("PATCH", f"/credentials/{name}", body, auth)


def _post_credential(body: dict, auth=_as_admin):
    return _call_as("POST", "/credentials", body, auth)


def _delete_credential(name: str, auth=_as_admin):
    return _call_as("DELETE", f"/credentials/{name}", auth=auth)


def _list_credentials():
    return _call_as("GET", "/credentials")


def _prisma_without_credential_rows() -> MagicMock:
    prisma_client = MagicMock()
    prisma_client.db.litellm_credentialstable.find_unique = AsyncMock(return_value=None)
    return prisma_client


@pytest.fixture
def credential_store():
    """Stands the credential store up for one test: whether the database is reachable, what
    the proxy is already serving from memory, which router deployments resolve against, and
    what each repository call hands back."""

    def install(
        *,
        connected: bool = True,
        in_memory: tuple[object, ...] = (),
        llm_router: object | None = None,
        **repository_calls: AsyncMock,
    ) -> None:
        patch(
            "litellm.proxy.proxy_server.prisma_client", _prisma_without_credential_rows() if connected else None
        ).start()
        patch("litellm.proxy.proxy_server.master_key", "sk-test-master").start()
        patch.object(litellm, "credential_list", list(in_memory)).start()
        app.dependency_overrides[get_llm_router] = lambda: llm_router
        repository = patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository").start()
        repository.return_value.find_by_name = AsyncMock(return_value=None)
        for call_name, result in repository_calls.items():
            setattr(repository.return_value, call_name, result)

    yield install
    patch.stopall()
    app.dependency_overrides.pop(get_llm_router, None)


@contextmanager
def _repository_holding(stored: CredentialItem | None):
    """The credentials repository seam, answering ``find_by_name`` with ``stored`` and recording
    the writes the handler attempts. Patched at both import sites, since the handlers resolve an
    existing credential through ``hydrate_named_credential`` (memory first, then this repository)
    and then write through their own ``CredentialsRepository`` binding."""
    with (
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ),
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.master_key", "sk-test-master"
        ),
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.credential_endpoints.endpoints.CredentialsRepository"
        ) as repository,
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.common_utils.credential_hydration.CredentialsRepository", repository
        ),
    ):
        repository.return_value.find_by_name = AsyncMock(return_value=stored)
        repository.return_value.create = AsyncMock(return_value=None)
        repository.return_value.update_by_name = AsyncMock(return_value=None)
        repository.return_value.delete_by_name = AsyncMock(return_value=stored)
        yield repository.return_value


def test_create_credential_write_omits_the_patch_only_deletion_field(restore_credential_list):
    """Regression: CredentialItem.credential_values_to_delete is a PATCH-only field that
    defaults to None on every other construction path. A bare .model_dump() (without
    exclude_none) on the create path put a `credential_values_to_delete: null` key into the
    Prisma write, which litellm_credentialstable has no column for."""
    with _repository_holding(None) as repository:
        response = _post_credential(
            {
                "credential_name": "new-cred",
                "credential_values": {"api_key": "sk-new"},
                "credential_info": {"custom_llm_provider": "openai"},
            }
        )

    assert response.status_code == 200, response.text
    written_data = repository.create.await_args.kwargs["data"]
    assert "credential_values_to_delete" not in written_data


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


def _credential_row(display_name: str, api_key: str, name: str = "replicated") -> dict:
    return {
        "credential_name": name,
        "display_name": display_name,
        "credential_values": {"api_key": api_key},
        "credential_info": {"custom_llm_provider": "openai"},
    }


def _credentials_table(row: dict | None) -> SimpleNamespace:
    return SimpleNamespace(
        find_many=AsyncMock(),
        create=AsyncMock(),
        find_unique=AsyncMock(return_value=row),
        update=AsyncMock(return_value=None),
    )


def test_update_credential_merges_onto_the_writer_row_not_a_lagging_replica(restore_credential_list):
    """Regression: PATCH read the row through the read replica and wrote the merged row to the
    writer, so an edit inside the replica lag window wrote back the replica's older label and
    secrets, undoing a relabel or key rotation that had already committed."""
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer_table = _credentials_table(_credential_row("Relabeled", "sk-B"))
    reader_table = _credentials_table(_credential_row("Original", "sk-A"))
    writer_inner = SimpleNamespace(litellm_credentialstable=writer_table)
    reader_inner = SimpleNamespace(litellm_credentialstable=reader_table)
    prisma_client = MagicMock()
    prisma_client.db = RoutingPrismaWrapper(
        writer=PrismaWrapper(original_prisma=writer_inner, iam_token_db_auth=False),
        reader=PrismaWrapper(original_prisma=reader_inner, iam_token_db_auth=False),
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
        patch("litellm.proxy.proxy_server.master_key", "sk-test-master"),
    ):
        response = _patch_credential(
            "replicated", {"credential_values": {"api_base": "https://example.test/v1"}, "credential_info": {}}
        )

    assert response.status_code == 200, response.text
    written = writer_table.update.await_args.kwargs["data"]
    assert written["display_name"] == "Relabeled"
    assert json.loads(written["credential_values"])["api_key"] == "sk-B"
    reader_table.find_unique.assert_not_awaited()


def _get_jwks(name: str):
    return _call_as("GET", f"/credentials/{name}/jwks")


@pytest.fixture
def restore_credential_list(monkeypatch):
    monkeypatch.setattr(litellm, "credential_list", [])


def test_update_credential_rejects_overlap_between_update_and_delete():
    """A key in both sets is ambiguous (set to what value, before or after the delete?), so the
    endpoint must reject it outright rather than picking a resolution order silently."""
    response = _patch_credential(
        "any-name",
        {
            "credential_name": "any-name",
            "credential_values": {"api_key": "sk-new"},
            "credential_values_to_delete": ["api_key"],
            "credential_info": {},
        },
    )

    assert response.status_code == 400, response.text
    assert "api_key" in response.json()["error"]["message"]


def test_update_credential_deletion_removes_the_key_from_the_db_write(restore_credential_list):
    """The bug this closes: switching WIF identity sources (or WIF -> api_key) left the old
    variant's fields behind in the DB row, which wif.py then rejects by presence."""
    stored = CredentialItem(
        credential_name="wif-cred",
        credential_values={"anthropic_identity_source": "keycloak", "anthropic_keycloak_client_id": "old-client"},
        credential_info={"custom_llm_provider": "anthropic"},
    )
    with (
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ),
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.credential_endpoints.endpoints.CredentialsRepository"
        ) as repository,  # test-quality-ok: the proxy wiring under test is what this patches
    ):
        repository.return_value.find_by_name = AsyncMock(return_value=stored)
        update_mock = AsyncMock(return_value=None)
        repository.return_value.update_by_name = update_mock

        response = _patch_credential(
            "wif-cred",
            {
                "credential_name": "wif-cred",
                "credential_values": {},
                "credential_values_to_delete": ["anthropic_keycloak_client_id"],
                "credential_info": {},
            },
        )

    assert response.status_code == 200, response.text
    written_values = json.loads(update_mock.await_args.kwargs["data"]["credential_values"])
    assert "anthropic_keycloak_client_id" not in written_values
    assert written_values["anthropic_identity_source"] == "keycloak"


def test_update_credential_deletion_updates_in_memory_credential_list(restore_credential_list, monkeypatch):
    """The in-memory list is what the request-time auth resolvers read; a deletion that only
    landed in the DB would leave the stale field servable until the next process restart."""
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="wif-cred",
                credential_values={
                    "anthropic_identity_source": "keycloak",
                    "anthropic_keycloak_client_id": "old-client",
                },
                credential_info={"custom_llm_provider": "anthropic"},
            )
        ],
    )
    stored = CredentialItem(
        credential_name="wif-cred",
        credential_values={"anthropic_identity_source": "keycloak", "anthropic_keycloak_client_id": "old-client"},
        credential_info={"custom_llm_provider": "anthropic"},
    )
    with (
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ),
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.credential_endpoints.endpoints.CredentialsRepository"
        ) as repository,  # test-quality-ok: the proxy wiring under test is what this patches
    ):
        repository.return_value.find_by_name = AsyncMock(return_value=stored)
        repository.return_value.update_by_name = AsyncMock(return_value=None)

        response = _patch_credential(
            "wif-cred",
            {
                "credential_name": "wif-cred",
                "credential_values": {},
                "credential_values_to_delete": ["anthropic_keycloak_client_id"],
                "credential_info": {},
            },
        )

    assert response.status_code == 200, response.text
    in_memory = next(c for c in litellm.credential_list if c.credential_name == "wif-cred")
    assert "anthropic_keycloak_client_id" not in in_memory.credential_values
    assert in_memory.credential_values["anthropic_identity_source"] == "keycloak"


def test_update_credential_leaves_untouched_fields_alone():
    """Regression for the masked-value hazard: GET /credentials masks values, so a PATCH that
    only names the field being changed must not let an untouched field be nulled or overwritten
    by anything a round-tripped (masked) form value could contain."""
    stored = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-real-value", "api_base": "https://api.anthropic.com"},
        credential_info={"custom_llm_provider": "anthropic"},
    )
    with (
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ),
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.master_key", "sk-test-master"
        ),
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.credential_endpoints.endpoints.CredentialsRepository"
        ) as repository,  # test-quality-ok: the proxy wiring under test is what this patches
    ):
        repository.return_value.find_by_name = AsyncMock(return_value=stored)
        update_mock = AsyncMock(return_value=None)
        repository.return_value.update_by_name = update_mock

        response = _patch_credential(
            "existing",
            {"credential_name": "existing", "credential_values": {"api_key": "sk-rotated"}, "credential_info": {}},
        )

    assert response.status_code == 200, response.text
    written_values = json.loads(update_mock.await_args.kwargs["data"]["credential_values"])
    assert written_values["api_base"] == "https://api.anthropic.com"


def test_create_credential_never_stores_a_null_credential_value(restore_credential_list):
    """The dashboard posts a key for every field on the provider's form, and the ones the operator
    left blank arrive as null. A null carries no credential, and the federation resolver refuses a
    foreign variant's field by key, so a stored null wedges every deployment naming this credential."""
    with _repository_holding(None) as repository:
        response = _post_credential(
            {
                "credential_name": "new-cred",
                "credential_values": {"api_key": "sk-new", "anthropic_issuer_url": None},
                "credential_info": {"custom_llm_provider": "anthropic"},
            }
        )

    assert response.status_code == 200, response.text
    written_values = json.loads(repository.create.await_args.kwargs["data"]["credential_values"])
    assert "anthropic_issuer_url" not in written_values
    assert "api_key" in written_values


def test_update_credential_never_stores_a_null_credential_value(restore_credential_list):
    """Same null on the update path, where the merge writes the whole row back: the field the null
    named keeps whatever it stored, since removing a field is what credential_values_to_delete is for."""
    stored = CredentialItem(
        credential_name="wif-cred",
        credential_values={"anthropic_identity_source": "keycloak", "anthropic_keycloak_client_id": "old-client"},
        credential_info={"custom_llm_provider": "anthropic"},
    )
    with _repository_holding(stored) as repository:
        response = _patch_credential(
            "wif-cred",
            {
                "credential_name": "wif-cred",
                "credential_values": {"anthropic_keycloak_client_id": None},
                "credential_info": {},
            },
        )

    assert response.status_code == 200, response.text
    written_values = json.loads(repository.update_by_name.await_args.kwargs["data"]["credential_values"])
    assert written_values["anthropic_keycloak_client_id"] == "old-client"


def test_update_credential_never_syncs_a_null_into_the_in_memory_credential(restore_credential_list, monkeypatch):
    """The in-memory list is what request-time resolution reads, so a null that only got kept out of
    the DB row would still wedge every deployment until the next restart."""
    in_memory = CredentialItem(
        credential_name="plain-cred",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "anthropic"},
    )
    monkeypatch.setattr(litellm, "credential_list", [in_memory])
    with _repository_holding(
        CredentialItem(
            credential_name="plain-cred",
            credential_values={"api_key": "sk-old"},
            credential_info={"custom_llm_provider": "anthropic"},
        )
    ):
        response = _patch_credential(
            "plain-cred",
            {
                "credential_name": "plain-cred",
                "credential_values": {"api_key": "sk-rotated", "anthropic_issuer_url": None},
                "credential_info": {},
            },
        )

    assert response.status_code == 200, response.text
    synced = next(c for c in litellm.credential_list if c.credential_name == "plain-cred")
    assert "anthropic_issuer_url" not in synced.credential_values
    assert synced.credential_values["api_key"] == "sk-rotated"


def _generate_es256_pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


class TestCredentialJwksExport:
    def test_jwks_export_succeeds_for_an_internal_issuer_credential(self, restore_credential_list, monkeypatch):
        monkeypatch.setenv("JWKS_TEST_SIGNING_KEY", _generate_es256_pem())
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="anthropic-issuer",
                    credential_values={
                        "anthropic_identity_source": "internal_issuer",
                        "anthropic_issuer_url": "https://issuer.example.com",
                        "anthropic_issuer_subject": "my-workload",
                        "anthropic_issuer_signing_key_ref": "os.environ/JWKS_TEST_SIGNING_KEY",
                    },
                    credential_info={"custom_llm_provider": "anthropic"},
                )
            ],
        )

        response = _get_jwks("anthropic-issuer")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["keys"][0]["kty"] == "EC"
        assert body["keys"][0]["crv"] == "P-256"
        # The private key material must never leave the process via this endpoint.
        assert "JWKS_TEST_SIGNING_KEY" not in response.text
        assert "PRIVATE KEY" not in response.text

    def test_jwks_export_treats_blank_optional_fields_as_unset(self, restore_credential_list, monkeypatch):
        monkeypatch.setenv("JWKS_TEST_SIGNING_KEY", _generate_es256_pem())
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="anthropic-issuer-blanks",
                    credential_values={
                        "anthropic_identity_source": "internal_issuer",
                        "anthropic_issuer_url": "https://issuer.example.com",
                        "anthropic_issuer_subject": "my-workload",
                        "anthropic_issuer_signing_key_ref": "os.environ/JWKS_TEST_SIGNING_KEY",
                        "anthropic_issuer_audience": "",
                        "anthropic_issuer_ttl_seconds": "",
                    },
                    credential_info={"custom_llm_provider": "anthropic"},
                )
            ],
        )

        response = _get_jwks("anthropic-issuer-blanks")

        assert response.status_code == 200, response.text
        assert response.json()["keys"][0]["kty"] == "EC"

    def test_jwks_export_accepts_the_dashboard_provider_casing(self, restore_credential_list, monkeypatch):
        monkeypatch.setenv("JWKS_TEST_SIGNING_KEY", _generate_es256_pem())
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="anthropic-from-modal",
                    credential_values={
                        "anthropic_identity_source": "internal_issuer",
                        "anthropic_issuer_url": "https://issuer.example.com",
                        "anthropic_issuer_subject": "my-workload",
                        "anthropic_issuer_signing_key_ref": "os.environ/JWKS_TEST_SIGNING_KEY",
                    },
                    credential_info={"custom_llm_provider": "Anthropic"},
                )
            ],
        )

        response = _get_jwks("anthropic-from-modal")

        assert response.status_code == 200, response.text
        assert response.json()["keys"][0]["kty"] == "EC"

    def test_jwks_export_404s_for_a_non_anthropic_credential(self, restore_credential_list, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="openai-key",
                    credential_values={"api_key": "sk-x"},
                    credential_info={"custom_llm_provider": "openai"},
                )
            ],
        )

        response = _get_jwks("openai-key")

        assert response.status_code == 404, response.text

    def test_jwks_export_404s_for_an_anthropic_credential_without_internal_issuer(
        self, restore_credential_list, monkeypatch
    ):
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="anthropic-apikey",
                    credential_values={"api_key": "sk-ant"},
                    credential_info={"custom_llm_provider": "anthropic"},
                )
            ],
        )

        response = _get_jwks("anthropic-apikey")

        assert response.status_code == 404, response.text

    def test_jwks_export_404s_for_an_unknown_credential(self, restore_credential_list):
        with patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.prisma_client", None
        ):  # test-quality-ok: the proxy wiring under test is what this patches
            response = _get_jwks("does-not-exist")

        assert response.status_code == 404, response.text

    def test_jwks_export_requires_proxy_admin(self, restore_credential_list, monkeypatch):
        monkeypatch.setenv("JWKS_TEST_SIGNING_KEY", _generate_es256_pem())
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="anthropic-issuer",
                    credential_values={
                        "anthropic_identity_source": "internal_issuer",
                        "anthropic_issuer_url": "https://issuer.example.com",
                        "anthropic_issuer_subject": "my-workload",
                        "anthropic_issuer_signing_key_ref": "os.environ/JWKS_TEST_SIGNING_KEY",
                    },
                    credential_info={"custom_llm_provider": "anthropic"},
                )
            ],
        )

        def _as_internal_user():
            return UserAPIKeyAuth(api_key="test-key", user_role="internal_user")

        app.dependency_overrides[user_api_key_auth] = _as_internal_user
        try:
            response = client.get("/credentials/anthropic-issuer/jwks", headers={"Authorization": "Bearer test-key"})
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 403, response.text


class TestNonAdminCannotPersistWifFieldsOnCredential:
    """A credential's ``credential_values`` feeds the same WIF resolution as a deployment's own
    ``litellm_params`` when referenced by ``litellm_credential_name``. A non-admin must not be
    able to create or update a credential carrying a server-owned WIF field such as
    ``anthropic_keycloak_token_url`` (destination) or ``anthropic_keycloak_client_secret_ref``
    (which secret to read and send there)."""

    def test_non_admin_cannot_create_a_credential_with_a_wif_destination(self):
        with patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ):  # test-quality-ok: the proxy wiring under test is what this patches
            response = _post_credential(
                {
                    "credential_name": "attacker-cred",
                    "credential_values": {"anthropic_keycloak_token_url": "https://evil.example.com/token"},
                    "credential_info": {"custom_llm_provider": "anthropic"},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text
        assert "anthropic_keycloak_token_url" in response.json()["error"]["message"]

    def test_non_admin_cannot_create_a_credential_with_a_wif_secret_ref(self):
        with patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ):  # test-quality-ok: the proxy wiring under test is what this patches
            response = _post_credential(
                {
                    "credential_name": "attacker-cred",
                    "credential_values": {"anthropic_keycloak_client_secret_ref": "os.environ/LITELLM_MASTER_KEY"},
                    "credential_info": {"custom_llm_provider": "anthropic"},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text

    def test_non_admin_can_create_a_credential_without_wif_fields(self, restore_credential_list):
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "ordinary-cred",
                    "credential_values": {"api_key": "sk-new"},
                    "credential_info": {"custom_llm_provider": "openai"},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 200, response.text
        repository.create.assert_awaited_once()

    def test_proxy_admin_can_create_a_credential_with_a_wif_destination(self, restore_credential_list):
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "admin-cred",
                    "credential_values": {"anthropic_keycloak_token_url": "https://keycloak.internal/token"},
                    "credential_info": {"custom_llm_provider": "anthropic"},
                },
                auth=_as_admin,
            )

        assert response.status_code == 200, response.text
        repository.create.assert_awaited_once()

    def test_non_admin_cannot_create_a_credential_with_oauth_token_exchange_endpoint(self, restore_credential_list):
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "attacker-oauth",
                    "credential_values": {"token_exchange_endpoint": "https://attacker.example/token"},
                    "credential_info": {"custom_llm_provider": "microsoft_365_copilot"},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text
        assert "token_exchange_endpoint" in response.json()["error"]["message"]
        repository.create.assert_not_awaited()

    def test_proxy_admin_can_create_a_credential_with_oauth_token_exchange_endpoint(self, restore_credential_list):
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "admin-oauth",
                    "credential_values": {"token_exchange_endpoint": "https://identity.example.com/token"},
                    "credential_info": {"custom_llm_provider": "microsoft_365_copilot"},
                },
                auth=_as_admin,
            )

        assert response.status_code == 200, response.text
        repository.create.assert_awaited_once()

    def test_non_admin_cannot_create_a_credential_with_an_openai_token_file(self):
        with patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ):
            response = _post_credential(
                {
                    "credential_name": "attacker-cred",
                    "credential_values": {"openai_identity_token_file": "/var/run/secrets/tokens/attacker"},
                    "credential_info": {"custom_llm_provider": "openai"},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text
        assert "openai_identity_token_file" in response.json()["error"]["message"]

    def test_proxy_admin_can_create_a_credential_with_the_openai_identity_trio(self, restore_credential_list):
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "openai-wif",
                    "credential_values": {
                        "openai_identity_provider_id": "idp_1",
                        "openai_service_account_id": "user-1",
                        "openai_identity_token_file": "/var/run/secrets/tokens/openai",
                    },
                    "credential_info": {"custom_llm_provider": "openai"},
                },
                auth=_as_admin,
            )

        assert response.status_code == 200, response.text
        repository.create.assert_awaited_once()

    def test_non_admin_cannot_update_a_credential_to_add_a_wif_destination(self):
        stored = CredentialItem(
            credential_name="existing",
            credential_values={"api_key": "sk-old"},
            credential_info={"custom_llm_provider": "anthropic"},
        )
        with (
            patch(  # test-quality-ok: the proxy wiring under test is what this patches
                "litellm.proxy.proxy_server.prisma_client", MagicMock()
            ),
            patch(  # test-quality-ok: the proxy wiring under test is what this patches
                "litellm.proxy.credential_endpoints.endpoints.CredentialsRepository"
            ) as repository,  # test-quality-ok: the proxy wiring under test is what this patches
        ):
            repository.return_value.find_by_name = AsyncMock(return_value=stored)
            update_mock = AsyncMock(return_value=None)
            repository.return_value.update_by_name = update_mock

            response = _patch_credential(
                "existing",
                {
                    "credential_name": "existing",
                    "credential_values": {"anthropic_keycloak_token_url": "https://evil.example.com/token"},
                    "credential_info": {},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text
        update_mock.assert_not_awaited()

    def test_non_admin_cannot_patch_wif_fields_onto_a_credential_through_model_id(self, credential_store):
        """Regression: the PATCH gate read only the submitted ``credential_values``, so a non-admin
        naming a federated deployment through ``model_id`` had its WIF fields copied onto an
        ordinary credential unchecked, while POST already gated the resolved values."""
        stored = CredentialItem(credential_name="existing", credential_values={"api_key": "sk-old"}, credential_info={})
        update_by_name = AsyncMock(return_value=None)
        router = MagicMock()
        router.get_deployment.return_value = {"model_name": "claude-opus-5-5"}
        router.get_deployment_credentials.return_value = {
            "anthropic_keycloak_token_url": "https://keycloak.internal/token",
            "anthropic_keycloak_client_secret_ref": "os.environ/KEYCLOAK_CLIENT_SECRET",
        }
        credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=update_by_name, llm_router=router)

        response = _patch_credential(
            "existing",
            {"credential_name": "existing", "model_id": "federated-deployment", "credential_info": {}},
            auth=_as_non_admin,
        )

        assert response.status_code == 403, response.text
        update_by_name.assert_not_awaited()

    def test_proxy_admin_can_patch_wif_fields_onto_a_credential_through_model_id(self, credential_store):
        stored = CredentialItem(credential_name="existing", credential_values={"api_key": "sk-old"}, credential_info={})
        update_by_name = AsyncMock(return_value=None)
        router = MagicMock()
        router.get_deployment.return_value = {"model_name": "claude-opus-5-5"}
        router.get_deployment_credentials.return_value = {
            "anthropic_keycloak_token_url": "https://keycloak.internal/token"
        }
        credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=update_by_name, llm_router=router)

        response = _patch_credential(
            "existing",
            {"credential_name": "existing", "model_id": "federated-deployment", "credential_info": {}},
            auth=_as_admin,
        )

        assert response.status_code == 200, response.text
        written = json.loads(update_by_name.await_args.kwargs["data"]["credential_values"])
        assert "anthropic_keycloak_token_url" in written, "the deployment's WIF field reaches the stored credential"

    def test_proxy_admin_can_update_a_credential_to_add_a_wif_destination(self):
        stored = CredentialItem(
            credential_name="existing",
            credential_values={"api_key": "sk-old"},
            credential_info={"custom_llm_provider": "anthropic"},
        )
        with (
            patch(  # test-quality-ok: the proxy wiring under test is what this patches
                "litellm.proxy.proxy_server.prisma_client", MagicMock()
            ),
            patch(  # test-quality-ok: the proxy wiring under test is what this patches
                "litellm.proxy.proxy_server.master_key", "sk-test-master"
            ),
            patch(  # test-quality-ok: the proxy wiring under test is what this patches
                "litellm.proxy.credential_endpoints.endpoints.CredentialsRepository"
            ) as repository,  # test-quality-ok: the proxy wiring under test is what this patches
        ):
            repository.return_value.find_by_name = AsyncMock(return_value=stored)
            update_mock = AsyncMock(return_value=None)
            repository.return_value.update_by_name = update_mock

            response = _patch_credential(
                "existing",
                {
                    "credential_name": "existing",
                    "credential_values": {"anthropic_keycloak_token_url": "https://keycloak.internal/token"},
                    "credential_info": {},
                },
                auth=_as_admin,
            )

        assert response.status_code == 200, response.text
        update_mock.assert_awaited_once()


def _wif_credential(name: str = "federated-cred", source: CredentialSource = "db") -> CredentialItem:
    return CredentialItem(
        credential_name=name,
        credential_values={
            "anthropic_keycloak_token_url": "https://keycloak.internal/token",
            "api_key": "sk-old",
        },
        credential_info={"custom_llm_provider": "anthropic"},
        source=source,
    )


def _plain_credential(name: str = "ordinary-cred") -> CredentialItem:
    return CredentialItem(
        credential_name=name,
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai"},
    )


class TestNonAdminCannotTouchAStoredWifCredential:
    """The WIF gate used to read only the incoming ``credential_values``, so a non-admin could
    drop a federation field by naming it in ``credential_values_to_delete`` (breaking every
    deployment that references the credential), or edit a stored admin-owned WIF credential by
    sending a payload carrying no WIF field at all. The gate is evaluated against the effective
    surface of the operation: incoming keys (a ``null`` value still persists the key), deleted
    keys, and the stored credential, wherever it lives (DB row or config-only ``credential_list``
    entry)."""

    def test_non_admin_cannot_delete_a_wif_field_off_a_credential(self, restore_credential_list):
        with _repository_holding(_plain_credential("some-cred")) as repository:
            response = _patch_credential(
                "some-cred",
                {
                    "credential_name": "some-cred",
                    "credential_values": {},
                    "credential_values_to_delete": ["anthropic_keycloak_token_url"],
                    "credential_info": {},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text
        assert "anthropic_keycloak_token_url" in response.text
        repository.update_by_name.assert_not_awaited()

    def test_non_admin_cannot_patch_a_stored_wif_credential(self, restore_credential_list):
        with _repository_holding(_wif_credential("federated-cred")) as repository:
            response = _patch_credential(
                "federated-cred",
                {
                    "credential_name": "federated-cred",
                    "credential_values": {"api_key": "sk-attacker"},
                    "credential_info": {},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text
        assert "anthropic_keycloak_token_url" in response.text
        repository.update_by_name.assert_not_awaited()

    def test_proxy_admin_can_delete_a_wif_field_off_a_credential(self, restore_credential_list):
        with _repository_holding(_wif_credential("federated-cred")) as repository:
            response = _patch_credential(
                "federated-cred",
                {
                    "credential_name": "federated-cred",
                    "credential_values": {},
                    "credential_values_to_delete": ["anthropic_keycloak_token_url"],
                    "credential_info": {},
                },
                auth=_as_admin,
            )

        assert response.status_code == 200, response.text
        written_values = json.loads(repository.update_by_name.await_args.kwargs["data"]["credential_values"])
        assert "anthropic_keycloak_token_url" not in written_values

    def test_proxy_admin_can_patch_a_stored_wif_credential(self, restore_credential_list):
        with _repository_holding(_wif_credential("federated-cred")) as repository:
            response = _patch_credential(
                "federated-cred",
                {
                    "credential_name": "federated-cred",
                    "credential_values": {"api_key": "sk-rotated"},
                    "credential_info": {},
                },
                auth=_as_admin,
            )

        assert response.status_code == 200, response.text
        written_values = json.loads(repository.update_by_name.await_args.kwargs["data"]["credential_values"])
        assert written_values["anthropic_keycloak_token_url"] is not None

    def test_non_admin_can_still_patch_a_credential_with_no_wif_fields_anywhere(self, restore_credential_list):
        with _repository_holding(_plain_credential("ordinary-cred")) as repository:
            response = _patch_credential(
                "ordinary-cred",
                {
                    "credential_name": "ordinary-cred",
                    "credential_values": {"api_key": "sk-rotated"},
                    "credential_info": {},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 200, response.text
        repository.update_by_name.assert_awaited_once()

    def test_non_admin_cannot_delete_a_stored_wif_credential(self, restore_credential_list):
        """DELETE takes the whole row, so it drops the admin-owned federation settings as surely
        as a targeted key deletion would."""
        with _repository_holding(_wif_credential("federated-cred")) as repository:
            response = _delete_credential("federated-cred", auth=_as_non_admin)

        assert response.status_code == 403, response.text
        assert response.json()["error"]["param"] == "anthropic_keycloak_token_url"
        repository.delete_by_name.assert_not_awaited()

    def test_a_stale_in_memory_copy_does_not_authorize_deleting_a_stored_wif_credential(
        self, restore_credential_list, monkeypatch
    ):
        """Resolution reads memory first and stops, which is right when serving a request. A pod
        whose in-memory copy predates an admin adding the federation fields must not read that
        stale object and authorize the delete: the gate takes the union of memory and the row."""
        monkeypatch.setattr(litellm, "credential_list", [_plain_credential("federated-cred")])

        with _repository_holding(_wif_credential("federated-cred")) as repository:
            response = _delete_credential("federated-cred", auth=_as_non_admin)

        assert response.status_code == 403, response.text
        assert response.json()["error"]["param"] == "anthropic_keycloak_token_url"
        repository.delete_by_name.assert_not_awaited()

    def test_proxy_admin_can_delete_a_stored_wif_credential(self, restore_credential_list):
        with _repository_holding(_wif_credential("federated-cred")) as repository:
            response = _delete_credential("federated-cred", auth=_as_admin)

        assert response.status_code == 200, response.text
        repository.delete_by_name.assert_awaited_once_with("federated-cred")

    def test_non_admin_can_still_delete_a_credential_with_no_wif_fields(self, restore_credential_list):
        with _repository_holding(_plain_credential("ordinary-cred")) as repository:
            response = _delete_credential("ordinary-cred", auth=_as_non_admin)

        assert response.status_code == 200, response.text
        repository.delete_by_name.assert_awaited_once_with("ordinary-cred")

    def test_non_admin_cannot_null_out_a_wif_field_on_a_credential(self, restore_credential_list):
        """A JSON ``null`` still lands as a key in ``credential_values``. ``get_litellm_params``
        forwards a WIF kwarg on key presence and the federation resolver rejects a foreign
        variant's field by key, so a value-based gate let a non-admin persist the key and wedge
        every deployment referencing the credential at request time."""
        with _repository_holding(_plain_credential("some-cred")) as repository:
            response = _patch_credential(
                "some-cred",
                {
                    "credential_name": "some-cred",
                    "credential_values": {"anthropic_issuer_url": None},
                    "credential_info": {},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text
        assert "anthropic_issuer_url" in response.text
        repository.update_by_name.assert_not_awaited()

    def test_non_admin_cannot_patch_a_credential_storing_a_null_wif_field(self, restore_credential_list):
        stored = CredentialItem(
            credential_name="nulled-cred",
            credential_values={"anthropic_issuer_url": None, "api_key": "sk-old"},
            credential_info={"custom_llm_provider": "anthropic"},
        )
        with _repository_holding(stored) as repository:
            response = _patch_credential(
                "nulled-cred",
                {
                    "credential_name": "nulled-cred",
                    "credential_values": {"api_key": "sk-attacker"},
                    "credential_info": {},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text
        assert "anthropic_issuer_url" in response.text
        repository.update_by_name.assert_not_awaited()

    def test_proxy_admin_can_null_out_a_wif_field_on_a_credential(self, restore_credential_list):
        with _repository_holding(_wif_credential("federated-cred")) as repository:
            response = _patch_credential(
                "federated-cred",
                {
                    "credential_name": "federated-cred",
                    "credential_values": {"anthropic_keycloak_token_url": None},
                    "credential_info": {},
                },
                auth=_as_admin,
            )

        assert response.status_code == 200, response.text
        repository.update_by_name.assert_awaited_once()

    def test_non_admin_cannot_delete_a_config_only_wif_credential(self, restore_credential_list, monkeypatch):
        """A ``credential_list`` entry from config.yaml has no DB row, so a gate that consulted
        only the DB let a non-admin evict the admin-owned federation settings from memory."""
        config_credential = _wif_credential("config-wif", source="config")
        monkeypatch.setattr(litellm, "credential_list", [config_credential])
        with _repository_holding(None) as repository:
            response = _delete_credential("config-wif", auth=_as_non_admin)

        assert response.status_code == 403, response.text
        assert response.json()["error"]["param"] == "anthropic_keycloak_token_url"
        repository.delete_by_name.assert_not_awaited()
        assert litellm.credential_list == [config_credential]

    def test_proxy_admin_can_delete_a_config_only_wif_credential(self, restore_credential_list, monkeypatch):
        """The gate lets the admin through to the row delete. The 400 that follows is the rule for
        every config-only credential (no row to delete, the entry is back on the next boot), so the
        in-memory entry stays put too."""
        config_credential = _wif_credential("config-wif", source="config")
        monkeypatch.setattr(litellm, "credential_list", [config_credential])
        with _repository_holding(None) as repository:
            response = _delete_credential("config-wif", auth=_as_admin)

        assert response.status_code == 405, response.text
        assert response.headers["allow"] == "GET"
        assert "defined in config" in response.json()["error"]["message"]
        repository.delete_by_name.assert_awaited_once_with("config-wif")
        assert litellm.credential_list == [config_credential]

    def test_non_admin_cannot_shadow_a_config_only_wif_credential(self, restore_credential_list, monkeypatch):
        """POST with the same name carries no WIF field and collides with no DB row, yet
        ``CredentialAccessor.upsert_credentials`` would replace the admin entry in memory and
        the periodic config sync would then make the takeover permanent."""
        config_credential = _wif_credential("config-wif", source="config")
        monkeypatch.setattr(litellm, "credential_list", [config_credential])
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "config-wif",
                    "credential_values": {"api_key": "sk-attacker"},
                    "credential_info": {"custom_llm_provider": "anthropic"},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text
        assert "anthropic_keycloak_token_url" in response.text
        repository.create.assert_not_awaited()
        assert litellm.credential_list == [config_credential]
        assert litellm.credential_list[0].credential_values["api_key"] == "sk-old"

    def test_proxy_admin_cannot_post_over_a_config_only_wif_credential(self, restore_credential_list, monkeypatch):
        config_credential = _wif_credential("config-wif", source="config")
        monkeypatch.setattr(litellm, "credential_list", [config_credential])
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "config-wif",
                    "credential_values": {"api_key": "sk-rotated"},
                    "credential_info": {"custom_llm_provider": "anthropic"},
                },
                auth=_as_admin,
            )

        assert response.status_code == 409, response.text
        assert response.json()["error"]["param"] == "credential_name"
        assert "defined in config" in response.json()["error"]["message"]
        repository.create.assert_not_awaited()
        assert litellm.credential_list == [config_credential]

    def test_non_admin_cannot_post_a_null_wif_field(self, restore_credential_list):
        """Same key-presence rule on the create path: ``{"anthropic_issuer_url": null}`` persists
        the key, and the resolver reacts to the key."""
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "nulled-cred",
                    "credential_values": {"anthropic_issuer_url": None, "api_key": "sk-new"},
                    "credential_info": {"custom_llm_provider": "anthropic"},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text
        assert "anthropic_issuer_url" in response.text
        repository.create.assert_not_awaited()
        assert litellm.credential_list == []

    def test_non_admin_cannot_shadow_a_db_stored_wif_credential(self, restore_credential_list):
        """Same hole for a WIF credential another pod wrote to the DB before this pod's in-memory
        list caught up: the existing-credential lookup falls through to the DB."""
        with _repository_holding(_wif_credential("federated-cred")) as repository:
            response = _post_credential(
                {
                    "credential_name": "federated-cred",
                    "credential_values": {"api_key": "sk-attacker"},
                    "credential_info": {"custom_llm_provider": "anthropic"},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 403, response.text
        repository.create.assert_not_awaited()

    def test_non_admin_can_still_post_a_credential_with_no_wif_fields_anywhere(self, restore_credential_list):
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "ordinary-cred",
                    "credential_values": {"api_key": "sk-new"},
                    "credential_info": {"custom_llm_provider": "openai"},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 200, response.text
        repository.create.assert_awaited_once()
        assert litellm.credential_list[0].credential_name == "ordinary-cred"


class TestManagementReadsTheStoredCredential:
    """Serving a request reads memory first, which is right. A management operation cannot: on a
    pod whose in-memory copy predates another pod's update it would act on superseded values."""

    @pytest.mark.asyncio
    async def test_authoritative_hydrate_prefers_the_row_over_a_stale_memory_copy(self):
        import litellm
        from litellm.proxy.common_utils.credential_hydration import (
            hydrate_named_credential,
            hydrate_named_credential_authoritative,
        )
        from litellm.types.utils import CredentialItem

        stale = CredentialItem(
            credential_name="anthropic-wif",
            credential_values={"anthropic_issuer_url": "https://old.example.com"},
            credential_info={"custom_llm_provider": "anthropic"},
        )
        row = {
            "credential_name": "anthropic-wif",
            "credential_values": {"anthropic_issuer_url": "https://new.example.com"},
            "credential_info": {"custom_llm_provider": "anthropic"},
        }

        prisma = MagicMock()
        prisma.db.litellm_credentialstable.find_unique = AsyncMock(return_value=row)

        with patch.object(litellm, "credential_list", [stale]):  # test-quality-ok: the stale copy under test
            served = await hydrate_named_credential("anthropic-wif", prisma)
            managed = await hydrate_named_credential_authoritative("anthropic-wif", prisma)

        assert served is not None and served.credential_values["anthropic_issuer_url"] == "https://old.example.com"
        assert managed is not None and managed.credential_values["anthropic_issuer_url"] == "https://new.example.com"

    @pytest.mark.asyncio
    async def test_authoritative_hydrate_falls_back_to_memory_when_the_row_is_absent(self):
        import litellm
        from litellm.proxy.common_utils.credential_hydration import hydrate_named_credential_authoritative
        from litellm.types.utils import CredentialItem

        only_in_memory = CredentialItem(
            credential_name="config-yaml-credential",
            credential_values={"anthropic_issuer_url": "https://configured.example.com"},
            credential_info={"custom_llm_provider": "anthropic"},
        )
        prisma = MagicMock()
        prisma.db.litellm_credentialstable.find_unique = AsyncMock(return_value=None)

        with patch.object(
            litellm, "credential_list", [only_in_memory]
        ):  # test-quality-ok: the config.yaml fallback under test
            resolved = await hydrate_named_credential_authoritative("config-yaml-credential", prisma)

        assert resolved is not None
        assert resolved.credential_values["anthropic_issuer_url"] == "https://configured.example.com"


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
    assert "definitely-not-there" in response.json()["error"]["message"]


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
    proxy boot."""
    config_only = CredentialItem(
        credential_name="from-config-yaml",
        credential_values={"api_key": "sk-config"},
        credential_info={},
        source="config",
    )
    credential_store(in_memory=(config_only,), delete_by_name=AsyncMock(return_value=None))

    response = _delete_credential("from-config-yaml")

    assert response.status_code == 405, response.text
    assert response.headers["allow"] == "GET"
    assert "defined in config" in response.json()["error"]["message"]
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
    return _call_as("POST", "/credentials", body)


class _UniqueViolation(Exception):
    code = "P2002"


def test_create_credential_answers_409_when_the_name_is_already_taken(credential_store):
    """Regression: the unique index used to surface as a Prisma 500 that callers string-matched."""
    credential_store(
        create=AsyncMock(side_effect=_UniqueViolation("Unique constraint failed on the fields: (`credential_name`)")),
    )

    response = _create_credential(
        {"credential_name": "aws_bedrock", "credential_values": {"aws_access_key_id": "new"}, "credential_info": {}},
    )

    assert response.status_code == 409, f"name collision answered {response.status_code}: {response.text}"
    message = response.json()["error"]["message"]
    assert message == (
        "Credential 'aws_bedrock' already exists. Update it with PATCH /credentials/aws_bedrock, or delete it first."
    ), f"the operator reads this message verbatim: {message}"
    assert "Unique constraint" not in response.text, f"the Prisma internals must not leak: {response.text}"


def test_create_credential_still_answers_500_when_the_write_fails_for_another_reason(credential_store):
    credential_store(create=AsyncMock(side_effect=Exception("connection reset by peer")))

    response = _create_credential(
        {"credential_name": "aws_bedrock", "credential_values": {"aws_access_key_id": "new"}, "credential_info": {}},
    )

    assert response.status_code == 500, f"database fault answered {response.status_code}: {response.text}"


def test_create_credential_still_answers_200_for_a_name_that_is_free(credential_store):
    find_by_name = AsyncMock()
    credential_store(find_by_name=find_by_name, create=AsyncMock(return_value=None))

    response = _create_credential(
        {"credential_name": "brand_new", "credential_values": {"aws_access_key_id": "new"}, "credential_info": {}},
    )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    find_by_name.assert_not_awaited(), "the unique index is the guard; create must not add a lookup"


def test_update_credential_resolves_credential_values_from_model_id_like_create(credential_store):
    """Regression: PATCH dropped ``model_id`` from the body, so an update that named a
    deployment instead of raw values wrote whatever the caller sent, or nothing."""
    stored = CredentialItem(
        credential_name="from-deployment",
        credential_values={"api_key": "sk-old"},
        credential_info={},
    )
    update_by_name = AsyncMock(return_value=None)
    router = MagicMock()
    router.get_deployment.return_value = {"model_name": "gpt-5.2"}
    router.get_deployment_credentials.return_value = {"api_key": "sk-from-deployment"}
    credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=update_by_name, llm_router=router)

    response = _patch_credential(
        "from-deployment",
        {"credential_name": "from-deployment", "model_id": "deployment-1", "credential_info": {}},
    )

    assert response.status_code == 200, response.text
    router.get_deployment_credentials.assert_called_once_with("deployment-1")
    written = json.loads(update_by_name.await_args.kwargs["data"]["credential_values"])
    assert set(written) == {"api_key"}
    assert written["api_key"] != "sk-old", "the deployment's values must replace the stored ones"
    assert written["api_key"] != "sk-from-deployment", "values are encrypted before they reach the table"


def test_update_credential_answers_404_when_model_id_names_no_deployment(credential_store):
    stored = CredentialItem(
        credential_name="from-deployment", credential_values={"api_key": "sk-old"}, credential_info={}
    )
    update_by_name = AsyncMock(return_value=None)
    router = MagicMock()
    router.get_deployment.return_value = None
    credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=update_by_name, llm_router=router)

    response = _patch_credential(
        "from-deployment",
        {"credential_name": "from-deployment", "model_id": "no-such-deployment", "credential_info": {}},
    )

    assert response.status_code == 404, response.text
    update_by_name.assert_not_awaited()


def test_update_credential_answers_500_when_model_id_is_given_but_no_router_is_loaded(credential_store):
    stored = CredentialItem(
        credential_name="from-deployment", credential_values={"api_key": "sk-old"}, credential_info={}
    )
    update_by_name = AsyncMock(return_value=None)
    credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=update_by_name, llm_router=None)

    response = _patch_credential(
        "from-deployment",
        {"credential_name": "from-deployment", "model_id": "deployment-1", "credential_info": {}},
    )

    assert response.status_code == 500, response.text
    update_by_name.assert_not_awaited()


def test_update_credential_still_accepts_a_body_without_credential_values(credential_store):
    """Renaming or re-tagging a credential sends only ``credential_info``; that must not 422."""
    stored = CredentialItem(credential_name="existing", credential_values={"api_key": "sk-old"}, credential_info={})
    update_by_name = AsyncMock(return_value=None)
    credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=update_by_name)

    response = _patch_credential(
        "existing",
        {"credential_name": "existing", "credential_info": {"custom_llm_provider": "openai"}},
    )

    assert response.status_code == 200, response.text
    written = update_by_name.await_args.kwargs["data"]
    assert json.loads(written["credential_info"]) == {"custom_llm_provider": "openai"}
    assert set(json.loads(written["credential_values"])) == {"api_key"}, "stored values survive an info-only patch"


from contextlib import contextmanager as _ctx

from litellm.caching.dual_cache import DualCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.credential_endpoints import user_provider_credentials as upc

_DEVICE_CODE_URL = "https://github.com/login/device/code"
_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
_GITHUB_USER_URL = "https://api.github.com/user"


def _as_user(user_id="user-a"):
    return lambda: UserAPIKeyAuth(api_key="test-key", user_role="internal_user", user_id=user_id)


def _per_user_credential(name="copilot-cred"):
    return CredentialItem(
        credential_name=name,
        credential_values={"github_copilot_auth_type": "per_user_oauth"},
        credential_info={},
    )


def _connection_row(user_id="user-a", credential_name="copilot-cred", login="octo"):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    return SimpleNamespace(
        user_id=user_id,
        credential_name=credential_name,
        provider="github_copilot",
        credential_b64=upc._encode(
            upc.GithubCopilotUserConnectionPayload(access_token="gho_secret", github_login=login)
        ),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture(autouse=True)
def _connection_master_key(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-test-master")


@_ctx
def _github_http(mapping):
    """Patch the shared async HTTP client to an httpx MockTransport that answers
    GitHub endpoints from {url: response-json dict or callable}; anything else fails."""

    def respond(request):
        entry = mapping.get(str(request.url))
        if entry is None:
            raise AssertionError(f"unexpected request to {request.url}")
        payload = entry(request) if callable(entry) else entry
        return httpx.Response(200, json=payload, request=request)

    client = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    with patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", return_value=client):
        yield


_DEVICE_FLOW_START = {
    "device_code": "dc-1",
    "user_code": "UC-1",
    "verification_uri": "https://github.com/login/device",
    "expires_in": 900,
    "interval": 5,
}


def _patch_user_connection_env(monkeypatch, credentials, rows=()):
    """Per-user env: credential in memory, prisma table answering find_many with rows,
    and a real in-memory cache for device-flow + credential caching."""
    monkeypatch.setattr(litellm, "credential_list", list(credentials))
    prisma_client = MagicMock()
    table = MagicMock()
    table.find_many = AsyncMock(return_value=list(rows))
    table.find_unique = AsyncMock(return_value=rows[0] if rows else None)
    table.upsert = AsyncMock(return_value=None)
    table.delete_many = AsyncMock(return_value=None)
    prisma_client.db.litellm_userprovidercredentials = table
    prisma_client.writer_db = prisma_client.db
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    cache = DualCache()
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", cache)
    return table


def test_user_connections_lists_per_user_credentials_with_connection_state(monkeypatch):
    _patch_user_connection_env(
        monkeypatch,
        [_per_user_credential(), CredentialItem(credential_name="shared", credential_values={}, credential_info={})],
        rows=[_connection_row()],
    )
    response = _call_as("GET", "/credentials/user_connections", auth=_as_user())
    assert response.status_code == 200, response.text
    assert response.json() == {
        "connections": [
            {
                "credential_name": "copilot-cred",
                "provider": "github_copilot",
                "connected": True,
                "github_login": "octo",
                "connected_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    }


def test_user_connections_requires_a_user_id(monkeypatch):
    _patch_user_connection_env(monkeypatch, [_per_user_credential()])
    response = _call_as("GET", "/credentials/user_connections", auth=_as_admin)
    assert response.status_code == 401


def test_user_connection_start_returns_device_flow_fields(monkeypatch):
    _patch_user_connection_env(monkeypatch, [_per_user_credential()])
    with _github_http({_DEVICE_CODE_URL: _DEVICE_FLOW_START}):
        response = _call_as("POST", "/credentials/copilot-cred/user_connection/start", auth=_as_user())
    assert response.status_code == 200, response.text
    assert {key: response.json()[key] for key in ("user_code", "verification_uri", "expires_in", "interval")} == {
        "user_code": "UC-1",
        "verification_uri": "https://github.com/login/device",
        "expires_in": 900,
        "interval": 5,
    }
    handle = response.json()["flow_handle"]
    assert isinstance(handle, str) and "device_code" not in handle and _DEVICE_FLOW_START["device_code"] not in handle


def test_user_connection_start_404s_for_shared_credential(monkeypatch):
    _patch_user_connection_env(
        monkeypatch, [CredentialItem(credential_name="shared", credential_values={}, credential_info={})]
    )
    response = _call_as("POST", "/credentials/shared/user_connection/start", auth=_as_user())
    assert response.status_code == 404


def test_user_connection_poll_rejects_an_undecryptable_flow_handle(monkeypatch):
    _patch_user_connection_env(monkeypatch, [_per_user_credential()])
    response = _call_as(
        "POST",
        "/credentials/copilot-cred/user_connection/poll",
        json_body={"flow_handle": "not-a-real-handle"},
        auth=_as_user(),
    )
    assert response.status_code == 400


def test_user_connection_poll_connected_persists_and_reports_login(monkeypatch):
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential()])
    with _github_http(
        {
            _DEVICE_CODE_URL: _DEVICE_FLOW_START,
            _ACCESS_TOKEN_URL: {"access_token": "gho_1"},
            _COPILOT_TOKEN_URL: {"token": "copilot-tok", "expires_at": 4102444800},
            _GITHUB_USER_URL: {"login": "octo"},
        }
    ):
        start = _call_as("POST", "/credentials/copilot-cred/user_connection/start", auth=_as_user())
        assert start.status_code == 200, start.text
        poll = _call_as(
            "POST",
            "/credentials/copilot-cred/user_connection/poll",
            json_body={"flow_handle": start.json()["flow_handle"]},
            auth=_as_user(),
        )
    assert poll.status_code == 200, poll.text
    assert poll.json() == {"status": "connected", "interval": None, "github_login": "octo"}
    table.upsert.assert_awaited_once()


def test_user_connection_poll_pending(monkeypatch):
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential()])
    with _github_http(
        {
            _DEVICE_CODE_URL: _DEVICE_FLOW_START,
            _ACCESS_TOKEN_URL: {"error": "authorization_pending"},
        }
    ):
        start = _call_as("POST", "/credentials/copilot-cred/user_connection/start", auth=_as_user())
        poll = _call_as(
            "POST",
            "/credentials/copilot-cred/user_connection/poll",
            json_body={"flow_handle": start.json()["flow_handle"]},
            auth=_as_user(),
        )
    assert poll.json()["status"] == "pending"
    table.upsert.assert_not_awaited()


def test_delete_user_connection_disconnects_and_is_idempotent(monkeypatch):
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential()], rows=[_connection_row()])
    response = _call_as("DELETE", "/credentials/copilot-cred/user_connection", auth=_as_user())
    assert response.status_code == 200
    assert response.json() == {"status": "disconnected"}
    table.delete_many.assert_awaited_once_with(where={"user_id": "user-a", "credential_name": "copilot-cred"})

    # idempotent: row now gone
    table.find_unique = AsyncMock(return_value=None)
    again = _call_as("DELETE", "/credentials/copilot-cred/user_connection", auth=_as_user())
    assert again.status_code == 200
    assert again.json() == {"status": "disconnected"}


def test_deleting_the_credential_purges_user_connections(monkeypatch, credential_store):
    rows = [_connection_row(user_id="u1"), _connection_row(user_id="u2")]
    credential_store(
        in_memory=[_per_user_credential()],
        delete_by_name=AsyncMock(return_value=_per_user_credential()),
    )
    prisma_client = MagicMock()
    table = MagicMock()
    table.find_many = AsyncMock(return_value=rows)
    table.delete_many = AsyncMock(return_value=None)
    prisma_client.db.litellm_userprovidercredentials = table
    prisma_client.writer_db = prisma_client.db
    prisma_client.db.litellm_credentialstable.find_unique = AsyncMock(return_value=None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", DualCache())

    response = _delete_credential("copilot-cred")
    assert response.status_code == 200, response.text
    table.delete_many.assert_awaited_with(where={"credential_name": "copilot-cred"})


def test_user_connections_isolated_per_user(monkeypatch):
    """Two users connected to the same credential: A's listing shows only A's login,
    A's delete removes only A's row, and a poll only reads A's device-flow entry."""

    rows = [_connection_row(user_id="user-a", login="octo"), _connection_row(user_id="user-b", login="hubot")]
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential()], rows=rows)

    async def _find_many_matching_user(*, where):
        return [row for row in rows if row.user_id == where["user_id"]]

    table.find_many = AsyncMock(side_effect=_find_many_matching_user)

    listing = _call_as("GET", "/credentials/user_connections", auth=_as_user("user-a"))
    assert listing.status_code == 200
    body = listing.json()["connections"]
    assert [c["github_login"] for c in body] == ["octo"]
    # find_many must be scoped to the calling user
    where = table.find_many.await_args.kwargs["where"]
    assert where["user_id"] == "user-a"

    # B's in-flight device flow handle is bound to B: A polling with it is rejected
    poll = _call_as(
        "POST",
        "/credentials/copilot-cred/user_connection/poll",
        json_body={"flow_handle": _flow_handle_for(user_id="user-b")},
        auth=_as_user("user-a"),
    )
    assert poll.status_code == 400
    table.upsert.assert_not_awaited()

    delete = _call_as("DELETE", "/credentials/copilot-cred/user_connection", auth=_as_user("user-a"))
    assert delete.json() == {"status": "disconnected"}
    table.delete_many.assert_awaited_once_with(where={"user_id": "user-a", "credential_name": "copilot-cred"})


@pytest.mark.parametrize("seat_status", [403, 404])
def test_user_connection_poll_reports_no_copilot_seat(monkeypatch, seat_status):
    """A GitHub seat check that rejects the user's token maps to no_copilot_seat, stores
    nothing, and clears the in-flight device code."""
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential()])

    def respond(request):
        url = str(request.url)
        if url == _DEVICE_CODE_URL:
            return httpx.Response(200, json=_DEVICE_FLOW_START, request=request)
        if url == _ACCESS_TOKEN_URL:
            return httpx.Response(200, json={"access_token": "gho_seat"}, request=request)
        if url == _COPILOT_TOKEN_URL:
            return httpx.Response(seat_status, json={}, request=request)
        pytest.fail(f"unexpected request to {url}")

    client_obj = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    with patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", return_value=client_obj):
        start = _call_as("POST", "/credentials/copilot-cred/user_connection/start", auth=_as_user())
        assert start.status_code == 200
        poll = _call_as(
            "POST",
            "/credentials/copilot-cred/user_connection/poll",
            json_body={"flow_handle": start.json()["flow_handle"]},
            auth=_as_user(),
        )
    assert poll.status_code == 200, poll.text
    assert poll.json()["status"] == "no_copilot_seat"
    table.upsert.assert_not_awaited()


def test_user_connection_poll_rate_limit_returns_429(monkeypatch):
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential()])

    def respond(request):
        url = str(request.url)
        if url == _DEVICE_CODE_URL:
            return httpx.Response(200, json=_DEVICE_FLOW_START, request=request)
        if url == _ACCESS_TOKEN_URL:
            return httpx.Response(200, json={"access_token": "gho_limited"}, request=request)
        if url == _COPILOT_TOKEN_URL:
            return httpx.Response(429, json={}, request=request)
        pytest.fail(f"unexpected request to {url}")

    client_obj = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    with patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", return_value=client_obj):
        start = _call_as("POST", "/credentials/copilot-cred/user_connection/start", auth=_as_user())
        poll = _call_as(
            "POST",
            "/credentials/copilot-cred/user_connection/poll",
            json_body={"flow_handle": start.json()["flow_handle"]},
            auth=_as_user(),
        )
    assert poll.status_code == 429
    table.upsert.assert_not_awaited()


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/credentials/missing/user_connection/start"),
        ("POST", "/credentials/missing/user_connection/poll"),
        ("DELETE", "/credentials/missing/user_connection"),
        ("POST", "/credentials/shared/user_connection/poll"),
        ("DELETE", "/credentials/shared/user_connection"),
    ],
)
def test_user_connection_routes_404_for_missing_or_shared_credentials(monkeypatch, method, path):
    _patch_user_connection_env(
        monkeypatch, [CredentialItem(credential_name="shared", credential_values={}, credential_info={})]
    )
    response = _call_as(method, path, json_body={"flow_handle": "x"}, auth=_as_user())
    assert response.status_code == 404


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/credentials/user_connections"),
        ("POST", "/credentials/copilot-cred/user_connection/start"),
        ("POST", "/credentials/copilot-cred/user_connection/poll"),
        ("DELETE", "/credentials/copilot-cred/user_connection"),
    ],
)
def test_user_connection_routes_401_without_user_id(monkeypatch, method, path):
    _patch_user_connection_env(monkeypatch, [_per_user_credential()])
    response = _call_as(method, path, json_body={"flow_handle": "x"}, auth=_as_admin)  # admin token has no user_id
    assert response.status_code == 401


def test_user_connection_routes_roles():
    """internal_user is allowed (self-scoped), internal_user_view_only is denied, proxy_admin
    is allowed; and POST /credentials stays admin-only for an internal user."""
    from litellm.proxy.auth.route_checks import RouteChecks

    routes = [
        ("/credentials/user_connections", "GET"),
        ("/credentials/copilot-cred/user_connection/start", "POST"),
        ("/credentials/copilot-cred/user_connection/poll", "POST"),
        ("/credentials/copilot-cred/user_connection", "DELETE"),
    ]

    def outcome(role, route, method="GET"):
        if role == LitellmUserRoles.PROXY_ADMIN:
            return "allowed"
        user_obj = LiteLLM_UserTable(user_id="u", user_email="u@x", user_role=role.value)
        valid_token = UserAPIKeyAuth(user_id="u", user_role=role)
        request = MagicMock(spec=Request)
        request.method = method
        request.query_params = {}
        try:
            RouteChecks.non_proxy_admin_allowed_routes_check(
                user_obj=user_obj,
                _user_role=role.value,
                route=route,
                request=request,
                valid_token=valid_token,
                request_data={},
            )
        except HTTPException as exc:
            return f"denied:{exc.status_code}"
        except Exception:
            return "denied"
        return "allowed"

    for route, method in routes:
        assert outcome(LitellmUserRoles.INTERNAL_USER, route, method) == "allowed"
        assert outcome(LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, route, method) == "denied"
    assert outcome(LitellmUserRoles.INTERNAL_USER, "/credentials", method="POST") != "allowed"


class TestNonAdminCannotSetPerUserOauthOnCredential:
    """github_copilot_auth_type selects the GitHub OAuth path; it is server-owned WIF, so a
    team admin must not set it on a credential and a proxy admin can."""

    def test_non_admin_cannot_create_a_per_user_oauth_credential(self):
        with patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ):
            response = _post_credential(
                {
                    "credential_name": "attacker-cred",
                    "credential_values": {"github_copilot_auth_type": "per_user_oauth"},
                    "credential_info": {"custom_llm_provider": "github_copilot"},
                },
                auth=_as_non_admin,
            )
        assert response.status_code == 403, response.text
        assert "github_copilot_auth_type" in response.json()["error"]["message"]

    def test_non_admin_cannot_patch_a_credential_to_per_user_oauth(self, restore_credential_list):
        stored = CredentialItem(
            credential_name="existing",
            credential_values={"api_key": "sk"},
            credential_info={"custom_llm_provider": "github_copilot"},
        )
        with _repository_holding(stored) as repository:
            repository.find_unique_by_name = AsyncMock(return_value=stored)
            response = _patch_credential(
                "existing",
                {
                    "credential_name": "existing",
                    "credential_values": {"github_copilot_auth_type": "per_user_oauth"},
                    "credential_info": {},
                },
                auth=_as_non_admin,
            )
        assert response.status_code == 403, response.text

    def test_proxy_admin_can_create_a_per_user_oauth_credential(self, restore_credential_list):
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "admin-cred",
                    "credential_values": {"github_copilot_auth_type": "per_user_oauth"},
                    "credential_info": {"custom_llm_provider": "github_copilot"},
                },
            )
        assert response.status_code == 200, response.text
        repository.create.assert_awaited_once()

    def test_proxy_admin_can_patch_a_credential_to_per_user_oauth(self, restore_credential_list):
        stored = CredentialItem(
            credential_name="existing",
            credential_values={"api_key": "sk"},
            credential_info={"custom_llm_provider": "github_copilot"},
        )
        with _repository_holding(stored) as repository:
            repository.find_unique_by_name = AsyncMock(return_value=stored)
            repository.update_by_name = AsyncMock(return_value=stored)
            response = _patch_credential(
                "existing",
                {
                    "credential_name": "existing",
                    "credential_values": {"github_copilot_auth_type": "per_user_oauth"},
                    "credential_info": {},
                },
            )
        assert response.status_code == 200, response.text


def test_label_only_patch_does_not_purge_user_connections():
    """A PATCH that only changes display_name sends no credential_name in the
    body, and that must not read as a rename that purges every user's stored
    connection."""
    stored: Final = CredentialItem(
        credential_name="copilot-cred",
        credential_values={"github_copilot_auth_type": "per_user_oauth"},
        credential_info={"custom_llm_provider": "github_copilot"},
    )
    with _repository_holding(stored):
        from litellm.proxy import proxy_server

        table: Final = MagicMock()
        table.find_many = AsyncMock(return_value=[_connection_row()])
        table.delete_many = AsyncMock(return_value=None)
        proxy_server.prisma_client.db.litellm_userprovidercredentials = table
        proxy_server.prisma_client.writer_db = proxy_server.prisma_client.db

        response: Final = _patch_credential(
            "copilot-cred",
            {"display_name": "Renamed Copilot", "credential_info": {"custom_llm_provider": "github_copilot"}},
        )

    assert response.status_code == 200, response.text
    table.find_many.assert_not_awaited()
    table.delete_many.assert_not_awaited()


def test_switching_away_from_per_user_oauth_purges_user_connections():
    stored: Final = CredentialItem(
        credential_name="copilot-cred",
        credential_values={"github_copilot_auth_type": "per_user_oauth"},
        credential_info={"custom_llm_provider": "github_copilot"},
    )
    with _repository_holding(stored):
        from litellm.proxy import proxy_server

        table: Final = MagicMock()
        table.find_many = AsyncMock(return_value=[_connection_row()])
        table.delete_many = AsyncMock(return_value=None)
        proxy_server.prisma_client.db.litellm_userprovidercredentials = table
        proxy_server.prisma_client.writer_db = proxy_server.prisma_client.db

        response: Final = _patch_credential(
            "copilot-cred",
            {
                "credential_name": "copilot-cred",
                "credential_values": {},
                "credential_values_to_delete": ["github_copilot_auth_type"],
                "credential_info": {"custom_llm_provider": "github_copilot"},
            },
        )

    assert response.status_code == 200, response.text
    table.delete_many.assert_awaited_with(where={"credential_name": "copilot-cred"})


def test_pending_device_flow_survives_disconnect_via_stateless_handle(monkeypatch):
    """The device flow is stateless: no worker-local entry exists to clear on
    disconnect, so a still-valid issued handle polled on any worker completes."""
    _patch_user_connection_env(monkeypatch, [_per_user_credential()], rows=[_connection_row()])

    with _github_http({_DEVICE_CODE_URL: _DEVICE_FLOW_START}):
        start = _call_as("POST", "/credentials/copilot-cred/user_connection/start", auth=_as_user())
        assert start.status_code == 200
    delete = _call_as("DELETE", "/credentials/copilot-cred/user_connection", auth=_as_user())
    assert delete.json() == {"status": "disconnected"}


def test_user_connection_cache_keys_cannot_collide_on_colons():
    """user_id/credential_name are joined losslessly, so ('a:b','c') and ('a','b:c')
    can never share a cache or device-flow entry."""
    assert upc._cache_key("a:b", "c") != upc._cache_key("a", "b:c")


def test_user_connection_slashed_credential_name_routes(monkeypatch):
    """A credential named 'team/copilot' must still reach start/poll/delete: the
    routes take :path parameters the same way the admin CRUD routes do."""
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential(name="team/copilot")])
    with _github_http(
        {
            _DEVICE_CODE_URL: _DEVICE_FLOW_START,
            _ACCESS_TOKEN_URL: {"error": "authorization_pending"},
        }
    ):
        start = _call_as("POST", "/credentials/team/copilot/user_connection/start", auth=_as_user())
        assert start.status_code == 200, start.text
        assert start.json()["user_code"] == "UC-1"

        poll = _call_as(
            "POST",
            "/credentials/team/copilot/user_connection/poll",
            json_body={"flow_handle": start.json()["flow_handle"]},
            auth=_as_user(),
        )
    assert poll.status_code == 200
    assert poll.json()["status"] == "pending"

    delete = _call_as("DELETE", "/credentials/team/copilot/user_connection", auth=_as_user())
    assert delete.status_code == 200
    assert delete.json() == {"status": "disconnected"}
    table.delete_many.assert_awaited_once_with(where={"user_id": "user-a", "credential_name": "team/copilot"})


def test_user_connection_route_check_allows_slashed_names_for_internal_users():
    """The RouteChecks allowlist entries use :path placeholders too, so an internal
    user's request to /credentials/team/copilot/user_connection/... still matches."""
    from litellm.proxy.auth.route_checks import RouteChecks

    for route in (
        "/credentials/team/copilot/user_connection/start",
        "/credentials/team/copilot/user_connection/poll",
        "/credentials/team/copilot/user_connection",
    ):
        assert RouteChecks._route_matches_pattern(
            route=route, pattern=route.replace("team/copilot", "{credential_name:path}")
        )

    user_obj = LiteLLM_UserTable(user_id="u", user_email="u@x", user_role=LitellmUserRoles.INTERNAL_USER.value)
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.query_params = {}
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.INTERNAL_USER.value,
        route="/credentials/team/copilot/user_connection/poll",
        request=request,
        valid_token=UserAPIKeyAuth(user_id="u", user_role=LitellmUserRoles.INTERNAL_USER),
        request_data={},
    )


def _flow_handle_for(user_id="user-a", credential_name="copilot-cred", expires_at=None):
    import time as _time

    from litellm.models.credentials import UserConnectionFlowHandle
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    return encrypt_value_helper(
        UserConnectionFlowHandle(
            user_id=user_id,
            credential_name=credential_name,
            device_code="DC-1",
            interval=5,
            expires_at=expires_at if expires_at is not None else _time.time() + 900,
        ).model_dump_json()
    )


def test_user_connection_poll_rejects_a_handle_bound_to_another_user(monkeypatch):
    _patch_user_connection_env(monkeypatch, [_per_user_credential()])
    response = _call_as(
        "POST",
        "/credentials/copilot-cred/user_connection/poll",
        json_body={"flow_handle": _flow_handle_for(user_id="user-b")},
        auth=_as_user("user-a"),
    )
    assert response.status_code == 400


def test_user_connection_poll_rejects_a_handle_bound_to_another_credential(monkeypatch):
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential()])
    with _github_http({_DEVICE_CODE_URL: _DEVICE_FLOW_START}):
        start = _call_as("POST", "/credentials/copilot-cred/user_connection/start", auth=_as_user())
        assert start.status_code == 200
    forged = _flow_handle_for(credential_name="other-cred")
    response = _call_as(
        "POST",
        "/credentials/copilot-cred/user_connection/poll",
        json_body={"flow_handle": forged},
        auth=_as_user(),
    )
    assert response.status_code == 400
    assert start.json()["flow_handle"] != forged
    table.upsert.assert_not_awaited()


def test_user_connection_poll_rejects_an_expired_handle(monkeypatch):
    import time as _time

    _patch_user_connection_env(monkeypatch, [_per_user_credential()])
    response = _call_as(
        "POST",
        "/credentials/copilot-cred/user_connection/poll",
        json_body={"flow_handle": _flow_handle_for(expires_at=_time.time() - 1)},
        auth=_as_user(),
    )
    assert response.status_code == 400


def test_user_connection_poll_valid_handle_works_with_a_fresh_cache(monkeypatch):
    """The handle carries the device code: a poll on a worker whose cache is empty
    still completes the flow, no cross-worker state required."""
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential()])
    with _github_http(
        {
            _DEVICE_CODE_URL: _DEVICE_FLOW_START,
            _ACCESS_TOKEN_URL: {"access_token": "gho_1"},
            _COPILOT_TOKEN_URL: {"token": "copilot-tok", "expires_at": 4102444800},
            _GITHUB_USER_URL: {"login": "octo"},
        }
    ):
        start = _call_as("POST", "/credentials/copilot-cred/user_connection/start", auth=_as_user())
        assert start.status_code == 200, start.text
        # simulate a different worker: swap in a brand new empty cache
        from litellm.caching.dual_cache import DualCache
        from litellm.proxy import proxy_server

        monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache())
        poll = _call_as(
            "POST",
            "/credentials/copilot-cred/user_connection/poll",
            json_body={"flow_handle": start.json()["flow_handle"]},
            auth=_as_user(),
        )
    assert poll.status_code == 200, poll.text
    assert poll.json()["status"] == "connected"
    table.upsert.assert_awaited_once()


def test_delete_user_connection_fails_closed_when_the_tombstone_write_fails(monkeypatch):
    """Redis attached but its set() raises: the disconnect must not delete the row,
    or the cached token would keep working until TTL with nothing left to revoke."""
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential()], rows=[_connection_row()])
    from types import SimpleNamespace

    from litellm.proxy import proxy_server

    failing_redis = SimpleNamespace(
        async_set_cache=AsyncMock(side_effect=ConnectionError("redis down")),
        async_get_cache=AsyncMock(return_value=None),
        async_delete_cache=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache(redis_cache=failing_redis))

    response = _call_as("DELETE", "/credentials/copilot-cred/user_connection", auth=_as_user())
    assert response.status_code == 503, response.text
    table.delete_many.assert_not_awaited()


def test_delete_user_connection_fails_when_the_tombstone_write_silently_noops(monkeypatch):
    """RedisCache.async_set_cache swallows client errors, so a set() that returns
    without writing still looks successful. The disconnect must verify the
    tombstone by reading the key back and refuse to delete the row."""
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential()], rows=[_connection_row()])
    from types import SimpleNamespace

    from litellm.proxy import proxy_server

    store: dict = {}
    silent_redis = SimpleNamespace(
        async_set_cache=AsyncMock(return_value=None),  # reports success, writes nothing
        async_get_cache=AsyncMock(side_effect=lambda key: store.get(key)),
        async_delete_cache=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache(redis_cache=silent_redis))

    response = _call_as("DELETE", "/credentials/copilot-cred/user_connection", auth=_as_user())
    assert response.status_code == 503, response.text
    table.delete_many.assert_not_awaited()


def test_user_connection_poll_reports_503_when_the_cache_overwrite_cannot_land(monkeypatch):
    """The connect save is durable, so a stale not-connected marker that survives
    both the overwrite and the delete must surface as a retryable 503 rather
    than a false "connected"."""
    table = _patch_user_connection_env(monkeypatch, [_per_user_credential()])
    from types import SimpleNamespace

    from litellm.proxy import proxy_server

    stale_redis = SimpleNamespace(
        async_set_cache=AsyncMock(return_value=None),
        async_get_cache=AsyncMock(return_value="stale-value"),
        async_delete_cache=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache(redis_cache=stale_redis))

    with _github_http(
        {
            _DEVICE_CODE_URL: _DEVICE_FLOW_START,
            _ACCESS_TOKEN_URL: {"access_token": "gho_1"},
            _COPILOT_TOKEN_URL: {"token": "copilot-tok", "expires_at": 4102444800},
            _GITHUB_USER_URL: {"login": "octo"},
        }
    ):
        start = _call_as("POST", "/credentials/copilot-cred/user_connection/start", auth=_as_user())
        poll = _call_as(
            "POST",
            "/credentials/copilot-cred/user_connection/poll",
            json_body={"flow_handle": start.json()["flow_handle"]},
            auth=_as_user(),
        )
    assert poll.status_code == 503, poll.text
    assert "cache could not be refreshed" in poll.text
    table.upsert.assert_awaited_once()


def _labeled_credential(name: str = "openai-prod", display_name: str | None = "Prod OpenAI") -> CredentialItem:
    return CredentialItem(
        credential_name=name,
        display_name=display_name,
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai"},
    )


class TestCredentialDisplayName:
    def test_create_stores_the_trimmed_display_name_in_the_row_and_in_memory(self, restore_credential_list):
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "openai-prod",
                    "display_name": "  Prod OpenAI  ",
                    "credential_values": {"api_key": "sk-new"},
                    "credential_info": {"custom_llm_provider": "openai"},
                }
            )

        assert response.status_code == 200, response.text
        assert repository.create.await_args.kwargs["data"]["display_name"] == "Prod OpenAI"
        assert [(c.credential_name, c.display_name) for c in litellm.credential_list] == [
            ("openai-prod", "Prod OpenAI")
        ]

    @pytest.mark.parametrize("display_name", ["", "   ", "x" * 256])
    def test_create_rejects_an_unusable_display_name(self, restore_credential_list, display_name):
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "openai-prod",
                    "display_name": display_name,
                    "credential_values": {"api_key": "sk-new"},
                    "credential_info": {},
                }
            )

        assert response.status_code == 400, response.text
        assert response.json()["error"]["param"] == "display_name"
        repository.create.assert_not_awaited()
        assert litellm.credential_list == []

    def test_create_accepts_a_display_name_of_exactly_the_maximum_length(self, restore_credential_list):
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "openai-prod",
                    "display_name": "x" * 255,
                    "credential_values": {"api_key": "sk-new"},
                    "credential_info": {},
                }
            )

        assert response.status_code == 200, response.text
        assert repository.create.await_args.kwargs["data"]["display_name"] == "x" * 255

    def test_patch_relabels_without_touching_the_name_or_the_values(self, restore_credential_list, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential()])
        with _repository_holding(_labeled_credential()) as repository:
            response = _patch_credential("openai-prod", {"display_name": "Staging OpenAI", "credential_info": {}})

        assert response.status_code == 200, response.text
        repository.update_by_name.assert_awaited_once()
        assert repository.update_by_name.await_args.args[0] == "openai-prod"
        written = repository.update_by_name.await_args.kwargs["data"]
        assert written["credential_name"] == "openai-prod"
        assert written["display_name"] == "Staging OpenAI"
        assert json.loads(written["credential_values"]) == {"api_key": "sk-old"}
        assert [(c.credential_name, c.display_name, c.credential_values) for c in litellm.credential_list] == [
            ("openai-prod", "Staging OpenAI", {"api_key": "sk-old"})
        ]

    def test_patch_without_display_name_keeps_the_stored_one(self, restore_credential_list, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential()])
        with _repository_holding(_labeled_credential()) as repository:
            response = _patch_credential("openai-prod", {"credential_info": {"description": "rotated"}})

        assert response.status_code == 200, response.text
        assert repository.update_by_name.await_args.kwargs["data"]["display_name"] == "Prod OpenAI"
        assert litellm.credential_list[0].display_name == "Prod OpenAI"

    def test_patch_with_null_display_name_clears_it(self, restore_credential_list, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential()])
        with _repository_holding(_labeled_credential()) as repository:
            response = _patch_credential("openai-prod", {"display_name": None, "credential_info": {}})

        assert response.status_code == 200, response.text
        written = repository.update_by_name.await_args.kwargs["data"]
        assert "display_name" in written
        assert written["display_name"] is None
        assert litellm.credential_list[0].display_name is None

    @pytest.mark.parametrize("display_name", ["", "   ", "x" * 256])
    def test_patch_rejects_an_unusable_display_name_and_changes_nothing(
        self, restore_credential_list, monkeypatch, display_name
    ):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential()])
        with _repository_holding(_labeled_credential()) as repository:
            response = _patch_credential("openai-prod", {"display_name": display_name, "credential_info": {}})

        assert response.status_code == 400, response.text
        assert response.json()["error"]["param"] == "display_name"
        repository.update_by_name.assert_not_awaited()
        assert litellm.credential_list == [_labeled_credential()]

    @pytest.mark.parametrize("display_name", ["\u200b", "Prod\u202eAI", "\ufeffProd", "Prod\u0085AI", "Prod\u2028AI"])
    def test_patch_rejects_invisible_or_control_characters_in_display_name(
        self, restore_credential_list, monkeypatch, display_name
    ):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential()])
        with _repository_holding(_labeled_credential()) as repository:
            response = _patch_credential("openai-prod", {"display_name": display_name, "credential_info": {}})

        assert response.status_code == 400, response.text
        assert response.json()["error"]["param"] == "display_name"
        repository.update_by_name.assert_not_awaited()
        assert litellm.credential_list == [_labeled_credential()]

    @pytest.mark.parametrize("display_name", ["\U0001f600" * 255, "Prod\u00a0OpenAI", "Équipe 東京"])
    def test_patch_accepts_visible_unicode_up_to_the_limit_in_code_points(
        self, restore_credential_list, monkeypatch, display_name
    ):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential()])
        table = _credentials_table(_credential_row("Prod OpenAI", "sk-old", name="openai-prod"))
        prisma_client = MagicMock()
        prisma_client.db = SimpleNamespace(litellm_credentialstable=table)
        with (
            patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
            patch("litellm.proxy.proxy_server.master_key", "sk-test-master"),
        ):
            response = _patch_credential("openai-prod", {"display_name": display_name, "credential_info": {}})

        assert response.status_code == 200, response.text
        assert table.update.await_args.kwargs["where"] == {"credential_name": "openai-prod"}
        assert table.update.await_args.kwargs["data"]["display_name"] == display_name

    def test_create_rejects_a_control_character_in_display_name(self, restore_credential_list):
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "new-cred",
                    "display_name": "Prod\u200bOpenAI",
                    "credential_values": {"api_key": "sk-new"},
                    "credential_info": {"custom_llm_provider": "openai"},
                }
            )

        assert response.status_code == 400, response.text
        assert response.json()["error"]["param"] == "display_name"
        repository.create.assert_not_awaited()

    def test_patch_stores_the_trimmed_display_name(self, restore_credential_list, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential()])
        with _repository_holding(_labeled_credential()) as repository:
            response = _patch_credential("openai-prod", {"display_name": "  Staging OpenAI  ", "credential_info": {}})

        assert response.status_code == 200, response.text
        assert repository.update_by_name.await_args.kwargs["data"]["display_name"] == "Staging OpenAI"
        assert litellm.credential_list[0].display_name == "Staging OpenAI"

    def test_patch_treats_an_empty_credential_name_as_omitted(self, restore_credential_list, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential()])
        with _repository_holding(_labeled_credential()) as repository:
            response = _patch_credential(
                "openai-prod", {"credential_name": "", "display_name": "Staging OpenAI", "credential_info": {}}
            )

        assert response.status_code == 200, response.text
        assert repository.update_by_name.await_args.args[0] == "openai-prod"
        assert repository.update_by_name.await_args.kwargs["data"]["credential_name"] == "openai-prod"

    def test_patch_rejects_a_different_credential_name_and_changes_nothing(self, restore_credential_list, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential()])
        with _repository_holding(_labeled_credential()) as repository:
            response = _patch_credential(
                "openai-prod",
                {
                    "credential_name": "openai-production",
                    "credential_values": {"api_key": "sk-new"},
                    "credential_info": {},
                },
            )

        assert response.status_code == 400, response.text
        assert response.json()["error"]["param"] == "credential_name"
        assert "display_name" in response.json()["error"]["message"]
        repository.update_by_name.assert_not_awaited()
        assert litellm.credential_list == [_labeled_credential()]

    def test_patch_accepts_the_same_credential_name_echoed_back(self, restore_credential_list, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential()])
        with _repository_holding(_labeled_credential()) as repository:
            response = _patch_credential(
                "openai-prod",
                {"credential_name": "openai-prod", "credential_values": {"api_key": "sk-new"}, "credential_info": {}},
            )

        assert response.status_code == 200, response.text
        assert [(c.credential_name, c.credential_values) for c in litellm.credential_list] == [
            ("openai-prod", {"api_key": "sk-new"})
        ]

    def test_patch_on_a_config_credential_is_rejected_as_config_owned(self, restore_credential_list, monkeypatch):
        config_credential = _labeled_credential(display_name=None).model_copy(update={"source": "config"})
        monkeypatch.setattr(litellm, "credential_list", [config_credential])
        with _repository_holding(None) as repository:
            response = _patch_credential("openai-prod", {"display_name": "Prod", "credential_info": {}})

        assert response.status_code == 405, response.text
        assert response.headers["allow"] == "GET"
        assert "defined in config" in response.json()["error"]["message"]
        repository.update_by_name.assert_not_awaited()
        assert litellm.credential_list == [config_credential]

    def test_post_on_a_config_credential_name_is_rejected_and_does_not_shadow_it(
        self, restore_credential_list, monkeypatch
    ):
        config_credential = _labeled_credential(display_name=None).model_copy(update={"source": "config"})
        monkeypatch.setattr(litellm, "credential_list", [config_credential])
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "openai-prod",
                    "credential_values": {"api_key": "sk-shadow"},
                    "credential_info": {"custom_llm_provider": "openai"},
                }
            )

        assert response.status_code == 409, response.text
        assert "defined in config" in response.json()["error"]["message"]
        repository.create.assert_not_awaited()
        assert litellm.credential_list == [config_credential]

    def test_post_with_a_display_name_matching_a_config_credential_name_is_allowed(
        self, restore_credential_list, monkeypatch
    ):
        config_credential = _labeled_credential(display_name=None).model_copy(update={"source": "config"})
        monkeypatch.setattr(litellm, "credential_list", [config_credential])
        with _repository_holding(None) as repository:
            response = _post_credential(
                {
                    "credential_name": "openai-staging",
                    "display_name": "openai-prod",
                    "credential_values": {"api_key": "sk-staging"},
                    "credential_info": {"custom_llm_provider": "openai"},
                }
            )

        assert response.status_code == 200, response.text
        repository.create.assert_awaited_once()

    def test_patch_on_an_unknown_credential_still_answers_404(self, restore_credential_list):
        with _repository_holding(None):
            response = _patch_credential("nowhere", {"display_name": "Prod", "credential_info": {}})

        assert response.status_code == 404, response.text

    @pytest.mark.parametrize("method", ["PATCH", "DELETE"])
    def test_a_db_credential_already_gone_from_the_db_answers_404_not_config_owned(
        self, restore_credential_list, monkeypatch, method
    ):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential()])
        with _repository_holding(None) as repository:
            response = (
                _patch_credential("openai-prod", {"display_name": "Prod", "credential_info": {}})
                if method == "PATCH"
                else _delete_credential("openai-prod")
            )

        assert response.status_code == 404, response.text
        repository.update_by_name.assert_not_awaited()

    def test_patch_on_a_name_stored_in_the_db_and_config_applies_to_the_row(self, restore_credential_list, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential(display_name=None)])
        with _repository_holding(_labeled_credential(display_name=None)) as repository:
            response = _patch_credential("openai-prod", {"display_name": "Prod", "credential_info": {}})

        assert response.status_code == 200, response.text
        assert repository.update_by_name.await_args.kwargs["data"]["display_name"] == "Prod"

    def test_reads_return_display_name_and_source(self, restore_credential_list, monkeypatch):
        config_credential = CredentialItem(
            credential_name="from-config",
            credential_values={"api_key": "sk-config-value"},
            credential_info={},
            source="config",
        )
        monkeypatch.setattr(litellm, "credential_list", [_labeled_credential(), config_credential])

        listed = {entry["credential_name"]: entry for entry in _list_credentials().json()["credentials"]}
        by_name = _call_as("GET", "/credentials/by_name/openai-prod").json()
        config_by_name = _call_as("GET", "/credentials/by_name/from-config").json()

        assert (listed["openai-prod"]["display_name"], listed["openai-prod"]["source"]) == ("Prod OpenAI", "db")
        assert (listed["from-config"]["display_name"], listed["from-config"]["source"]) == (None, "config")
        assert (by_name["display_name"], by_name["source"]) == ("Prod OpenAI", "db")
        assert config_by_name["source"] == "config"
        assert listed["openai-prod"]["credential_values"]["api_key"] != "sk-old"
