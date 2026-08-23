"""Tests for the credential management endpoints."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app
from litellm.types.utils import CredentialItem

client = TestClient(app)


def _as_admin():
    return UserAPIKeyAuth(api_key="test-key", user_role="proxy_admin")


def _as_non_admin():
    return UserAPIKeyAuth(api_key="test-key", user_role="internal_user")


def _patch_credential(name: str, body: dict, auth=_as_admin):
    missing = object()
    previous_override = app.dependency_overrides.get(user_api_key_auth, missing)
    app.dependency_overrides[user_api_key_auth] = auth
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


def _post_credential(body: dict, auth=_as_admin):
    missing = object()
    previous_override = app.dependency_overrides.get(user_api_key_auth, missing)
    app.dependency_overrides[user_api_key_auth] = auth
    try:
        return client.post("/credentials", json=body, headers={"Authorization": "Bearer test-key"})
    finally:
        if previous_override is missing:
            app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            app.dependency_overrides[user_api_key_auth] = previous_override


def test_create_credential_write_omits_the_patch_only_deletion_field():
    """Regression: CredentialItem.credential_values_to_delete is a PATCH-only field that
    defaults to None on every other construction path. A bare .model_dump() (without
    exclude_none) on the create path put a `credential_values_to_delete: null` key into the
    Prisma write, which litellm_credentialstable has no column for."""
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
        create_mock = AsyncMock(return_value=None)
        repository.return_value.create = create_mock

        response = _post_credential(
            {
                "credential_name": "new-cred",
                "credential_values": {"api_key": "sk-new"},
                "credential_info": {"custom_llm_provider": "openai"},
            }
        )

    assert response.status_code == 200, response.text
    written_data = create_mock.await_args.kwargs["data"]
    assert "credential_values_to_delete" not in written_data


def test_update_credential_answers_404_when_the_credential_does_not_exist():
    """Regression: the handler used to ``return handle_exception_on_proxy(e)``, which makes
    the exception the response body and lets FastAPI answer 200, so a write the handler
    rejected read as a success to every caller that checks the status. The dashboard's API
    client branches on the status, so it reported a failed edit as applied."""
    with (
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ),
        patch(  # test-quality-ok: the proxy wiring under test is what this patches
            "litellm.proxy.credential_endpoints.endpoints.CredentialsRepository"
        ) as repository,  # test-quality-ok: the proxy wiring under test is what this patches
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
        repository.return_value.update_by_name = AsyncMock(return_value=None)

        response = _patch_credential(
            "existing",
            {"credential_name": "existing", "credential_values": {"api_key": "sk-new"}, "credential_info": {}},
        )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True


def _get_jwks(name: str):
    missing = object()
    previous_override = app.dependency_overrides.get(user_api_key_auth, missing)
    app.dependency_overrides[user_api_key_auth] = _as_admin
    try:
        return client.get(f"/credentials/{name}/jwks", headers={"Authorization": "Bearer test-key"})
    finally:
        if previous_override is missing:
            app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            app.dependency_overrides[user_api_key_auth] = previous_override


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

    def test_non_admin_can_create_a_credential_without_wif_fields(self):
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
            repository.return_value.create = AsyncMock(return_value=None)

            response = _post_credential(
                {
                    "credential_name": "ordinary-cred",
                    "credential_values": {"api_key": "sk-new"},
                    "credential_info": {"custom_llm_provider": "openai"},
                },
                auth=_as_non_admin,
            )

        assert response.status_code == 200, response.text

    def test_proxy_admin_can_create_a_credential_with_a_wif_destination(self):
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
            repository.return_value.create = AsyncMock(return_value=None)

            response = _post_credential(
                {
                    "credential_name": "admin-cred",
                    "credential_values": {"anthropic_keycloak_token_url": "https://keycloak.internal/token"},
                    "credential_info": {"custom_llm_provider": "anthropic"},
                },
                auth=_as_admin,
            )

        assert response.status_code == 200, response.text

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
