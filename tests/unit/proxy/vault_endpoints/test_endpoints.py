"""Tests for the experimental secrets vault endpoints."""

from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app

_CREATED: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _VaultTable:
    """In-memory stand-in for the Prisma ``litellm_vaultsecrets`` table."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], SimpleNamespace] = {}

    @staticmethod
    def _key(where: Mapping[str, Mapping[str, str]]) -> tuple[str, str]:
        owned: Final = where["user_id_secret_name"]
        return owned["user_id"], owned["secret_name"]

    async def upsert(self, where, data):
        key: Final = self._key(where)
        existing: Final = self.rows.get(key)
        fields: Final = data["update"] if existing is not None else data["create"]
        base: Final = vars(existing) if existing is not None else {"created_at": _CREATED, "last_accessed_at": None}
        self.rows[key] = SimpleNamespace(**{**base, **fields, "updated_at": _CREATED})
        return self.rows[key]

    async def find_many(self, where, order=None):
        return sorted(
            (row for row in self.rows.values() if row.user_id == where["user_id"]),
            key=lambda row: row.secret_name,
        )

    async def find_unique(self, where):
        return self.rows.get(self._key(where))

    async def update_many(self, where, data):
        key: Final = (where["user_id"], where["secret_name"])
        if key not in self.rows:
            return 0
        self.rows[key] = SimpleNamespace(**{**vars(self.rows[key]), **data})
        return 1

    async def delete_many(self, where):
        return 0 if self.rows.pop((where["user_id"], where["secret_name"]), None) is None else 1


@pytest.fixture
def table(monkeypatch) -> _VaultTable:
    vault_table: Final = _VaultTable()
    database: Final = SimpleNamespace(litellm_vaultsecrets=vault_table)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        SimpleNamespace(db=database, writer_db=database),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-test-master")
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt")
    return vault_table


@pytest.fixture
def as_user() -> Iterator[object]:
    def client_for(user_id: str | None) -> TestClient:
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id=user_id)
        return TestClient(app)

    yield client_for
    app.dependency_overrides.pop(user_api_key_auth, None)


def test_stored_secret_is_encrypted_at_rest_and_revealed_to_its_owner(table, as_user):
    alice: Final = as_user("alice")

    saved: Final = alice.put("/vault/secrets/GITHUB_TOKEN", json={"value": "ghp_plaintext", "description": "ci"})
    revealed: Final = alice.get("/vault/secrets/GITHUB_TOKEN")

    assert saved.status_code == 200, saved.text
    assert "value" not in saved.json()
    stored: Final = table.rows[("alice", "GITHUB_TOKEN")].value_b64
    assert "ghp_plaintext" not in stored
    assert revealed.status_code == 200, revealed.text
    assert revealed.json()["value"] == "ghp_plaintext"
    assert revealed.json()["description"] == "ci"
    assert revealed.headers["cache-control"] == "no-store"


def test_list_returns_only_the_callers_secrets_and_never_their_values(table, as_user):
    as_user("alice").put("/vault/secrets/B_KEY", json={"value": "alice-b"})
    as_user("alice").put("/vault/secrets/A_KEY", json={"value": "alice-a"})
    as_user("bob").put("/vault/secrets/BOB_KEY", json={"value": "bob-secret"})

    listed: Final = as_user("alice").get("/vault/secrets")

    assert listed.status_code == 200, listed.text
    assert [secret["secret_name"] for secret in listed.json()["secrets"]] == ["A_KEY", "B_KEY"]
    assert "alice-a" not in listed.text
    assert all("value" not in secret for secret in listed.json()["secrets"])


def test_another_user_cannot_reveal_overwrite_or_delete_a_secret(table, as_user):
    as_user("alice").put("/vault/secrets/SHARED_NAME", json={"value": "alice-secret"})
    bob: Final = as_user("bob")

    reveal: Final = bob.get("/vault/secrets/SHARED_NAME")
    bob.put("/vault/secrets/SHARED_NAME", json={"value": "bob-secret"})
    delete: Final = bob.delete("/vault/secrets/SHARED_NAME")

    assert reveal.status_code == 404, reveal.text
    assert delete.json() == {"secret_name": "SHARED_NAME", "deleted": True}
    assert as_user("alice").get("/vault/secrets/SHARED_NAME").json()["value"] == "alice-secret"


def test_put_on_an_existing_secret_replaces_its_value(table, as_user):
    alice: Final = as_user("alice")
    alice.put("/vault/secrets/API_KEY", json={"value": "old"})

    alice.put("/vault/secrets/API_KEY", json={"value": "new"})

    assert alice.get("/vault/secrets/API_KEY").json()["value"] == "new"
    assert len(table.rows) == 1


def test_reveal_records_when_the_secret_was_last_accessed(table, as_user):
    alice: Final = as_user("alice")
    alice.put("/vault/secrets/API_KEY", json={"value": "v"})
    assert alice.get("/vault/secrets").json()["secrets"][0]["last_accessed_at"] is None

    revealed: Final = alice.get("/vault/secrets/API_KEY")

    listed_at: Final = alice.get("/vault/secrets").json()["secrets"][0]["last_accessed_at"]
    assert listed_at is not None
    assert datetime.fromisoformat(revealed.json()["last_accessed_at"]) == datetime.fromisoformat(listed_at)


def test_delete_removes_the_secret_and_is_idempotent(table, as_user):
    alice: Final = as_user("alice")
    alice.put("/vault/secrets/API_KEY", json={"value": "v"})

    first: Final = alice.delete("/vault/secrets/API_KEY")
    second: Final = alice.delete("/vault/secrets/API_KEY")

    assert first.json() == {"secret_name": "API_KEY", "deleted": True}
    assert second.json() == {"secret_name": "API_KEY", "deleted": False}
    assert alice.get("/vault/secrets/API_KEY").status_code == 404


def test_secret_that_no_longer_decrypts_is_an_error_not_ciphertext(table, as_user, monkeypatch):
    alice: Final = as_user("alice")
    alice.put("/vault/secrets/API_KEY", json={"value": "plaintext"})
    stored: Final = table.rows[("alice", "API_KEY")].value_b64
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-a-different-salt")

    revealed: Final = alice.get("/vault/secrets/API_KEY")

    assert revealed.status_code == 500, revealed.text
    assert stored not in revealed.text


@pytest.mark.parametrize("user_id", [None, ""])
def test_key_without_a_user_cannot_use_the_vault(table, as_user, user_id):
    response: Final = as_user(user_id).put("/vault/secrets/API_KEY", json={"value": "v"})

    assert response.status_code == 401, response.text
    assert table.rows == {}


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("has space", {"value": "v"}),
        ("-leading-dash", {"value": "v"}),
        ("x" * 129, {"value": "v"}),
        ("OK_NAME", {"value": ""}),
        ("OK_NAME", {"value": "v", "user_id": "bob"}),
    ],
)
def test_invalid_name_or_body_is_rejected_without_writing(table, as_user, name, body):
    response: Final = as_user("alice").put(f"/vault/secrets/{name}", json=body)

    assert response.status_code == 422, response.text
    assert table.rows == {}


def _request(method: str) -> Request:
    return Request({"type": "http", "method": method, "path": "/", "headers": []})


@pytest.mark.parametrize(
    ("method", "route"),
    [
        ("GET", "/vault/secrets"),
        ("GET", "/vault/secrets/API_KEY"),
        ("PUT", "/vault/secrets/API_KEY"),
        ("DELETE", "/vault/secrets/API_KEY"),
    ],
)
def test_vault_routes_are_self_scoped_so_internal_user_keys_may_call_them(method, route):
    assert RouteChecks.is_self_scoped_credential_route(route=route, request=_request(method))


def test_vault_route_allowance_is_limited_to_the_methods_the_vault_serves():
    assert not RouteChecks.is_self_scoped_credential_route(route="/vault/secrets/API_KEY", request=_request("PATCH"))
    assert not RouteChecks.is_self_scoped_credential_route(route="/vault/other", request=_request("GET"))
