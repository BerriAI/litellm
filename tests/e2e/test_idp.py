"""Harness coverage for idp.py: the pure parts of the Keycloak client, which are
the ones a wrong value in silently mistargets. No proxy and no IdP needed, so
these carry no `e2e` marker and run everywhere."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import SimpleQueue
from threading import Thread
from typing import Final

import pytest
from e2e_http import ExternalWrite
from idp import (
    KEYCLOAK_ADMIN_PASSWORD_ENV,
    KEYCLOAK_ADMIN_USER_ENV,
    KEYCLOAK_REALM_ENV,
    KEYCLOAK_URL_ENV,
    Keycloak,
    PasswordCredential,
    UserCreateBody,
    created_id,
    keycloak_from_env,
)

_REALM: Final = Keycloak(
    base_url="http://keycloak:8080", realm="litellm-e2e", admin_username="admin", admin_password="pw"
)


def test_realm_urls_match_keycloaks_own_layout() -> None:
    assert _REALM.issuer == "http://keycloak:8080/realms/litellm-e2e"
    assert _REALM.jwks_url == "http://keycloak:8080/realms/litellm-e2e/protocol/openid-connect/certs"
    assert _REALM.token_url("master") == "http://keycloak:8080/realms/master/protocol/openid-connect/token"


def test_created_id_is_the_last_segment_of_the_location_header() -> None:
    created: Final = ExternalWrite(
        status_code=201, location="http://keycloak:8080/admin/realms/litellm-e2e/groups/abc-123"
    )
    assert created_id(created, "a group") == "abc-123"


def test_a_refused_create_fails_the_test_with_the_idps_own_words() -> None:
    with pytest.raises(BaseException, match=r"409.*already exists"):
        created_id(ExternalWrite(status_code=409, body="Group already exists"), "a group")


@pytest.mark.parametrize("location", ["", "http://keycloak/groups/"])
def test_create_without_a_resource_id_fails(location: str) -> None:
    with pytest.raises(pytest.fail.Exception, match="resource id"):
        created_id(ExternalWrite(status_code=201, location=location), "a group")


@contextmanager
def _idp_server(
    *, user_status: int = 201, delete_status: int = 204, admin_status: int = 200
) -> Generator[tuple[Keycloak, SimpleQueue[str]]]:
    """Exercise provisioning failures through the same HTTP transport as live tests."""
    deletions: SimpleQueue[str] = SimpleQueue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path.endswith("/token"):
                self.send_response(admin_status)
                self.end_headers()
                self.wfile.write(b'{"access_token":"synthetic-harness-token"}')
            else:
                self.send_response(user_status if self.path.endswith("/users") else 201)
                self.send_header("Location", f"{self.path}/resource-1")
                self.end_headers()
                if user_status != 201 and self.path.endswith("/users"):
                    self.wfile.write(b"injected create failure")

        def do_DELETE(self) -> None:
            deletions.put(self.path)
            self.send_response(delete_status)
            self.end_headers()

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            Keycloak(
                base_url=f"http://127.0.0.1:{server.server_port}",
                realm="test",
                admin_username="admin",
                admin_password="pw",
            ),
            deletions,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_partial_provisioning_removes_the_group_when_user_creation_fails() -> None:
    with _idp_server(user_status=500) as (idp, deletions):
        with ExitStack() as cleanup:

            def defer(callback: Callable[[], object]) -> None:
                cleanup.callback(callback)

            with pytest.raises(pytest.fail.Exception, match="injected create failure"):
                idp.provision(marker="partial", group="team", defer=defer)
        assert deletions.get_nowait() == "/admin/realms/test/groups/resource-1"
        assert deletions.empty()


def test_successful_provisioning_cleans_up_user_before_group() -> None:
    with _idp_server() as (idp, deletions):
        with ExitStack() as cleanup:

            def defer(callback: Callable[[], object]) -> None:
                cleanup.callback(callback)

            idp.provision(marker="complete", group="team", defer=defer)
        assert deletions.get_nowait() == "/admin/realms/test/users/resource-1"
        assert deletions.get_nowait() == "/admin/realms/test/groups/resource-1"
        assert deletions.empty()


def test_cleanup_failure_is_visible() -> None:
    with _idp_server(delete_status=500) as (idp, _):
        with pytest.warns(RuntimeWarning, match="cleanup failed.*HTTP 500"):
            idp.delete_group("group")


def test_expired_admin_credentials_do_not_abort_remaining_cleanups() -> None:
    with _idp_server(admin_status=401) as (idp, _):
        cleanup: Final = ExitStack()
        cleanup.callback(idp.delete_group, "group")
        cleanup.callback(idp.delete_user, "user")
        with pytest.warns(RuntimeWarning, match="cleanup could not authenticate") as warnings:
            cleanup.close()
        assert len(warnings) == 2


def test_new_users_are_born_fully_set_up() -> None:
    """A user without a profile or with a pending required action authenticates
    nowhere: Keycloak answers every grant with "Account is not fully set up"."""
    body: Final = UserCreateBody(
        username="e2e", email="e2e@example.com", groups=("team",), credentials=(PasswordCredential(value="pw"),)
    ).model_dump(by_alias=True)

    assert body["requiredActions"] == ()
    assert body["firstName"] and body["lastName"] and body["emailVerified"] is True
    assert body["credentials"][0]["temporary"] is False


def test_connection_details_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEYCLOAK_URL_ENV, "http://keycloak.litellm.svc.cluster.local:8080/")
    monkeypatch.setenv(KEYCLOAK_REALM_ENV, "other-realm")
    monkeypatch.setenv(KEYCLOAK_ADMIN_USER_ENV, "admin")
    monkeypatch.setenv(KEYCLOAK_ADMIN_PASSWORD_ENV, "pw")

    resolved: Final = keycloak_from_env()

    assert resolved.issuer == "http://keycloak.litellm.svc.cluster.local:8080/realms/other-realm"
    assert resolved.admin_username == "admin" and resolved.admin_password == "pw"


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_missing_admin_credential_fails_loudly_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    monkeypatch.setenv(KEYCLOAK_ADMIN_USER_ENV, "admin")
    monkeypatch.setenv(KEYCLOAK_ADMIN_PASSWORD_ENV, blank)

    with pytest.raises(BaseException, match=KEYCLOAK_ADMIN_PASSWORD_ENV):
        keycloak_from_env()
