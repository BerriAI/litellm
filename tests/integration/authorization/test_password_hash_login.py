"""Stored passwords are PBKDF2-HMAC-SHA256 rows; scrypt and legacy SHA256 rows still sign in and get rehashed.

Every cell drives the real /login form and the management endpoints of a proxy this module owns (two workers,
login throttle off, breach screening off so no request leaves the box) and reads the stored row back from
Postgres. The expected hash is recomputed here with hashlib, never with litellm code.
"""

import base64
import hashlib
import os
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import jwt
import psutil
import psycopg
import pytest
import yaml

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.database import read_rows
from tests.integration._support.process import OwnedProxy, owned_proxy_process

PBKDF2_ITERATIONS: Final = 600_000
PASSWORD: Final = "Correct-Horse-9-Battery"
WRONG_PASSWORD: Final = "Wrong-Horse-9-Battery"
BURST_SIZE: Final = 30


@pytest.fixture(scope="module")
def password_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("password-proxy")
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    settings: Final = {**config["general_settings"], "password_policy_check_breached_passwords": False}
    path: Final = directory / "config.yaml"
    path.write_text(yaml.safe_dump({**config, "general_settings": settings}))
    with gateway_from_environment() as suite_gateway:
        with owned_proxy_process(
            suite_gateway,
            directory,
            {"LITELLM_DISABLE_LOGIN_RATE_LIMIT": "true"},
            config=path,
            workers=2,
        ) as owned:
            yield owned


@pytest.fixture
def proxy(password_proxy: OwnedProxy) -> Gateway:
    return password_proxy.gateway


def _stored_password(user_id: str) -> str:
    rows: Final = read_rows('SELECT password FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))
    assert len(rows) == 1, rows
    stored: Final = rows[0]["password"]
    assert isinstance(stored, str), rows
    return stored


def _write_stored_password(user_id: str, stored: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute('UPDATE "LiteLLM_UserTable" SET password = %s WHERE user_id = %s', (stored, user_id))
        connection.commit()


def _scrypt_row(password: str) -> str:
    salt: Final = os.urandom(16)
    derived: Final = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return "scrypt:" + base64.b64encode(salt + derived).decode()


def _sha256_row(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _pbkdf2_matches(stored: str, password: str) -> bool:
    """Independent check of a ``pbkdf2:sha256:<iterations>:<salt b64>:<key b64>`` row with the stdlib."""
    scheme, digest, iterations, salt, derived = stored.split(":")
    assert (scheme, digest, int(iterations)) == ("pbkdf2", "sha256", PBKDF2_ITERATIONS), stored
    expected: Final = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt), int(iterations))
    return base64.b64decode(derived) == expected


def _login(proxy: Gateway, email: str, password: str) -> int:
    response: Final = proxy.client.post(
        "/login", data={"username": email, "password": password}, follow_redirects=False
    )
    if response.status_code == 303:
        assert response.headers["location"].endswith("/ui?login=success"), response.headers
        assert "token=" in response.headers.get("set-cookie", ""), response.headers
    return response.status_code


def _user_with_password(proxy: Gateway, password: str | None) -> tuple[str, str]:
    email: Final = f"integration-{uuid.uuid4().hex}@example.com"
    created: Final = proxy.post(
        "/user/new",
        {"user_id": f"integration-{uuid.uuid4().hex}", "user_email": email, "auto_create_key": False},
    )
    user_id: Final = created["user_id"]
    assert isinstance(user_id, str), created
    if password is not None:
        proxy.post("/user/update", {"user_id": user_id, "password": password})
    return user_id, email


def _delete_user(proxy: Gateway, user_id: str) -> None:
    response: Final = proxy.request("POST", "/user/delete", {"user_ids": [user_id]})
    assert response.status_code == 200, response.text


def test_new_user_password_is_stored_as_pbkdf2_and_signs_in(proxy: Gateway) -> None:
    user_id, email = _user_with_password(proxy, PASSWORD)
    try:
        stored: Final = _stored_password(user_id)
        assert stored.startswith("pbkdf2:sha256:"), stored
        assert _pbkdf2_matches(stored, PASSWORD), stored
        assert _login(proxy, email, PASSWORD) == 303
        assert _stored_password(user_id) == stored, "a pbkdf2 row must not be rewritten on login"
    finally:
        _delete_user(proxy, user_id)


def test_wrong_password_is_rejected_and_row_is_untouched(proxy: Gateway) -> None:
    user_id, email = _user_with_password(proxy, PASSWORD)
    try:
        before: Final = _stored_password(user_id)
        assert _login(proxy, email, WRONG_PASSWORD) == 401
        assert _login(proxy, email, "") == 401
        assert _stored_password(user_id) == before
    finally:
        _delete_user(proxy, user_id)


@pytest.mark.parametrize("legacy_row", (_scrypt_row, _sha256_row), ids=("scrypt", "sha256"))
def test_legacy_hash_signs_in_and_is_rehashed_to_pbkdf2(proxy: Gateway, legacy_row: Callable[[str], str]) -> None:
    user_id, email = _user_with_password(proxy, None)
    try:
        legacy: Final = legacy_row(PASSWORD)
        _write_stored_password(user_id, legacy)
        assert _login(proxy, email, WRONG_PASSWORD) == 401
        assert _stored_password(user_id) == legacy, "a rejected login must not rewrite the row"
        assert _login(proxy, email, PASSWORD) == 303
        rehashed: Final = eventually(lambda: _stored_password(user_id), lambda row: row.startswith("pbkdf2:"))
        assert _pbkdf2_matches(rehashed, PASSWORD), rehashed
        assert _login(proxy, email, PASSWORD) == 303
        assert _login(proxy, email, WRONG_PASSWORD) == 401
    finally:
        _delete_user(proxy, user_id)


@pytest.mark.parametrize("garbage", ("", "plaintext-password", "pbkdf2:sha256:1:not-base64:!!", "scrypt:%%%"))
def test_unreadable_stored_row_rejects_every_password(proxy: Gateway, garbage: str) -> None:
    user_id, email = _user_with_password(proxy, None)
    try:
        _write_stored_password(user_id, garbage)
        assert _login(proxy, email, garbage) == 401
        assert _login(proxy, email, PASSWORD) == 401
        assert _stored_password(user_id) == garbage
    finally:
        _delete_user(proxy, user_id)


def test_changed_password_is_stored_as_pbkdf2(proxy: Gateway) -> None:
    user_id, email = _user_with_password(proxy, PASSWORD)
    try:
        signed_in: Final = proxy.client.post(
            "/login", data={"username": email, "password": PASSWORD}, follow_redirects=False
        )
        assert signed_in.status_code == 303, signed_in.text
        session: Final = jwt.decode(signed_in.cookies["token"], options={"verify_signature": False})["key"]
        assert isinstance(session, str) and session.startswith("sk-"), session
        new_password: Final = "Fresh-Horse-7-Battery"
        changed: Final = proxy.request(
            "POST",
            "/user/password/change",
            {"current_password": PASSWORD, "new_password": new_password},
            key=session,
        )
        assert changed.status_code == 200, changed.text
        stored: Final = _stored_password(user_id)
        assert stored.startswith("pbkdf2:sha256:"), stored
        assert _pbkdf2_matches(stored, new_password), stored
        assert _login(proxy, email, PASSWORD) == 401
        assert _login(proxy, email, new_password) == 303
    finally:
        _delete_user(proxy, user_id)


def test_concurrent_logins_on_a_scrypt_row_rehash_once_while_one_worker_is_killed(
    password_proxy: OwnedProxy,
) -> None:
    proxy: Final = password_proxy.gateway
    user_id, email = _user_with_password(proxy, None)
    try:
        _write_stored_password(user_id, _scrypt_row(PASSWORD))
        workers: Final = psutil.Process(password_proxy.process.pid).children(recursive=True)
        assert len(workers) >= 2, workers
        victim: Final = workers[0]
        victim.kill()
        eventually(lambda: victim.is_running() and victim.status() != psutil.STATUS_ZOMBIE, lambda alive: not alive)
        attempts: Final = tuple(PASSWORD if index % 3 else WRONG_PASSWORD for index in range(BURST_SIZE))
        with ThreadPoolExecutor(max_workers=BURST_SIZE) as pool:
            statuses: Final = tuple(pool.map(lambda password: _login(proxy, email, password), attempts))
        assert sorted(statuses) == sorted(303 if password == PASSWORD else 401 for password in attempts), statuses
        rehashed: Final = eventually(lambda: _stored_password(user_id), lambda row: row.startswith("pbkdf2:"))
        assert _pbkdf2_matches(rehashed, PASSWORD), rehashed
        assert _login(proxy, email, PASSWORD) == 303
        assert _stored_password(user_id) == rehashed, "a settled pbkdf2 row must stay stable across logins"
    finally:
        _delete_user(proxy, user_id)
