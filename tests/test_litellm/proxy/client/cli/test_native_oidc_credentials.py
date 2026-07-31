"""Tests for native OIDC credential persistence, verification and refresh."""

import json
import os
import stat
import threading
import time

import pytest
import requests

from litellm.litellm_core_utils.cli_token_utils import (
    get_cli_token_file_path,
    load_cli_token,
)
from litellm.proxy.client.cli.native_oidc import credentials as creds
from litellm.proxy.client.cli.native_oidc.credentials import (
    AUTH_TYPE_NATIVE_OIDC,
    REFRESH_BUFFER_SECONDS,
    build_native_credential,
    is_native_credential,
    needs_refresh,
    refresh_lock,
    refresh_native_credential,
    save_credential,
    verify_token_with_litellm,
)
from litellm.proxy.client.cli.native_oidc.errors import (
    NativeOIDCAuthRejected,
    NativeOIDCError,
)
from litellm.proxy.client.cli.native_oidc.metadata import (
    NativeOIDCMetadata,
    ProviderMetadata,
)
from litellm.proxy.client.cli.native_oidc.tokens import TokenResponse

BASE_URL = "https://proxy.example.com"
ISSUER = "https://idp.example.com"
CLIENT_ID = "litellm-cli"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Redirect ~/.litellm to a temp dir; Path.home() honours HOME on POSIX."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def make_metadata(issuer=ISSUER, client_id=CLIENT_ID):
    return NativeOIDCMetadata(issuer=issuer, client_id=client_id, scopes=("openid", "profile"))


def make_provider(issuer=ISSUER, token_endpoint=None):
    return ProviderMetadata(
        issuer=issuer,
        authorization_endpoint=f"{issuer}/authorize",
        token_endpoint=token_endpoint or f"{issuer}/token",
        device_authorization_endpoint=None,
        response_types_supported=("code",),
        grant_types_supported=("authorization_code", "refresh_token"),
        code_challenge_methods_supported=("S256",),
        token_endpoint_auth_methods_supported=("none",),
    )


def make_token(access_token="access-1", refresh_token="refresh-1", expires_at=None):
    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_at=time.time() + 3600 if expires_at is None else expires_at,
        refresh_token=refresh_token,
        scopes=("openid", "profile"),
    )


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


# --------------------------------------------------------------------------
# credential shape
# --------------------------------------------------------------------------


def test_is_native_credential_distinguishes_legacy_tokens():
    assert is_native_credential({"auth_type": AUTH_TYPE_NATIVE_OIDC, "key": "k"})
    # A legacy proxy-minted token has no auth_type and must stay supported.
    assert not is_native_credential({"key": "k"})
    assert not is_native_credential(None)
    assert not is_native_credential({})


def test_build_native_credential_stores_bearer_under_key():
    credential = build_native_credential(
        base_url=BASE_URL + "/",
        metadata=make_metadata(),
        token=make_token(),
        now=1000.0,
    )
    assert credential["key"] == "access-1"
    assert credential["auth_type"] == AUTH_TYPE_NATIVE_OIDC
    assert credential["schema_version"] == 2
    # Trailing slash normalised so the origin comparison on refresh is stable.
    assert credential["base_url"] == BASE_URL
    assert credential["issuer"] == ISSUER
    assert credential["client_id"] == CLIENT_ID
    assert credential["scopes"] == ["openid", "profile"]
    assert credential["timestamp"] == 1000.0


def test_build_native_credential_never_persists_flow_secrets():
    credential = build_native_credential(base_url=BASE_URL, metadata=make_metadata(), token=make_token())
    forbidden = {
        "code",
        "code_verifier",
        "code_challenge",
        "state",
        "device_code",
        "user_code",
        "client_secret",
        "id_token",
    }
    assert forbidden.isdisjoint(credential)


def test_rotated_refresh_token_replaces_the_stored_one():
    credential = build_native_credential(
        base_url=BASE_URL,
        metadata=make_metadata(),
        token=make_token(refresh_token="refresh-2"),
        previous_refresh_token="refresh-1",
    )
    assert credential["refresh_token"] == "refresh-2"


def test_omitted_refresh_token_retains_the_previous_one():
    credential = build_native_credential(
        base_url=BASE_URL,
        metadata=make_metadata(),
        token=make_token(refresh_token=None),
        previous_refresh_token="refresh-1",
    )
    assert credential["refresh_token"] == "refresh-1"


def test_no_refresh_token_anywhere_omits_the_field():
    credential = build_native_credential(
        base_url=BASE_URL,
        metadata=make_metadata(),
        token=make_token(refresh_token=None),
    )
    assert "refresh_token" not in credential


def test_save_credential_is_owner_only(home):
    save_credential({"auth_type": AUTH_TYPE_NATIVE_OIDC, "key": "secret"})
    path = get_cli_token_file_path()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    with open(path) as handle:
        assert json.load(handle)["key"] == "secret"


# --------------------------------------------------------------------------
# freshness
# --------------------------------------------------------------------------


def test_fresh_credential_does_not_need_refresh():
    assert not needs_refresh({"expires_at": 2000.0}, now=1000.0)


def test_credential_inside_the_buffer_needs_refresh():
    expires_at = 2000.0
    assert needs_refresh({"expires_at": expires_at}, now=expires_at - REFRESH_BUFFER_SECONDS + 1)


def test_expired_credential_needs_refresh():
    assert needs_refresh({"expires_at": 500.0}, now=1000.0)


@pytest.mark.parametrize("expires_at", [None, "soon", True, {}])
def test_malformed_expiry_fails_closed(expires_at):
    assert needs_refresh({"expires_at": expires_at}, now=1000.0)


# --------------------------------------------------------------------------
# verification against LiteLLM
# --------------------------------------------------------------------------


def test_verification_probes_the_user_accessible_models_route():
    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers, timeout=timeout)
        return FakeResponse(200)

    verify_token_with_litellm(BASE_URL + "/", "access-1", get=fake_get)
    assert seen["url"] == f"{BASE_URL}/v1/models"
    assert seen["headers"] == {"Authorization": "Bearer access-1"}
    assert seen["timeout"] == 10


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_token_raises_auth_rejected(status):
    with pytest.raises(NativeOIDCAuthRejected) as excinfo:
        verify_token_with_litellm(BASE_URL, "access-1", get=lambda *a, **k: FakeResponse(status))
    message = str(excinfo.value)
    assert "issuer" in message and "audience" in message
    # The token itself must never appear in an error surfaced to the user.
    assert "access-1" not in message


@pytest.mark.parametrize("status", [200, 404, 500, 503])
def test_non_auth_statuses_are_tolerated(status):
    verify_token_with_litellm(BASE_URL, "access-1", get=lambda *a, **k: FakeResponse(status))


def test_unreachable_proxy_is_not_an_auth_rejection():
    def fake_get(*args, **kwargs):
        raise requests.ConnectionError("boom")

    with pytest.raises(NativeOIDCError) as excinfo:
        verify_token_with_litellm(BASE_URL, "access-1", get=fake_get)
    assert not isinstance(excinfo.value, NativeOIDCAuthRejected)
    assert "ConnectionError" in str(excinfo.value)


# --------------------------------------------------------------------------
# cross-process lock
# --------------------------------------------------------------------------


def test_lock_is_released_on_exit(home):
    with refresh_lock() as _:
        lock_path = get_cli_token_file_path() + ".lock"
        assert os.path.exists(lock_path)
    assert not os.path.exists(lock_path)


def test_lock_is_released_when_the_body_raises(home):
    lock_path = get_cli_token_file_path() + ".lock"
    with pytest.raises(ValueError):
        with refresh_lock():
            raise ValueError("boom")
    assert not os.path.exists(lock_path)


def test_second_holder_times_out_while_the_first_holds(home):
    with refresh_lock():
        with pytest.raises(NativeOIDCError) as excinfo:
            with refresh_lock(timeout=0.2, sleep=lambda _: None):
                pass
    assert "another process" in str(excinfo.value)


def test_stale_lock_is_reclaimed(home):
    lock_path = get_cli_token_file_path() + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as handle:
        handle.write("99999")
    stale = time.time() - creds.LOCK_STALE_AFTER_SECONDS - 10
    os.utime(lock_path, (stale, stale))

    with refresh_lock(timeout=1.0, sleep=lambda _: None):
        assert os.path.exists(lock_path)
    assert not os.path.exists(lock_path)


def test_lock_serialises_concurrent_holders(home):
    order = []

    def worker(tag):
        with refresh_lock(timeout=5.0):
            order.append(f"{tag}-in")
            time.sleep(0.05)
            order.append(f"{tag}-out")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # No holder may enter before the previous one has left.
    assert len(order) == 6
    for index in range(0, 6, 2):
        assert order[index].endswith("-in")
        assert order[index + 1] == order[index].replace("-in", "-out")


# --------------------------------------------------------------------------
# refresh
# --------------------------------------------------------------------------


@pytest.fixture
def stored(home):
    """An expired native credential on disk -- the state that triggers a refresh."""
    credential = build_native_credential(
        base_url=BASE_URL,
        metadata=make_metadata(),
        token=make_token(
            access_token="old-access",
            refresh_token="refresh-1",
            expires_at=time.time() - 10,
        ),
    )
    save_credential(credential)
    return credential


def install_token_endpoint(monkeypatch, payload, recorder=None):
    def fake_post_form(url, data, **kwargs):
        if recorder is not None:
            recorder.update(url=url, data=dict(data))
        return type("R", (), {"status_code": 200, "payload": payload})()

    monkeypatch.setattr(creds, "post_form", fake_post_form)


def test_refresh_rotates_and_persists(stored, monkeypatch):
    recorder = {}
    install_token_endpoint(
        monkeypatch,
        {
            "access_token": "new-access",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "refresh-2",
        },
        recorder,
    )
    verified = {}

    result = refresh_native_credential(
        stored,
        verify=lambda base_url, token: verified.update(base_url=base_url, token=token),
        fetch_metadata=lambda base_url: make_metadata(),
        fetch_provider=lambda issuer: make_provider(),
    )

    assert result["key"] == "new-access"
    assert result["refresh_token"] == "refresh-2"
    # Persisted, so a sibling process sees the rotation.
    assert load_cli_token()["key"] == "new-access"
    # Verified against the proxy before it was trusted.
    assert verified == {"base_url": BASE_URL, "token": "new-access"}
    # Public client: refresh_token grant, client_id, and no secret.
    assert recorder["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "refresh-1",
        "client_id": CLIENT_ID,
    }


def test_refresh_uses_the_rediscovered_token_endpoint(stored, monkeypatch):
    recorder = {}
    install_token_endpoint(
        monkeypatch,
        {"access_token": "new-access", "token_type": "Bearer", "expires_in": 3600},
        recorder,
    )
    seen_issuers = []

    refresh_native_credential(
        # A stale endpoint smuggled into the credential file must be ignored.
        {**stored, "token_endpoint": "https://attacker.example.com/token"},
        verify=lambda *a: None,
        fetch_metadata=lambda base_url: make_metadata(),
        fetch_provider=lambda issuer: (
            seen_issuers.append(issuer) or make_provider(token_endpoint=f"{ISSUER}/rediscovered")
        ),
    )

    assert seen_issuers == [ISSUER]
    assert recorder["url"] == f"{ISSUER}/rediscovered"


@pytest.mark.parametrize(
    "metadata",
    [
        make_metadata(issuer="https://evil.example.com"),
        make_metadata(client_id="other-client"),
    ],
)
def test_refresh_rejects_a_changed_trust_anchor(stored, monkeypatch, metadata):
    install_token_endpoint(monkeypatch, {"access_token": "new", "token_type": "Bearer"})

    with pytest.raises(NativeOIDCError) as excinfo:
        refresh_native_credential(
            stored,
            verify=lambda *a: None,
            fetch_metadata=lambda base_url: metadata,
            fetch_provider=lambda issuer: make_provider(issuer),
        )
    assert "run 'lite login' again" in str(excinfo.value)
    # The stored credential is untouched.
    assert load_cli_token()["key"] == "old-access"


def test_refresh_defers_to_a_sibling_process_that_already_refreshed(stored, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("provider must not be contacted")

    monkeypatch.setattr(creds, "post_form", explode)
    save_credential({**stored, "key": "sibling-access", "expires_at": time.time() + 3600})

    result = refresh_native_credential(
        stored,
        verify=explode,
        fetch_metadata=explode,
        fetch_provider=explode,
    )
    assert result["key"] == "sibling-access"


def test_refresh_ignores_a_sibling_credential_for_another_proxy(stored, monkeypatch):
    install_token_endpoint(
        monkeypatch,
        {"access_token": "new-access", "token_type": "Bearer", "expires_in": 3600},
    )
    save_credential(
        {
            **stored,
            "base_url": "https://other-proxy.example.com",
            "key": "other-access",
            "expires_at": time.time() + 3600,
        }
    )

    result = refresh_native_credential(
        stored,
        verify=lambda *a: None,
        fetch_metadata=lambda base_url: make_metadata(),
        fetch_provider=lambda issuer: make_provider(),
    )
    assert result["key"] == "new-access"


def test_refresh_without_a_refresh_token_tells_the_user_to_log_in(home):
    with pytest.raises(NativeOIDCError, match="no refresh token"):
        refresh_native_credential({"base_url": BASE_URL, "issuer": ISSUER})


def test_refresh_without_a_base_url_tells_the_user_to_log_in(home):
    with pytest.raises(NativeOIDCError, match="no base_url"):
        refresh_native_credential({"refresh_token": "refresh-1"})


def test_refresh_surfaces_a_rejected_new_token(stored, monkeypatch):
    install_token_endpoint(
        monkeypatch,
        {"access_token": "new-access", "token_type": "Bearer", "expires_in": 3600},
    )

    def reject(base_url, token):
        raise NativeOIDCAuthRejected("nope")

    with pytest.raises(NativeOIDCAuthRejected):
        refresh_native_credential(
            stored,
            verify=reject,
            fetch_metadata=lambda base_url: make_metadata(),
            fetch_provider=lambda issuer: make_provider(),
        )
    # An unverified token is never written.
    assert load_cli_token()["key"] == "old-access"


def test_refresh_surfaces_the_oauth_error_code(stored, monkeypatch):
    def fake_post_form(url, data, **kwargs):
        return type("R", (), {"status_code": 400, "payload": {"error": "invalid_grant"}})()

    monkeypatch.setattr(creds, "post_form", fake_post_form)

    with pytest.raises(NativeOIDCError, match="invalid_grant"):
        refresh_native_credential(
            stored,
            verify=lambda *a: None,
            fetch_metadata=lambda base_url: make_metadata(),
            fetch_provider=lambda issuer: make_provider(),
        )
    assert load_cli_token()["key"] == "old-access"


def test_refresh_releases_the_lock_on_failure(stored, monkeypatch):
    monkeypatch.setattr(creds, "post_form", lambda *a, **k: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        refresh_native_credential(
            stored,
            verify=lambda *a: None,
            fetch_metadata=lambda base_url: make_metadata(),
            fetch_provider=lambda issuer: make_provider(),
        )
    assert not os.path.exists(get_cli_token_file_path() + ".lock")
