"""Tests for native OIDC login orchestration and the CLI commands using it."""

import time

import pytest
from click.testing import CliRunner

from litellm.proxy.client.cli.commands.auth import login, print_token, whoami
from litellm.proxy.client.cli.native_oidc import login as login_module
from litellm.proxy.client.cli.native_oidc.credentials import (
    build_native_credential,
    save_credential,
)
from litellm.proxy.client.cli.native_oidc.errors import (
    NativeOIDCError,
    NativeOIDCUnavailable,
)
from litellm.proxy.client.cli.native_oidc.login import (
    FLOW_BROWSER,
    FLOW_DEVICE,
    run_native_login,
    select_flow,
)
from litellm.proxy.client.cli.native_oidc.metadata import (
    NativeOIDCMetadata,
    ProviderMetadata,
)
from litellm.proxy.client.cli.native_oidc.tokens import TokenResponse

BASE_URL = "https://proxy.example.com"
ISSUER = "https://idp.example.com"
CLIENT_ID = "litellm-cli"

AUTH_MODULE = "litellm.proxy.client.cli.commands.auth"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def make_metadata():
    return NativeOIDCMetadata(issuer=ISSUER, client_id=CLIENT_ID, scopes=("openid", "profile"))


def make_provider(
    *,
    authorization_endpoint=f"{ISSUER}/authorize",
    device_authorization_endpoint=f"{ISSUER}/device",
    grant_types=("authorization_code", "urn:ietf:params:oauth:grant-type:device_code"),
):
    return ProviderMetadata(
        issuer=ISSUER,
        authorization_endpoint=authorization_endpoint,
        token_endpoint=f"{ISSUER}/token",
        device_authorization_endpoint=device_authorization_endpoint,
        response_types_supported=("code",),
        grant_types_supported=grant_types,
        code_challenge_methods_supported=("S256",),
        token_endpoint_auth_methods_supported=("none",),
    )


def make_token(access_token="access-1"):
    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_at=time.time() + 3600,
        refresh_token="refresh-1",
        scopes=("openid", "profile"),
    )


# --------------------------------------------------------------------------
# flow selection
# --------------------------------------------------------------------------


def test_auto_prefers_the_browser_flow():
    assert select_flow(make_provider(), "auto", open_browser=True) == FLOW_BROWSER


def test_auto_uses_the_device_flow_when_no_browser_is_available():
    assert select_flow(make_provider(), "auto", open_browser=False) == FLOW_DEVICE


def test_auto_falls_back_to_device_when_the_browser_flow_is_unsupported():
    provider = make_provider(authorization_endpoint=None)
    assert select_flow(provider, "auto", open_browser=True) == FLOW_DEVICE


def test_auto_falls_back_to_browser_when_the_device_flow_is_unsupported():
    provider = make_provider(device_authorization_endpoint=None, grant_types=("authorization_code",))
    assert select_flow(provider, "auto", open_browser=False) == FLOW_BROWSER


def test_explicit_browser_is_never_downgraded_to_device():
    provider = make_provider(authorization_endpoint=None)
    with pytest.raises(NativeOIDCError):
        select_flow(provider, FLOW_BROWSER, open_browser=True)


def test_explicit_device_is_never_downgraded_to_browser():
    provider = make_provider(device_authorization_endpoint=None, grant_types=("authorization_code",))
    with pytest.raises(NativeOIDCError):
        select_flow(provider, FLOW_DEVICE, open_browser=True)


def test_no_usable_flow_raises_a_specific_reason():
    provider = make_provider(
        authorization_endpoint=None,
        device_authorization_endpoint=None,
        grant_types=("client_credentials",),
    )
    with pytest.raises(NativeOIDCError):
        select_flow(provider, "auto", open_browser=True)


# --------------------------------------------------------------------------
# run_native_login
# --------------------------------------------------------------------------


@pytest.fixture
def native_login_env(home, monkeypatch):
    calls = {"browser": 0, "device": 0, "verified": None, "saved": None}

    monkeypatch.setattr(login_module, "fetch_native_oidc_metadata", lambda base_url: make_metadata())
    monkeypatch.setattr(login_module, "fetch_provider_metadata", lambda issuer: make_provider())

    def fake_browser(metadata, provider, *, open_browser=True, echo=None):
        calls["browser"] += 1
        calls["open_browser"] = open_browser
        return make_token("browser-token")

    def fake_device(metadata, provider, *, open_browser=True, echo=None):
        calls["device"] += 1
        calls["open_browser"] = open_browser
        return make_token("device-token")

    monkeypatch.setattr(login_module, "run_browser_flow", fake_browser)
    monkeypatch.setattr(login_module, "run_device_flow", fake_device)
    monkeypatch.setattr(
        login_module,
        "verify_token_with_litellm",
        lambda base_url, token: calls.update(verified=(base_url, token)),
    )
    return calls


def test_native_login_saves_a_verified_credential(native_login_env):
    credential = run_native_login(BASE_URL, flow=FLOW_BROWSER, echo=lambda *a: None)

    assert credential["key"] == "browser-token"
    assert credential["issuer"] == ISSUER
    # Verified against the proxy before it was stored.
    assert native_login_env["verified"] == (BASE_URL, "browser-token")
    # And actually persisted.
    from litellm.litellm_core_utils.cli_token_utils import load_cli_token

    assert load_cli_token()["key"] == "browser-token"


def test_native_login_does_not_store_a_token_the_proxy_rejects(native_login_env, monkeypatch, home):
    def reject(base_url, token):
        raise NativeOIDCError("rejected")

    monkeypatch.setattr(login_module, "verify_token_with_litellm", reject)

    with pytest.raises(NativeOIDCError):
        run_native_login(BASE_URL, flow=FLOW_BROWSER, echo=lambda *a: None)

    from litellm.litellm_core_utils.cli_token_utils import load_cli_token

    assert load_cli_token() is None


def test_no_browser_reaches_the_device_flow(native_login_env):
    run_native_login(BASE_URL, open_browser=False, echo=lambda *a: None)
    assert native_login_env["device"] == 1
    assert native_login_env["browser"] == 0
    assert native_login_env["open_browser"] is False


# --------------------------------------------------------------------------
# `lite login`
# --------------------------------------------------------------------------


def invoke_login(args=()):
    return CliRunner().invoke(login, list(args), obj={"base_url": BASE_URL})


def test_flow_proxy_never_attempts_native_login(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("native login must not be attempted")

    monkeypatch.setattr(f"{AUTH_MODULE}.run_native_login", explode)
    monkeypatch.setattr(f"{AUTH_MODULE}._run_proxy_sso_login", lambda base_url: None)

    assert invoke_login(["--flow", "proxy"]).exit_code == 0


def test_auto_falls_back_when_the_proxy_offers_no_native_oidc(monkeypatch):
    fell_back = []

    def unavailable(*args, **kwargs):
        raise NativeOIDCUnavailable("no native_oidc advertised")

    monkeypatch.setattr(f"{AUTH_MODULE}.run_native_login", unavailable)
    monkeypatch.setattr(
        f"{AUTH_MODULE}._run_proxy_sso_login",
        lambda base_url: fell_back.append(base_url),
    )

    result = invoke_login()
    assert result.exit_code == 0
    assert fell_back == [BASE_URL]
    assert "proxy-mediated SSO" in result.output


def test_a_native_failure_never_falls_back_to_proxy_sso(monkeypatch):
    """The security property: no silent downgrade once native OIDC is offered."""

    def failed(*args, **kwargs):
        raise NativeOIDCError("issuer mismatch")

    monkeypatch.setattr(f"{AUTH_MODULE}.run_native_login", failed)

    def explode(base_url):
        raise AssertionError("must not downgrade to proxy-mediated SSO")

    monkeypatch.setattr(f"{AUTH_MODULE}._run_proxy_sso_login", explode)

    result = invoke_login()
    assert result.exit_code == 1
    assert "issuer mismatch" in result.output


@pytest.mark.parametrize("flow", ["browser", "device"])
def test_an_explicit_native_flow_does_not_fall_back_when_unavailable(monkeypatch, flow):
    def unavailable(*args, **kwargs):
        raise NativeOIDCUnavailable("no native_oidc advertised")

    monkeypatch.setattr(f"{AUTH_MODULE}.run_native_login", unavailable)

    def explode(base_url):
        raise AssertionError("must not downgrade to proxy-mediated SSO")

    monkeypatch.setattr(f"{AUTH_MODULE}._run_proxy_sso_login", explode)

    result = invoke_login(["--flow", flow])
    assert result.exit_code == 1
    assert "not available" in result.output


def test_login_reports_the_issuer_and_never_the_token(monkeypatch, home):
    credential = build_native_credential(base_url=BASE_URL, metadata=make_metadata(), token=make_token("super-secret"))
    monkeypatch.setattr(f"{AUTH_MODULE}.run_native_login", lambda *a, **k: credential)
    monkeypatch.setattr("litellm.proxy.client.cli.interface.show_commands", lambda: None)

    result = invoke_login()
    assert result.exit_code == 0
    assert ISSUER in result.output
    assert "openid profile" in result.output
    assert "super-secret" not in result.output
    assert "refresh-1" not in result.output


def test_login_passes_the_requested_flow_through(monkeypatch, home):
    seen = {}

    def fake_native_login(base_url, *, flow, open_browser):
        seen.update(base_url=base_url, flow=flow, open_browser=open_browser)
        raise NativeOIDCUnavailable("stop here")

    monkeypatch.setattr(f"{AUTH_MODULE}.run_native_login", fake_native_login)
    monkeypatch.setattr(f"{AUTH_MODULE}._run_proxy_sso_login", lambda base_url: None)

    invoke_login(["--no-browser"])
    assert seen == {"base_url": BASE_URL, "flow": "auto", "open_browser": False}


# --------------------------------------------------------------------------
# `lite auth print-token`
# --------------------------------------------------------------------------


def store_native_credential(*, expires_in=3600, access_token="access-1"):
    credential = build_native_credential(
        base_url=BASE_URL,
        metadata=make_metadata(),
        token=TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_at=time.time() + expires_in,
            refresh_token="refresh-1",
            scopes=("openid",),
        ),
    )
    save_credential(credential)
    return credential


def invoke_print_token():
    return CliRunner().invoke(print_token, obj={"base_url_explicit": False})


def test_print_token_emits_a_fresh_native_token_without_refreshing(monkeypatch, home):
    store_native_credential(access_token="fresh-token")

    def explode(*args, **kwargs):
        raise AssertionError("a fresh token must not trigger a refresh")

    monkeypatch.setattr(f"{AUTH_MODULE}.refresh_native_credential", explode)

    result = invoke_print_token()
    assert result.exit_code == 0
    assert result.stdout.strip() == "fresh-token"


def test_print_token_refreshes_an_expired_native_token(monkeypatch, home):
    stored = store_native_credential(expires_in=-10, access_token="stale-token")
    refreshed = {**stored, "key": "renewed-token", "expires_at": time.time() + 3600}
    monkeypatch.setattr(f"{AUTH_MODULE}.refresh_native_credential", lambda token_data: refreshed)

    result = invoke_print_token()
    assert result.exit_code == 0
    # apiKeyHelper contract: stdout is the token and nothing else.
    assert result.stdout.strip() == "renewed-token"


def test_print_token_fails_when_the_refresh_fails(monkeypatch, home):
    store_native_credential(expires_in=-10)

    def failed(token_data):
        raise NativeOIDCError("refresh token revoked")

    monkeypatch.setattr(f"{AUTH_MODULE}.refresh_native_credential", failed)

    result = invoke_print_token()
    assert result.exit_code == 1
    assert result.stdout.strip() == ""
    assert "refresh token revoked" in result.stderr
    assert "lite login" in result.stderr


# --------------------------------------------------------------------------
# `lite whoami`
# --------------------------------------------------------------------------


def test_whoami_describes_a_native_credential(home):
    store_native_credential()
    result = CliRunner().invoke(whoami)

    assert result.exit_code == 0
    assert "native OIDC" in result.output
    assert ISSUER in result.output
    assert CLIENT_ID in result.output
    # The credential itself is never echoed.
    assert "access-1" not in result.output
    assert "refresh-1" not in result.output


def test_whoami_flags_an_expired_native_credential(home):
    store_native_credential(expires_in=-10)
    result = CliRunner().invoke(whoami)
    assert "expired" in result.output
    assert "refreshed on next use" in result.output


def test_whoami_tells_the_user_to_log_in_without_a_refresh_token(home):
    credential = store_native_credential(expires_in=-10)
    save_credential({k: v for k, v in credential.items() if k != "refresh_token"})

    result = CliRunner().invoke(whoami)
    assert "no refresh token" in result.output
    assert "lite login" in result.output
