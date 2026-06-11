from __future__ import annotations

from litellm.proxy.auth_v2 import OIDCProviderConfig
from litellm.proxy.auth_v2.oidc.router import _provider_key, _user_from_userinfo
from litellm.proxy.auth_v2.resolver import InMemoryIdentityStore


def test_userinfo_maps_standard_claims_to_scim_user():
    user = _user_from_userinfo(
        {
            "sub": "idp-subject-123",
            "preferred_username": "dana",
            "email": "dana@example.com",
            "name": "Dana D",
        }
    )
    assert user.external_id == "idp-subject-123"
    assert user.user_name == "dana"
    assert user.display_name == "Dana D"


def test_userinfo_falls_back_to_email_when_no_preferred_username():
    user = _user_from_userinfo({"sub": "s1", "email": "eve@example.com"})
    assert user.user_name == "eve@example.com"


def test_provider_key_sanitizes_issuer_url():
    key = _provider_key(
        OIDCProviderConfig(issuer="https://Login.Example.com/realm", audience=["x"])
    )
    assert key == "https-login-example-com-realm"
    assert " " not in key


async def test_callback_seam_upserts_userinfo_into_store():
    store = InMemoryIdentityStore()
    userinfo = {
        "sub": "idp-subject-123",
        "preferred_username": "dana",
        "email": "dana@example.com",
        "name": "Dana D",
    }
    # this is exactly what the OIDC callback does: map userinfo -> SCIM user -> upsert
    stored = await store.upsert_user(_user_from_userinfo(userinfo))

    assert stored.id  # store assigned an id
    fetched = await store.get_user(stored.id)
    assert fetched is not None
    assert fetched.external_id == "idp-subject-123"
    assert fetched.user_name == "dana"


async def _oidc_login_session_roles(userinfo, provider):
    # mirror the callback's identity build: map userinfo, gate roles, store a session,
    # then authenticate + resolve through the same seam a request would
    from litellm.proxy.auth_v2.authenticators import _apply_role_policy
    from litellm.proxy.auth_v2.oidc.router import _mapped_claims
    from litellm.proxy.auth_v2.session import SessionAuthenticator, SessionStore

    from auth_v2_helpers import make_request

    claims = _mapped_claims(userinfo)
    _apply_role_policy(claims, provider)
    store = SessionStore()
    sid = store.create_session(
        {"method": "oidc", "subject": userinfo["sub"], "claims": claims}
    )
    authenticator = SessionAuthenticator("litellm_session", store)
    credential = await authenticator.authenticate(
        make_request(cookies={"litellm_session": sid})
    )
    principal = await InMemoryIdentityStore().resolve(credential)
    return [role.value for role in principal.roles]


async def test_oidc_login_platform_role_denied_by_default():
    provider = OIDCProviderConfig(issuer="https://idp.example.com", audience=["x"])
    userinfo = {
        "sub": "u",
        "email": "e@x.com",
        "roles": ["platform_admin", "org_admin"],
    }
    assert await _oidc_login_session_roles(userinfo, provider) == []


async def test_oidc_login_roles_filtered_to_allowlist():
    provider = OIDCProviderConfig(
        issuer="https://idp.example.com", audience=["x"], allowed_roles=["org_admin"]
    )
    userinfo = {
        "sub": "u",
        "email": "e@x.com",
        "roles": ["platform_admin", "org_admin"],
    }
    assert await _oidc_login_session_roles(userinfo, provider) == ["org_admin"]
