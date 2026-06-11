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
