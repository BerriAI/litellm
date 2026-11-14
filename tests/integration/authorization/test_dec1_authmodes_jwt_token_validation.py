"""JWT auth: which bearer tokens a proxy with `general_settings.enable_jwt_auth: true` accepts, and the exact
refusal for every way a token can be bad.

Contract (litellm/proxy/auth/user_api_key_auth.py `user_api_key_auth_builder`, litellm/proxy/auth/handle_jwt.py
`JWTHandler.auth_jwt`):

* A bearer value with exactly three dot separated segments is a JWT and takes the JWT path; anything else
  (an `sk-` virtual key, the master key, a two segment string) takes the virtual key path on the same proxy.
* The JWT's `kid` header picks a key from the JWKS at `JWT_PUBLIC_KEY_URL`. The signature is checked with
  PyJWT against the asymmetric algorithms RS*, PS*, ES* and EdDSA only, so HS256 and `alg: none` are refused.
* An expired token is a 401 `expired_key` "Token Expired". Every other decode failure is a 401 `auth_error`
  "Authentication Error, Validation fails: <PyJWT reason>". A `kid` no JWKS publishes is a 401 naming the
  JWKS URL list and the `kid`. A token whose header cannot be parsed is a 401 with PyJWT's reason and no
  "Validation fails" prefix, because the header is read before the guarded decode.
* JWT auth needs an enterprise license; without one a JWT is a 403 naming `premium_user`.

The Rust rewrite of `gateway-auth` must keep these status codes, error types and message texts: clients and
dashboards key on them, and the split between 401 `expired_key` and 401 `auth_error` drives token refresh.

One proxy serves the module: `litellm_jwtauth: {user_id_jwt_field: sub, user_id_upsert: true}`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import jwt
import pytest

from tests.integration._support.client import string_value
from tests.integration.authorization.dec1_authmodes_support import (
    ADMIN_ONLY_PREFIX,
    NOT_PREMIUM,
    JwtRig,
    chat_as,
    jwt_rig,
    message_of,
    no_matching_key,
    rsa_key,
    served_body,
    spend_row,
    unique,
    user_row,
    validation_failure,
)
from tests.integration.authorization.dec1_keys_support import assert_denied, digest, upstream_requests


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-token-validation"), user_id_jwt_field="sub", user_id_upsert=True
    ) as built:
        yield built


def test_a_valid_rs256_token_is_served_and_reaches_the_provider(rig: JwtRig) -> None:
    """A token signed by the published RS256 key with a fresh `sub` is sent on a chat completion.

    The proxy must answer 200 with the provider's reply, and the scripted upstream must have received the
    prompt, so JWT auth admitted the call all the way to the provider.
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("sub")
        text: Final = unique("valid-token")
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=subject), model, text=text))
        assert len(upstream_requests(rig.proxy, text)) == 1


def test_user_id_upsert_creates_the_user_row_named_by_the_sub_claim(rig: JwtRig) -> None:
    """A token whose `sub` names a user the proxy has never seen is sent once.

    With `user_id_upsert: true` the request must be served and a `LiteLLM_UserTable` row whose `user_id`
    is exactly the `sub` value must exist afterwards.
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("upsert")
        assert user_row(rig.proxy, subject) is None
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=subject), model))
        created: Final = user_row(rig.proxy, subject)
        assert created is not None and created["user_id"] == subject, created
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_the_spend_row_of_a_jwt_call_is_attributed_to_the_sub_claim_and_never_stores_the_token(rig: JwtRig) -> None:
    """A JWT caller makes one served chat completion.

    The `LiteLLM_SpendLogs` row for that response id must carry `user` equal to the `sub` claim, and its
    `api_key` column must not be the raw JWT (the proxy hashes or blanks credentials before storing them).
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("spend")
        token: Final = rig.idp.token(sub=subject)
        row: Final = spend_row(rig.proxy, chat_as(rig.proxy, token, model))
        assert row["user"] == subject, row
        assert row["api_key"] != token and token not in str(row), row
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_an_expired_token_is_a_401_expired_key_token_expired(rig: JwtRig) -> None:
    """A correctly signed token whose `exp` is a minute in the past is sent.

    The proxy must answer 401 with `{"error": {"message": "Token Expired", "type": "expired_key",
    "param": null, "code": "401"}}` (handle_jwt.py:1404). Clients refresh on this type, so it must not
    collapse into the generic `auth_error`.
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        now: Final = int(time.time())
        token: Final = rig.idp.token(sub=unique("expired"), iat=now - 600, exp=now - 60)
        assert_denied(chat_as(rig.proxy, token, model), status=401, error_type="expired_key", message="Token Expired")


def test_a_token_that_is_not_yet_valid_is_refused_with_the_nbf_reason(rig: JwtRig) -> None:
    """A correctly signed token whose `nbf` is ten minutes in the future is sent.

    The proxy must answer 401 `auth_error` with exactly "Authentication Error, Validation fails: The token
    is not yet valid (nbf)". The proxy decodes with zero leeway, so there is no clock skew allowance.
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = rig.idp.token(sub=unique("nbf"), nbf=int(time.time()) + 600)
        assert_denied(
            chat_as(rig.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=validation_failure("The token is not yet valid (nbf)"),
        )


def test_a_token_issued_in_the_future_is_refused_with_the_iat_reason(rig: JwtRig) -> None:
    """A correctly signed token whose `iat` is ten minutes in the future (and `exp` after that) is sent.

    Pinned current behaviour of the pinned PyJWT 2.15: the proxy must answer 401 `auth_error` with
    "Authentication Error, Validation fails: The token is not yet valid (iat)". A rewrite on another JWT
    library may silently accept such tokens.
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        now: Final = int(time.time())
        token: Final = rig.idp.token(sub=unique("iat"), iat=now + 600, exp=now + 900)
        assert_denied(
            chat_as(rig.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=validation_failure("The token is not yet valid (iat)"),
        )


def test_a_token_signed_by_a_different_key_under_the_published_kid_fails_signature_verification(
    rig: JwtRig,
) -> None:
    """A token carries the published `kid` but is signed by a different RSA private key.

    The proxy must answer 401 `auth_error` "Authentication Error, Validation fails: Signature verification
    failed" and must not call the provider.
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        impostor: Final = rsa_key(kid=rig.idp.key.kid)
        text: Final = unique("forged")
        refused: Final = chat_as(rig.proxy, rig.idp.token(impostor, sub=unique("forged")), model, text=text)
        assert_denied(
            refused,
            status=401,
            error_type="auth_error",
            message=validation_failure("Signature verification failed"),
        )
        assert upstream_requests(rig.proxy, text) == []


def test_a_token_with_an_unknown_kid_names_the_jwks_url_and_the_kid(rig: JwtRig) -> None:
    """A token signed by the real key but carrying a `kid` the JWKS does not publish is sent.

    The proxy must answer 401 `auth_error` with exactly "Authentication Error, No matching public key found.
    keys=['<JWT_PUBLIC_KEY_URL>'], kid=<kid>" (handle_jwt.py:1003, the URL list is a Python list repr).
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        kid: Final = unique("unknown-kid")
        token: Final = rig.idp.token(sub=unique("kid"), headers={"kid": kid})
        assert_denied(
            chat_as(rig.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=no_matching_key((rig.idp.jwks_url,), kid),
        )


def test_an_hs256_token_under_the_published_kid_is_refused_as_a_disallowed_algorithm(rig: JwtRig) -> None:
    """A token is HMAC-signed (HS256) with a shared secret and carries the published RSA `kid`.

    HS* is deliberately absent from `JWTHandler.SUPPORTED_JWT_ALGORITHMS`, which blocks the classic
    "public key as HMAC secret" confusion. The proxy must answer 401 `auth_error` "Authentication Error,
    Validation fails: The specified alg value is not allowed".
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        now: Final = int(time.time())
        token: Final = jwt.encode(
            {"sub": unique("hs256"), "iat": now, "exp": now + 300},
            "dec1-authmodes-shared-secret-that-is-long-enough",
            algorithm="HS256",
            headers={"kid": rig.idp.key.kid},
        )
        assert_denied(
            chat_as(rig.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=validation_failure("The specified alg value is not allowed"),
        )


def test_an_unsigned_alg_none_token_is_refused_as_a_disallowed_algorithm(rig: JwtRig) -> None:
    """A token with `alg: none`, an empty signature segment and the published `kid` is sent.

    It still has three segments, so it takes the JWT path. The proxy must answer 401 `auth_error`
    "Authentication Error, Validation fails: The specified alg value is not allowed" and never serve it.
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        now: Final = int(time.time())
        token: Final = jwt.encode(
            {"sub": unique("none"), "iat": now, "exp": now + 300},
            None,
            algorithm="none",
            headers={"kid": rig.idp.key.kid},
        )
        assert token.endswith(".")
        assert_denied(
            chat_as(rig.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=validation_failure("The specified alg value is not allowed"),
        )


def test_a_three_segment_value_with_an_unparseable_header_is_a_401_with_the_pyjwt_reason(rig: JwtRig) -> None:
    """The bearer value `not-a-jwt.x.y` has three segments but its header is not base64url JSON.

    `auth_jwt` reads the header before its guarded decode, so the proxy must answer 401 `auth_error` with
    exactly "Authentication Error, Invalid header padding" (no "Validation fails" prefix).
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        assert_denied(
            chat_as(rig.proxy, "not-a-jwt.x.y", model),
            status=401,
            error_type="auth_error",
            message="Authentication Error, Invalid header padding",
        )


def test_an_id_jag_grant_presented_as_a_bearer_token_is_refused(rig: JwtRig) -> None:
    """A correctly signed token whose header has `typ: oauth-id-jag+jwt` is sent as a bearer token.

    An ID-JAG is an authorization grant for the token endpoint, never an access token, so the proxy must
    answer 401 `auth_error` "Authentication Error, an ID-JAG is an authorization grant, not a bearer
    token; redeem it at the token endpoint" (handle_jwt.py:1376).
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = rig.idp.token(sub=unique("idjag"), headers={"typ": "oauth-id-jag+jwt"})
        assert_denied(
            chat_as(rig.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=(
                "Authentication Error, an ID-JAG is an authorization grant, not a bearer token; "
                "redeem it at the token endpoint"
            ),
        )


def test_a_token_without_an_exp_claim_is_accepted(rig: JwtRig) -> None:
    """A correctly signed token with `sub` and `iat` but no `exp` claim is sent.

    Pinned current behaviour: PyJWT requires no claims by default and the proxy passes no `require` option,
    so a token that never expires is served with 200. A rewrite that starts requiring `exp` would lock out
    identity providers that mint such tokens, and a rewrite should decide that on purpose.
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("no-exp")
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=subject, omit=("exp",)), model))
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_virtual_key_on_the_same_proxy_takes_the_key_path(rig: JwtRig) -> None:
    """A virtual key generated on the JWT proxy is used for a chat completion.

    An `sk-` value has no dots, so it is not a JWT and must be served by the virtual key path with 200; its
    spend row must carry the key's sha256 hash in `api_key`, which a JWT row never does.
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        row: Final = spend_row(rig.proxy, chat_as(rig.proxy, key, model))
        assert row["api_key"] == digest(key), row


def test_the_master_key_still_works_on_a_jwt_proxy(rig: JwtRig) -> None:
    """The proxy's master key is used for a chat completion and for `GET /key/list`.

    Enabling JWT auth must not disable the master key: both calls must answer 200.
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        assert served_body(chat_as(rig.proxy, rig.proxy.key, model))
        assert rig.proxy.request("GET", "/key/list").status_code == 200


def test_a_virtual_key_revoked_on_a_jwt_proxy_is_still_refused_as_a_key(rig: JwtRig) -> None:
    """A virtual key is generated on the JWT proxy, then deleted, then used.

    The deleted `sk-` key must take the key path and get the generic 401 `token_not_found_in_db` "Invalid API
    key" refusal; JWT auth must not make the proxy treat an unknown `sk-` value any differently.
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = string_value(rig.proxy.post("/key/generate", {"models": [model]})["key"])
        rig.proxy.post("/key/delete", {"keys": [key]})
        assert_denied(
            chat_as(rig.proxy, key, model),
            status=401,
            error_type="token_not_found_in_db",
            message="Authentication Error, Invalid API key provided.",
            param="key",
        )


def test_a_two_segment_value_is_not_a_jwt_and_gets_the_virtual_key_refusal_without_the_jwt_hint(rig: JwtRig) -> None:
    """The bearer value `dec1authmodes.twosegments` has one dot, so it is not JWT shaped.

    It takes the virtual key path and must get 401 `auth_error` "LiteLLM Virtual Key expected.
    Received=dec1****ents, expected to start with 'sk-'." with no `enable_jwt_auth` hint, because JWT auth
    is already on (user_api_key_auth.py:2952).
    """
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        value: Final = "dec1authmodes.twosegments"
        assert_denied(
            chat_as(rig.proxy, value, model),
            status=401,
            error_type="auth_error",
            message=f"LiteLLM Virtual Key expected. Received={value[:4]}****{value[-4:]}, expected to start with 'sk-'.",
        )


def test_a_non_admin_jwt_user_is_refused_on_an_admin_only_management_route(rig: JwtRig) -> None:
    """A plain JWT user (no admin scope) calls `POST /user/new`.

    The JWT path resolves an internal user, and the shared non-admin route check must refuse the management
    route with 401 `auth_error` whose message starts "Authentication Error, Only proxy admin can be used to
    generate, delete, update info for new keys/users/teams. Route=/user/new. Your role=". No user is created.
    UNVERIFIED: the role text after "Your role=" depends on the upserted row's default role, so only the
    prefix is pinned.
    """
    with rig.proxy.scenario() as scenario:
        subject: Final = unique("non-admin")
        target: Final = unique("never-created")
        refused: Final = rig.proxy.request("POST", "/user/new", {"user_id": target}, key=rig.idp.token(sub=subject))
        assert refused.status_code == 401, refused.text
        assert message_of(refused).startswith(ADMIN_ONLY_PREFIX.split(" Your role=")[0].format(route="/user/new")), (
            refused.text
        )
        assert user_row(rig.proxy, target) is None
        if user_row(rig.proxy, subject) is not None:
            scenario.cleanups.callback(scenario.delete_user, subject)


def test_without_a_license_a_jwt_is_refused_with_403_naming_premium_user(tmp_path: Path) -> None:
    """A JWT proxy is booted with `LITELLM_LICENSE` removed, and a correctly signed token is sent.

    The proxy must answer 403 `auth_error`, `param: premium_user`, with exactly "JWT Auth is an enterprise
    only feature. <not premium user text>" (user_api_key_auth.py:2419), while the master key keeps working.
    """
    with (
        jwt_rig(tmp_path, remove_environment=("LITELLM_LICENSE",), user_id_jwt_field="sub") as unlicensed,
        unlicensed.proxy.scenario() as scenario,
    ):
        model: Final = scenario.model()
        assert_denied(
            chat_as(unlicensed.proxy, unlicensed.idp.token(sub=unique("unlicensed")), model),
            status=403,
            error_type="auth_error",
            param="premium_user",
            message=f"JWT Auth is an enterprise only feature. {NOT_PREMIUM}",
        )
        assert served_body(chat_as(unlicensed.proxy, unlicensed.proxy.key, model))
