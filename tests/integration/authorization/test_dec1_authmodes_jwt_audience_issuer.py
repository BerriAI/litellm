"""JWT auth: audience and issuer scoping, globally through `JWT_AUDIENCE` / `JWT_ISSUER` and per issuer through
`litellm_jwtauth.issuers`.

Contract (litellm/proxy/auth/handle_jwt.py `_build_decode_kwargs`, `_get_configured_issuer`,
`_auth_jwt_with_issuer`):

* With `JWT_AUDIENCE` set, PyJWT verifies `aud`: a missing claim and a non-matching claim are both 401
  `auth_error` "Validation fails: ..." with PyJWT's own reason. A list `aud` passes when it contains the
  expected audience. `JWT_ISSUER` does the same for `iss`. With neither set only signature and time claims
  are checked (and a warning is logged once).
* An `issuers` entry whose `issuer` equals the token's unverified `iss` takes over: the token is checked only
  against that issuer's `jwks_url` and `audience`, and the issuer's claim mappings (here `user_id_jwt_field`)
  replace the global ones. A key lookup miss on that path names the single issuer JWKS URL without list
  brackets. A token whose `iss` matches no entry falls back to the global path.

These scopes are what stop a token minted for another application on the same identity provider from
being accepted, so the Rust auth must keep every refusal below.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration.authorization.dec1_authmodes_support import (
    IdentityProvider,
    JwtRig,
    authmodes_proxy,
    chat_as,
    identity_provider,
    jwt_rig,
    jwt_settings,
    rsa_key,
    served_body,
    spend_row,
    unique,
    validation_failure,
)
from tests.integration.authorization.dec1_keys_support import assert_denied

AUDIENCE: Final = "dec1-authmodes-api"
ISSUER: Final = "https://idp.dec1-authmodes.test"
ISSUER_A: Final = "https://issuer-a.dec1-authmodes.test"
AUDIENCE_A: Final = "dec1-authmodes-issuer-a-api"


@pytest.fixture(scope="module")
def scoped(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-audience"),
        environment={"JWT_AUDIENCE": AUDIENCE, "JWT_ISSUER": ISSUER},
        user_id_jwt_field="sub",
        user_id_upsert=True,
    ) as built:
        yield built


@dataclass(frozen=True, slots=True)
class IssuerRig:
    issuer_a: IdentityProvider
    global_idp: IdentityProvider
    proxy: Gateway


@pytest.fixture(scope="module")
def issuers(tmp_path_factory: pytest.TempPathFactory) -> Iterator[IssuerRig]:
    with (
        identity_provider(rsa_key(unique("issuer-a-kid"))) as issuer_a,
        identity_provider(rsa_key(unique("global-kid"))) as global_idp,
        authmodes_proxy(
            tmp_path_factory.mktemp("dec1-authmodes-issuers"),
            environment={"JWT_PUBLIC_KEY_URL": global_idp.jwks_url},
            general_settings=jwt_settings(
                user_id_jwt_field="sub",
                user_id_upsert=True,
                issuers=[
                    {
                        "issuer": ISSUER_A,
                        "jwks_url": issuer_a.jwks_url,
                        "audience": AUDIENCE_A,
                        "user_id_jwt_field": "upn",
                    }
                ],
            ),
        ) as proxy,
    ):
        yield IssuerRig(issuer_a, global_idp, proxy)


def test_a_token_with_the_configured_audience_and_issuer_is_served(scoped: JwtRig) -> None:
    """`JWT_AUDIENCE` and `JWT_ISSUER` are set; a token carries exactly that `aud` and `iss`.

    The proxy must answer 200 with the provider's reply.
    """
    with scoped.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("scoped")
        assert served_body(chat_as(scoped.proxy, scoped.idp.token(sub=subject, aud=AUDIENCE, iss=ISSUER), model))
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_token_for_another_audience_is_refused(scoped: JwtRig) -> None:
    """A correctly signed token from the configured issuer carries `aud: some-other-app`.

    The proxy must answer 401 `auth_error` "Authentication Error, Validation fails: Audience doesn't match".
    This is the cross-application token the audience setting exists to block.
    """
    with scoped.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = scoped.idp.token(sub=unique("aud"), aud="some-other-app", iss=ISSUER)
        assert_denied(
            chat_as(scoped.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=validation_failure("Audience doesn't match"),
        )


def test_a_token_without_an_aud_claim_is_refused_as_missing_a_required_claim(scoped: JwtRig) -> None:
    """With `JWT_AUDIENCE` set, a correctly signed token carries `iss` but no `aud` claim.

    The proxy must answer 401 `auth_error` with exactly 'Authentication Error, Validation fails: Token is
    missing the "aud" claim'.
    """
    with scoped.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = scoped.idp.token(sub=unique("no-aud"), iss=ISSUER)
        assert_denied(
            chat_as(scoped.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=validation_failure('Token is missing the "aud" claim'),
        )


def test_a_list_audience_containing_the_configured_audience_is_served(scoped: JwtRig) -> None:
    """A token's `aud` is the list `["first-app", <JWT_AUDIENCE>]`.

    PyJWT accepts a list audience that contains the expected one, so the proxy must answer 200.
    """
    with scoped.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("list-aud")
        token: Final = scoped.idp.token(sub=subject, aud=["first-app", AUDIENCE], iss=ISSUER)
        assert served_body(chat_as(scoped.proxy, token, model))
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_list_audience_without_the_configured_audience_is_refused(scoped: JwtRig) -> None:
    """A token's `aud` is the list `["first-app", "second-app"]`, neither of which is `JWT_AUDIENCE`.

    The proxy must answer 401 `auth_error` "Authentication Error, Validation fails: Audience doesn't match".
    """
    with scoped.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = scoped.idp.token(sub=unique("list-aud-miss"), aud=["first-app", "second-app"], iss=ISSUER)
        assert_denied(
            chat_as(scoped.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=validation_failure("Audience doesn't match"),
        )


def test_a_token_from_another_issuer_is_refused(scoped: JwtRig) -> None:
    """A correctly signed token has the configured `aud` but `iss: https://elsewhere.test`.

    The proxy must answer 401 `auth_error` "Authentication Error, Validation fails: Invalid issuer".
    """
    with scoped.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = scoped.idp.token(sub=unique("iss"), aud=AUDIENCE, iss="https://elsewhere.test")
        assert_denied(
            chat_as(scoped.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=validation_failure("Invalid issuer"),
        )


def test_a_token_without_an_iss_claim_is_refused_as_missing_a_required_claim(scoped: JwtRig) -> None:
    """With `JWT_ISSUER` set, a correctly signed token carries the right `aud` but no `iss`.

    The proxy must answer 401 `auth_error` 'Authentication Error, Validation fails: Token is missing the
    "iss" claim'.
    """
    with scoped.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = scoped.idp.token(sub=unique("no-iss"), aud=AUDIENCE)
        assert_denied(
            chat_as(scoped.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=validation_failure('Token is missing the "iss" claim'),
        )


def test_a_token_for_a_configured_issuer_is_checked_against_that_issuers_jwks_and_claim_mapping(
    issuers: IssuerRig,
) -> None:
    """`issuers` lists issuer A with its own JWKS, audience and `user_id_jwt_field: upn`. A token from A
    carries `iss` A, A's audience, and both a `sub` and a `upn`.

    The proxy must answer 200 and the spend row's `user` must be the `upn` value, not the `sub`, because the
    issuer's claim mapping overrides the global `user_id_jwt_field: sub`.
    """
    with issuers.proxy.scenario() as scenario:
        model: Final = scenario.model()
        upn: Final = unique("upn")
        token: Final = issuers.issuer_a.token(iss=ISSUER_A, aud=AUDIENCE_A, sub=unique("ignored-sub"), upn=upn)
        row: Final = spend_row(issuers.proxy, chat_as(issuers.proxy, token, model))
        assert row["user"] == upn, row
        scenario.cleanups.callback(scenario.delete_user, upn)


def test_a_token_claiming_a_configured_issuer_but_signed_by_another_key_names_only_that_issuers_jwks(
    issuers: IssuerRig,
) -> None:
    """A token says `iss` A but is signed by the global identity provider's key (its `kid` too).

    The issuer path looks only at A's JWKS, so the proxy must answer 401 `auth_error` with exactly
    "Authentication Error, No matching public key found. keys=<A's JWKS URL>, kid=<global kid>": a single
    URL with no list brackets (handle_jwt.py:983), unlike the global path.
    """
    with issuers.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = issuers.global_idp.token(iss=ISSUER_A, aud=AUDIENCE_A, upn=unique("upn"))
        assert_denied(
            chat_as(issuers.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=(
                "Authentication Error, No matching public key found. "
                f"keys={issuers.issuer_a.jwks_url}, kid={issuers.global_idp.key.kid}"
            ),
        )


def test_a_configured_issuer_enforces_its_own_audience(issuers: IssuerRig) -> None:
    """A token from issuer A, signed by A, carries `aud: some-other-app` instead of A's audience.

    The proxy must answer 401 `auth_error` "Authentication Error, Validation fails: Audience doesn't match",
    even though no global `JWT_AUDIENCE` is set on this proxy.
    """
    with issuers.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = issuers.issuer_a.token(iss=ISSUER_A, aud="some-other-app", upn=unique("upn"))
        assert_denied(
            chat_as(issuers.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=validation_failure("Audience doesn't match"),
        )


def test_a_token_whose_issuer_is_not_listed_falls_back_to_the_global_jwks(issuers: IssuerRig) -> None:
    """A token signed by the global identity provider carries an `iss` that no `issuers` entry lists.

    `issuers` is additive routing, not an allow list, so the proxy must validate it against
    `JWT_PUBLIC_KEY_URL`, answer 200, and attribute spend to the global `user_id_jwt_field` (`sub`).
    """
    with issuers.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("global-sub")
        token: Final = issuers.global_idp.token(iss="https://unlisted.dec1-authmodes.test", sub=subject)
        row: Final = spend_row(issuers.proxy, chat_as(issuers.proxy, token, model))
        assert row["user"] == subject, row
        scenario.cleanups.callback(scenario.delete_user, subject)
