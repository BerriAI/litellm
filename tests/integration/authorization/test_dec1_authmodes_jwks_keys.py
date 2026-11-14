"""JWT auth: how the proxy finds, caches and refreshes the identity provider's signing keys.

Contract (litellm/proxy/auth/handle_jwt.py `get_public_key`, `_get_public_key_from_jwks_url`,
`_cached_with_stale_fallback`, `_refresh_or_serve_stale`, `_refresh_or_record_outage`, `parse_keys`):

* `JWT_PUBLIC_KEY_URL` may list several JWKS URLs separated by commas; they are tried in order and the first
  one that publishes the token's `kid` wins. A URL containing `.well-known/openid-configuration` is read as an
  OIDC discovery document and its `jwks_uri` is fetched.
* A fetched key set is cached for `litellm_jwtauth.public_key_ttl` seconds (default 600). Within that window
  the provider is not asked again, even for a `kid` it does not hold, so a rotated key is unknown until the
  cache expires.
* A transport failure is retried three times and then answered 503 `auth_provider_unavailable`; the failure is
  remembered for 30 seconds so a sustained outage costs one fetch per window. A provider that answers with an
  HTTP error status is not an outage: that is a 401.
* After the TTL lapses during an outage, the last good key set keeps serving for `public_key_stale_ttl`
  (default 3600). `public_key_stale_ttl: 0` fails closed with the 503.
* With one published key a token without `kid` uses it; with several, a token must name its `kid`.

Each test owns its proxy (function scope), because every case depends on an empty key cache.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway, eventually, held
from tests.integration.authorization.dec1_authmodes_support import (
    JWKS_UNAVAILABLE_PREFIX,
    IdentityProvider,
    JwtRig,
    authmodes_proxy,
    chat_as,
    ec_key,
    identity_provider,
    jwt_rig,
    jwt_settings,
    message_of,
    no_matching_key,
    rsa_key,
    served_body,
    unique,
)
from tests.integration.authorization.dec1_keys_support import assert_denied, error_of

pytestmark: Final = pytest.mark.timeout(240)


@dataclass(frozen=True, slots=True)
class TwoProviders:
    first: IdentityProvider
    second: IdentityProvider
    proxy: Gateway


@pytest.fixture(scope="module")
def two_providers(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TwoProviders]:
    with (
        identity_provider(rsa_key(unique("first-kid"))) as first,
        identity_provider(rsa_key(unique("second-kid"))) as second,
        authmodes_proxy(
            tmp_path_factory.mktemp("dec1-authmodes-two-jwks"),
            environment={"JWT_PUBLIC_KEY_URL": f"{first.jwks_url}, {second.jwks_url}"},
            general_settings=jwt_settings(user_id_jwt_field="sub", user_id_upsert=True),
        ) as proxy,
    ):
        yield TwoProviders(first, second, proxy)


def _status(rig: JwtRig, model: str, **claims: str) -> int:
    return chat_as(rig.proxy, rig.idp.token(sub=unique("probe"), **claims), model).status_code


def _assert_jwks_unavailable(response: httpx.Response, url: str) -> None:
    assert response.status_code == 503, f"{response.status_code} {response.text}"
    error: Final = error_of(response)
    assert error["type"] == "auth_provider_unavailable", response.text
    assert error["code"] == "503" and error["param"] is None, response.text
    message: Final = message_of(response)
    assert message.startswith(JWKS_UNAVAILABLE_PREFIX), message
    assert message.endswith(f"fetching {url} after 3 attempts"), message


def test_a_token_signed_by_the_key_at_the_second_jwks_url_is_served(two_providers: TwoProviders) -> None:
    """`JWT_PUBLIC_KEY_URL` lists two JWKS URLs (with a space after the comma). A token is signed by the key
    only the second URL publishes.

    The proxy must skip the first URL's key miss and answer 200 using the second URL's key.
    """
    with two_providers.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("second")
        assert served_body(chat_as(two_providers.proxy, two_providers.second.token(sub=subject), model))
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_token_signed_by_the_key_at_the_first_jwks_url_is_served(two_providers: TwoProviders) -> None:
    """Same two URL proxy; a token is signed by the key the first URL publishes. The proxy must answer 200."""
    with two_providers.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("first")
        assert served_body(chat_as(two_providers.proxy, two_providers.first.token(sub=subject), model))
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_an_unknown_kid_with_two_jwks_urls_names_both_urls(two_providers: TwoProviders) -> None:
    """Same two URL proxy; a token carries a `kid` neither URL publishes.

    The proxy must answer 401 `auth_error` "Authentication Error, No matching public key found.
    keys=['<first>', '<second>'], kid=<kid>": both URLs, stripped of the space, in configured order.
    """
    with two_providers.proxy.scenario() as scenario:
        model: Final = scenario.model()
        kid: Final = unique("nowhere")
        token: Final = two_providers.first.token(sub=unique("nowhere"), headers={"kid": kid})
        assert_denied(
            chat_as(two_providers.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=no_matching_key((two_providers.first.jwks_url, two_providers.second.jwks_url), kid),
        )


def test_an_oidc_discovery_url_is_followed_to_its_jwks_uri(tmp_path: Path) -> None:
    """`JWT_PUBLIC_KEY_URL` is the provider's `/.well-known/openid-configuration` document, whose `jwks_uri`
    names `/jwks`.

    A correctly signed token must be served with 200, and the provider must have answered exactly one
    discovery request and one JWKS request.
    """
    with (
        jwt_rig(tmp_path, discovery=True, user_id_jwt_field="sub", user_id_upsert=True) as rig,
        rig.proxy.scenario() as scenario,
    ):
        model: Final = scenario.model()
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=unique("discovery")), model))
        assert (rig.idp.discovery_gets, rig.idp.fetches()) == (1, 1)


def test_the_key_set_is_fetched_once_for_many_requests(tmp_path: Path) -> None:
    """Five tokens for five different subjects, all signed by the same key, are sent in a row.

    All five must be served and the JWKS endpoint must have been fetched exactly once: the key set is
    cached for `public_key_ttl` (600 seconds by default), not per token or per subject.
    """
    with jwt_rig(tmp_path, user_id_jwt_field="sub", user_id_upsert=True) as rig, rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        for _ in range(5):
            assert served_body(chat_as(rig.proxy, rig.idp.token(sub=unique("cached")), model))
        assert rig.idp.fetches() == 1


def test_an_es256_key_from_the_jwks_verifies_an_es256_token(tmp_path: Path) -> None:
    """The provider publishes a single P-256 EC key and signs an ES256 token with it.

    ES256 is in the supported algorithm list, so the proxy must answer 200. A rewrite that only wires RSA
    would break every identity provider that signs with EC keys.
    """
    with (
        jwt_rig(tmp_path, keys=(ec_key(),), user_id_jwt_field="sub", user_id_upsert=True) as rig,
        rig.proxy.scenario() as scenario,
    ):
        model: Final = scenario.model()
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=unique("ec")), model))


def test_a_token_without_a_kid_uses_the_only_published_key(tmp_path: Path) -> None:
    """The provider publishes exactly one key, and a token signed by it carries no usable `kid` (null).

    `parse_keys` takes a lone key when the token names no `kid`, so the proxy must answer 200.
    """
    with jwt_rig(tmp_path, user_id_jwt_field="sub", user_id_upsert=True) as rig, rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=unique("no-kid"), headers={"kid": None}), model))


def test_a_token_without_a_kid_is_refused_when_several_keys_are_published(tmp_path: Path) -> None:
    """The provider publishes two keys, and a token signed by the first carries no usable `kid` (null).

    With more than one key a `kid` is required, so the proxy must answer 401 `auth_error` "Authentication
    Error, No matching public key found. keys=['<url>'], kid=None".
    """
    with (
        jwt_rig(tmp_path, keys=(rsa_key(), rsa_key()), user_id_jwt_field="sub", user_id_upsert=True) as rig,
        rig.proxy.scenario() as scenario,
    ):
        model: Final = scenario.model()
        token: Final = rig.idp.token(sub=unique("no-kid-many"), headers={"kid": None})
        assert_denied(
            chat_as(rig.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=no_matching_key((rig.idp.jwks_url,), None),
        )


def test_a_jwks_endpoint_answering_500_is_a_401_quoting_the_status_and_body(tmp_path: Path) -> None:
    """The JWKS endpoint answers HTTP 500 with a JSON body on the first and only fetch.

    An HTTP error status is not treated as an outage: the proxy must answer 401 `auth_error` with exactly
    "Authentication Error, JWT Auth: JWKS endpoint <url> returned status 500: <body>" (handle_jwt.py:958).
    A rewrite may well turn this into the 503 the transport failure gets; that would be a visible change.
    """
    with jwt_rig(tmp_path, user_id_jwt_field="sub", user_id_upsert=True) as rig:
        rig.idp.set_mode("error_500")
        with rig.proxy.scenario() as scenario:
            model: Final = scenario.model()
            assert_denied(
                chat_as(rig.proxy, rig.idp.token(sub=unique("idp-500")), model),
                status=401,
                error_type="auth_error",
                message=(
                    f"Authentication Error, JWT Auth: JWKS endpoint {rig.idp.jwks_url} returned status 500: "
                    '{"error": "dec1 identity provider down"}'
                ),
            )


def test_an_unreachable_jwks_endpoint_is_a_503_after_three_attempts(tmp_path: Path) -> None:
    """The JWKS endpoint drops every connection without answering, before any key was ever cached.

    The proxy must answer 503 `auth_provider_unavailable` with the "Service Unavailable, the identity
    provider's JWKS endpoint is temporarily unreachable ..." message ending "fetching <url> after 3
    attempts", and the provider must have seen exactly three fetches (JWKS_FETCH_ATTEMPTS).
    """
    with jwt_rig(tmp_path, user_id_jwt_field="sub", user_id_upsert=True) as rig:
        rig.idp.set_mode("drop")
        with rig.proxy.scenario() as scenario:
            model: Final = scenario.model()
            _assert_jwks_unavailable(chat_as(rig.proxy, rig.idp.token(sub=unique("down")), model), rig.idp.jwks_url)
            assert rig.idp.fetches() == 3


def test_a_recorded_outage_is_not_retried_for_thirty_seconds(tmp_path: Path) -> None:
    """The JWKS endpoint drops connections, a first token gets the 503, then the endpoint recovers at once and
    a second token is sent right away.

    The failure is remembered for JWKS_UNREACHABLE_BACKOFF_SECONDS (30), so the second request must also get
    the 503 and the provider must not have been asked again (still three fetches).
    """
    with jwt_rig(tmp_path, user_id_jwt_field="sub", user_id_upsert=True) as rig:
        rig.idp.set_mode("drop")
        with rig.proxy.scenario() as scenario:
            model: Final = scenario.model()
            _assert_jwks_unavailable(chat_as(rig.proxy, rig.idp.token(sub=unique("down")), model), rig.idp.jwks_url)
            rig.idp.set_mode("serve")
            _assert_jwks_unavailable(chat_as(rig.proxy, rig.idp.token(sub=unique("again")), model), rig.idp.jwks_url)
            assert rig.idp.fetches() == 3


def test_cached_keys_keep_serving_while_the_jwks_endpoint_is_down(tmp_path: Path) -> None:
    """One token is served (which caches the key set), then the JWKS endpoint starts dropping connections.

    Within the default 600 second `public_key_ttl` the proxy must keep answering 200 for new tokens and must
    not contact the provider again (exactly one fetch).
    """
    with jwt_rig(tmp_path, user_id_jwt_field="sub", user_id_upsert=True) as rig, rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=unique("warm")), model))
        rig.idp.set_mode("drop")
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=unique("during-outage")), model))
        assert rig.idp.fetches() == 1


def test_after_the_ttl_lapses_during_an_outage_the_stale_key_set_keeps_serving(tmp_path: Path) -> None:
    """`public_key_ttl: 1` with the default `public_key_stale_ttl` (3600). One token warms the cache, then the
    endpoint starts dropping connections.

    Requests must keep answering 200 for a three second hold that outlasts the TTL, served from the
    last-known-good copy, and the provider must have been asked again (the refresh was attempted).
    """
    with (
        jwt_rig(tmp_path, user_id_jwt_field="sub", user_id_upsert=True, public_key_ttl=1) as rig,
        rig.proxy.scenario() as scenario,
    ):
        model: Final = scenario.model()
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=unique("warm")), model))
        rig.idp.set_mode("drop")
        held(lambda: _status(rig, model), lambda status: status == 200, holding=3, seconds=30)
        assert rig.idp.fetches() > 1


def test_a_zero_stale_ttl_fails_closed_once_the_ttl_lapses_during_an_outage(tmp_path: Path) -> None:
    """`public_key_ttl: 1` and `public_key_stale_ttl: 0`. One token warms the cache, then the endpoint starts
    dropping connections.

    Once the one second TTL lapses the proxy must stop trusting the cached keys and answer 503
    `auth_provider_unavailable` (polled, never slept).
    """
    with (
        jwt_rig(
            tmp_path, user_id_jwt_field="sub", user_id_upsert=True, public_key_ttl=1, public_key_stale_ttl=0
        ) as rig,
        rig.proxy.scenario() as scenario,
    ):
        model: Final = scenario.model()
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=unique("warm")), model))
        rig.idp.set_mode("drop")
        refused: Final = eventually(
            lambda: chat_as(rig.proxy, rig.idp.token(sub=unique("closed")), model),
            lambda response: response.status_code == 503,
            seconds=30,
        )
        _assert_jwks_unavailable(refused, rig.idp.jwks_url)


def test_a_rotated_key_is_unknown_until_the_cached_key_set_expires(tmp_path: Path) -> None:
    """The cache is warmed with key A under the default 600 second TTL, then the provider publishes only a
    new key B and a token signed by B is sent.

    Pinned current behaviour: an unknown `kid` does not trigger a refetch, so the proxy must answer 401
    "No matching public key found ... kid=<B>" and the provider must still have been fetched only once.
    Identity providers that rotate without overlap lock users out for up to `public_key_ttl`.
    """
    with jwt_rig(tmp_path, user_id_jwt_field="sub", user_id_upsert=True) as rig, rig.proxy.scenario() as scenario:
        model: Final = scenario.model()
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=unique("old-key")), model))
        rotated: Final = rsa_key()
        rig.idp.publish(rotated)
        assert_denied(
            chat_as(rig.proxy, rig.idp.token(rotated, sub=unique("new-key")), model),
            status=401,
            error_type="auth_error",
            message=no_matching_key((rig.idp.jwks_url,), rotated.kid),
        )
        assert rig.idp.fetches() == 1


def test_with_a_short_ttl_a_rotated_key_is_accepted_and_the_retired_key_refused(tmp_path: Path) -> None:
    """`public_key_ttl: 1`. The cache is warmed with key A, then the provider publishes only a new key B.

    Polling, a token signed by B must eventually be served with 200, and after that a token signed by the
    retired key A must be refused with 401 "No matching public key found ... kid=<A>".
    """
    with (
        jwt_rig(tmp_path, user_id_jwt_field="sub", user_id_upsert=True, public_key_ttl=1) as rig,
        rig.proxy.scenario() as scenario,
    ):
        model: Final = scenario.model()
        retired: Final = rig.idp.key
        assert served_body(chat_as(rig.proxy, rig.idp.token(sub=unique("old-key")), model))
        rotated: Final = rsa_key()
        rig.idp.publish(rotated)
        eventually(
            lambda: chat_as(rig.proxy, rig.idp.token(rotated, sub=unique("rotated")), model).status_code,
            lambda status: status == 200,
            seconds=30,
        )
        assert_denied(
            chat_as(rig.proxy, rig.idp.token(retired, sub=unique("retired")), model),
            status=401,
            error_type="auth_error",
            message=no_matching_key((rig.idp.jwks_url,), retired.kid),
        )
