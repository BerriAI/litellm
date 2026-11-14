import json
import os
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal

import httpx
import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import Gateway, eventually, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.database_relay import held_statement_relay
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

KEY_ID: Final = "integration-jwt-map-existing-key"
MAPPING_INSERT: Final = b'INSERT INTO "public"."LiteLLM_JWTKeyMapping"'

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]


def _hash(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _config(
    directory: Path,
    claim_field: str,
    *,
    behavior: Literal["auto_register", "link_only", "reject"] = "auto_register",
    map_existing_key: bool = True,
    user_id_upsert: bool = False,
    issuers: tuple[dict[str, JsonValue], ...] = (),
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"] = {
        **config["general_settings"],
        "enable_jwt_auth": True,
        "litellm_jwtauth": {
            "user_id_jwt_field": "sub",
            "user_email_jwt_field": "email",
            "virtual_key_claim_field": claim_field,
            "unregistered_jwt_client_behavior": behavior,
            "auto_register_map_existing_key": map_existing_key,
            "user_id_upsert": user_id_upsert,
            "issuers": list(issuers),
        },
    }
    path: Final = directory / f"jwt_map_existing_key_{claim_field}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _issuer() -> Iterator[tuple[rsa.RSAPrivateKey, str]]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    jwks: Final = json.dumps({"keys": [{**public_jwk, "kid": KEY_ID, "use": "sig", "alg": "RS256"}]}).encode()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=jwks)

    with wire_server(respond) as server:
        yield private_key, server.url


def _token(private_key: rsa.RSAPrivateKey, subject: str, **claims: str) -> str:
    now: Final = int(time.time())
    return jwt.encode(
        {"sub": subject, **claims, "iat": now, "exp": now + 300},
        private_key,
        algorithm="RS256",
        headers={"kid": KEY_ID},
    )


def _chat(candidate: Gateway, model: str, token: str) -> httpx.Response:
    return candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "map existing key control"}]},
        key=token,
    )


def _mapped_token(claim_name: str, claim_value: str) -> str:
    rows: Final = read_rows(
        'SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_name = %s AND jwt_claim_value = %s',
        (claim_name, claim_value),
    )
    assert len(rows) == 1, rows
    return string_value(rows[0]["token"])


def _user_key_hashes(user: str) -> frozenset[str]:
    rows: Final = read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE user_id = %s', (user,))
    return frozenset(string_value(row["token"]) for row in rows)


def _billed_key(response: httpx.Response) -> str:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT api_key FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (str(response.json()["id"]),)
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return string_value(rows[0]["api_key"])


def test_first_jwt_call_reuses_the_newest_durable_key_and_skips_every_ineligible_newer_key(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        older_durable: Final = scenario.key(user_id=user)
        durable: Final = scenario.key(user_id=user)
        skipped: Final = {
            "older_durable": older_durable,
            "expiring": scenario.key(user_id=user, duration="1h"),
            "auto_registered_look_alike": scenario.key(user_id=user, metadata={"auto_registered": True}),
            "other_team": scenario.key(user_id=user, team_id=scenario.team()),
            "blocked": scenario.key(user_id=user),
        }
        gateway.post("/key/block", {"key": skipped["blocked"]})
        keys_before: Final = _user_key_hashes(user)

        with owned_proxy(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks_url}, config=_config(tmp_path, "sub")
        ) as candidate:
            response: Final = _chat(candidate, model, _token(private_key, user))

        assert response.status_code == 200, response.text
        mapped: Final = _mapped_token("sub", user)
        assert mapped == _hash(durable), {
            "mapped_to": next((name for name, key in skipped.items() if _hash(key) == mapped), mapped)
        }
        assert _user_key_hashes(user) == keys_before, "a key was minted although a reusable one existed"
        assert _billed_key(response) == _hash(durable)


@pytest.mark.parametrize(
    ("behavior", "map_existing_key"), [("auto_register", True), ("link_only", False), ("link_only", True)]
)
def test_route_restricted_keys_are_linked_without_bypassing_other_key_checks(
    gateway: Gateway,
    tmp_path: Path,
    behavior: Literal["auto_register", "link_only"],
    map_existing_key: bool,
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        scenario.key(user_id=user)
        restricted: Final = scenario.key(user_id=user, allowed_routes=["/v1/chat/completions"])
        scenario.key(user_id=user, duration="1h")
        scenario.key(user_id=user, metadata={"auto_registered": True})
        scenario.key(user_id=user, team_id=scenario.team())
        blocked: Final = scenario.key(user_id=user)
        gateway.post("/key/block", {"key": blocked})
        keys_before: Final = _user_key_hashes(user)
        token: Final = _token(private_key, user)

        with owned_proxy(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": jwks_url},
            config=_config(
                tmp_path,
                "sub",
                behavior=behavior,
                map_existing_key=map_existing_key,
            ),
        ) as candidate:
            response: Final = _chat(candidate, model, token)
            repeated: Final = _chat(candidate, model, token)
            restricted_route: Final = candidate.request("GET", "/v1/models", key=token)

        assert response.status_code == 200, response.text
        assert repeated.status_code == 200, repeated.text
        assert restricted_route.status_code == 403, restricted_route.text
        assert _mapped_token("sub", user) == _hash(restricted)
        assert _user_key_hashes(user) == keys_before, "a new key was minted instead of linking an existing key"
        assert _billed_key(response) == _hash(restricted)
        assert _billed_key(repeated) == _hash(restricted)


@pytest.mark.parametrize("behavior", ["auto_register", "link_only"])
def test_management_route_keys_are_linked_by_default_and_keep_their_route_permissions(
    gateway: Gateway, tmp_path: Path, behavior: Literal["auto_register", "link_only"]
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        target: Final = f"integration-{uuid.uuid4().hex}"
        scenario.key(user_id=user)
        management_key: Final = scenario.key(user_id=user, allowed_routes=["/user/new"])
        keys_before: Final = _user_key_hashes(user)
        token: Final = _token(private_key, user)
        with owned_proxy(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks_url}, config=_config(tmp_path, "sub", behavior=behavior)
        ) as candidate:
            denied: Final = _chat(candidate, model, token)
            allowed: Final = candidate.request(
                "POST",
                "/user/new",
                {"user_id": target, "auto_create_key": False, "user_role": "internal_user"},
                key=token,
            )
            assert allowed.status_code == 200, allowed.text
            scenario.cleanups.callback(scenario.delete_user, target)
            repeated: Final = _chat(candidate, model, token)
        assert denied.status_code == 403, denied.text
        assert repeated.status_code == 403, repeated.text
        assert allowed.json()["user_id"] == target
        assert _mapped_token("sub", user) == _hash(management_key)
        assert _user_key_hashes(user) == keys_before


def test_user_matched_by_email_instead_of_sub_still_reuses_their_existing_key(gateway: Gateway, tmp_path: Path) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        email: Final = f"integration-{uuid.uuid4().hex}@example.com"
        user: Final = scenario.user(user_role="internal_user", user_email=email)
        existing: Final = scenario.key(user_id=user)
        subject: Final = f"integration-idp-subject-{uuid.uuid4().hex}"

        with owned_proxy(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks_url}, config=_config(tmp_path, "sub")
        ) as candidate:
            response: Final = _chat(candidate, model, _token(private_key, subject, email=email.upper()))

        assert response.status_code == 200, response.text
        assert _mapped_token("sub", subject) == _hash(existing)
        assert _user_key_hashes(user) == frozenset({_hash(existing)}), "a key was minted for an email-matched user"
        assert _billed_key(response) == _hash(existing)


def test_shared_client_claim_never_maps_a_second_user_onto_the_first_users_personal_key(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        first_user: Final = scenario.user(user_role="internal_user")
        second_user: Final = scenario.user(user_role="internal_user")
        personal: Final = scenario.key(user_id=first_user)
        client_id: Final = f"integration-shared-client-{uuid.uuid4().hex}"

        with owned_proxy(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks_url}, config=_config(tmp_path, "client_id")
        ) as candidate:
            first: Final = _chat(candidate, model, _token(private_key, first_user, client_id=client_id))
            second: Final = _chat(candidate, model, _token(private_key, second_user, client_id=client_id))

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        mapped: Final = _mapped_token("client_id", client_id)
        assert mapped != _hash(personal), "the shared client claim was mapped to the first user's personal key"
        assert (_billed_key(first), _billed_key(second)) == (mapped, mapped)
        assert read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (_hash(personal),)) == []


@pytest.mark.parametrize("behavior", ["auto_register", "link_only"])
def test_concurrent_first_jwt_calls_both_succeed_on_one_surviving_mapped_key(
    gateway: Gateway, tmp_path: Path, behavior: Literal["auto_register", "link_only"]
) -> None:
    writer_url: Final = os.environ.get("INTEGRATION_PROXY_DATABASE_URL") or os.environ["DATABASE_URL"]
    with (
        _issuer() as (private_key, jwks_url),
        gateway.scenario() as scenario,
        held_statement_relay(writer_url, MAPPING_INSERT) as (relay, relayed_url),
    ):
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        existing: Final = scenario.key(user_id=user) if behavior == "link_only" else None
        token: Final = _token(private_key, user)
        overrides: Final = {
            "JWT_PUBLIC_KEY_URL": jwks_url,
            "DATABASE_URL": relayed_url,
            "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
        }

        with (
            owned_proxy(
                gateway,
                tmp_path,
                overrides,
                config=_config(tmp_path, "sub", behavior=behavior, map_existing_key=False),
            ) as candidate,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            held_call: Final = pool.submit(_chat, candidate, model, token)
            assert relay.held.wait(60), "the first call never reached its mapping insert"
            racing: Final = _chat(candidate, model, token)
            relay.release()
            held: Final = held_call.result(timeout=60)

        assert racing.status_code == 200, racing.text
        assert held.status_code == 200, held.text
        keys: Final = _user_key_hashes(user)
        assert len(keys) == 1, keys
        if existing is not None:
            assert keys == frozenset({_hash(existing)})
        assert _mapped_token("sub", user) in keys
        assert (_billed_key(held), _billed_key(racing)) == (_mapped_token("sub", user),) * 2


@pytest.mark.parametrize("ineligible_keys", [False, True])
def test_link_only_rejects_without_writes_then_links_a_newly_available_key(
    gateway: Gateway, tmp_path: Path, ineligible_keys: bool
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        if ineligible_keys:
            scenario.key(user_id=user, duration="1h")
            scenario.key(user_id=user, metadata={"auto_registered": True})
            scenario.key(user_id=user, team_id=scenario.team())
            blocked: Final = scenario.key(user_id=user)
            gateway.post("/key/block", {"key": blocked})
        keys_before: Final = _user_key_hashes(user)
        token: Final = _token(private_key, user)
        with owned_proxy(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": jwks_url},
            config=_config(tmp_path, "sub", behavior="link_only", map_existing_key=False),
        ) as candidate:
            first: Final = _chat(candidate, model, token)
            repeated: Final = _chat(candidate, model, token)
            assert first.status_code == 403, first.text
            assert repeated.status_code == 403, repeated.text
            assert _user_key_hashes(user) == keys_before
            assert read_rows('SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_value=%s', (user,)) == []
            available: Final = scenario.key(user_id=user)
            accepted: Final = _chat(candidate, model, token)
        assert accepted.status_code == 200, accepted.text
        assert _mapped_token("sub", user) == _hash(available)
        assert _user_key_hashes(user) == keys_before | {_hash(available)}
        assert _billed_key(accepted) == _hash(available)


@pytest.mark.parametrize("claim_field", ["client_id", "email"])
def test_link_only_rejects_shared_or_missing_claims_without_linking_a_personal_key(
    gateway: Gateway, tmp_path: Path, claim_field: str
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=user)
        shared_claim: Final = f"integration-client-{uuid.uuid4().hex}"
        with owned_proxy(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": jwks_url},
            config=_config(tmp_path, claim_field, behavior="link_only", map_existing_key=False),
        ) as candidate:
            response: Final = _chat(candidate, model, _token(private_key, user, client_id=shared_claim))
        assert response.status_code == 403, response.text
        assert _user_key_hashes(user) == frozenset({_hash(key)})
        assert read_rows('SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE token=%s', (_hash(key),)) == []
        assert read_rows('SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_value=%s', (shared_claim,)) == []


@pytest.mark.parametrize("restriction", ["models", "allowed_routes"])
def test_link_only_enforces_key_permissions_on_the_first_and_cached_requests(
    gateway: Gateway, tmp_path: Path, restriction: str
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        denied: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(
            user_id=user,
            **({"models": [allowed]} if restriction == "models" else {"allowed_routes": ["/v1/models"]}),
        )
        token: Final = _token(private_key, user)
        with owned_proxy(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": jwks_url},
            config=_config(tmp_path, "sub", behavior="link_only", map_existing_key=False),
        ) as candidate:
            first: Final = _chat(candidate, denied, token)
            repeated: Final = _chat(candidate, denied, token)
            control: Final = (
                _chat(candidate, allowed, token)
                if restriction == "models"
                else candidate.request("GET", "/v1/models", key=token)
            )
        assert first.status_code == 403, first.text
        assert repeated.status_code == 403, repeated.text
        assert control.status_code == 200, control.text
        assert _mapped_token("sub", user) == _hash(key)
        assert _user_key_hashes(user) == frozenset({_hash(key)})


def test_link_only_does_not_replace_an_existing_mapping_to_a_blocked_key(gateway: Gateway, tmp_path: Path) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        blocked: Final = scenario.key(user_id=user)
        available: Final = scenario.key(user_id=user)
        gateway.post("/jwt/key/mapping/new", {"jwt_claim_name": "sub", "jwt_claim_value": user, "key": blocked})
        gateway.post("/key/block", {"key": blocked})
        token: Final = _token(private_key, user)
        with owned_proxy(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": jwks_url},
            config=_config(tmp_path, "sub", behavior="link_only", map_existing_key=False),
        ) as candidate:
            first: Final = _chat(candidate, model, token)
            repeated: Final = _chat(candidate, model, token)
        assert first.status_code == 401, first.text
        assert repeated.status_code == 401, repeated.text
        assert _mapped_token("sub", user) == _hash(blocked)
        assert _user_key_hashes(user) == frozenset({_hash(blocked), _hash(available)})


def test_link_only_exempts_proxy_admins_but_rechecks_the_role_after_a_cached_exemption(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        admin: Final = _token(private_key, user, scope="litellm_proxy_admin")
        ordinary: Final = _token(private_key, user)
        with owned_proxy(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": jwks_url},
            config=_config(tmp_path, "sub", behavior="link_only", map_existing_key=False),
        ) as candidate:
            first: Final = candidate.request("GET", "/v1/models", key=admin)
            repeated: Final = candidate.request("GET", "/v1/models", key=admin)
            denied: Final = candidate.request("GET", "/v1/models", key=ordinary)
            forbidden_route: Final = _chat(candidate, model, admin)
        assert first.status_code == 200, first.text
        assert repeated.status_code == 200, repeated.text
        assert denied.status_code == 403, denied.text
        assert forbidden_route.status_code == 401, forbidden_route.text
        assert _user_key_hashes(user) == frozenset()
        assert read_rows('SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_value=%s', (user,)) == []


@pytest.mark.parametrize("user_id_upsert", [False, True])
def test_link_only_keeps_user_provisioning_independent_of_key_creation(
    gateway: Gateway, tmp_path: Path, user_id_upsert: bool
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = f"integration-link-only-{uuid.uuid4().hex}"
        try:
            with owned_proxy(
                gateway,
                tmp_path,
                {"JWT_PUBLIC_KEY_URL": jwks_url},
                config=_config(
                    tmp_path, "sub", behavior="link_only", map_existing_key=False, user_id_upsert=user_id_upsert
                ),
            ) as candidate:
                response: Final = _chat(candidate, model, _token(private_key, subject))
            assert response.status_code == (403 if user_id_upsert else 401), response.text
            users: Final = read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id=%s', (subject,))
            assert users == ([{"user_id": subject}] if user_id_upsert else [])
            assert _user_key_hashes(subject) == frozenset()
            assert read_rows('SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_value=%s', (subject,)) == []
        finally:
            if read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id=%s', (subject,)):
                scenario.delete_user(subject)


@pytest.mark.parametrize("issuer_behavior", ["auto_register", "link_only"])
def test_issuer_specific_registration_mode_overrides_the_global_mode(
    gateway: Gateway, tmp_path: Path, issuer_behavior: Literal["auto_register", "link_only"]
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        issuer: Final = f"https://integration.example.com/{uuid.uuid4().hex}"
        token: Final = _token(private_key, user, iss=issuer, aud="integration")
        path: Final = _config(
            tmp_path,
            "sub",
            behavior="auto_register" if issuer_behavior == "link_only" else "link_only",
            map_existing_key=False,
            issuers=(
                {
                    "issuer": issuer,
                    "jwks_url": jwks_url,
                    "audience": "integration",
                    "user_id_jwt_field": "sub",
                    "virtual_key_claim_field": "sub",
                    "unregistered_jwt_client_behavior": issuer_behavior,
                },
            ),
        )
        with owned_proxy(gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks_url}, config=path) as candidate:
            response: Final = _chat(candidate, model, token)
        assert response.status_code == (403 if issuer_behavior == "link_only" else 200), response.text
        if issuer_behavior == "link_only":
            assert _user_key_hashes(user) == frozenset()
            assert read_rows('SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_value=%s', (user,)) == []
        else:
            assert len(_user_key_hashes(user)) == 1
            assert _mapped_token("sub", user) in _user_key_hashes(user)
            assert _billed_key(response) == _mapped_token("sub", user)


def test_link_only_links_after_a_cached_rejection_from_the_previous_mode(gateway: Gateway, tmp_path: Path) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=user)
        token: Final = _token(private_key, user)
        with owned_proxy(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": jwks_url},
            config=_config(tmp_path, "sub", behavior="reject", map_existing_key=False),
        ) as rejecting:
            denied: Final = _chat(rejecting, model, token)
            assert denied.status_code == 403, denied.text
            with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
                eventually(
                    lambda: cache.get(f"jwt_key_mapping:sub:{user}"),
                    lambda value: value is not None and b"__NO_MAPPING__" in value,
                )
        with owned_proxy(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": jwks_url},
            config=_config(tmp_path, "sub", behavior="link_only", map_existing_key=False),
        ) as linking:
            accepted: Final = _chat(linking, model, token)
        assert accepted.status_code == 200, accepted.text
        assert _mapped_token("sub", user) == _hash(key)
        assert _user_key_hashes(user) == frozenset({_hash(key)})
        assert _billed_key(accepted) == _hash(key)
