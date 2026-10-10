from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeVar
from urllib.parse import parse_qs

import anthropic
import httpx
import jwt
import openai
import psutil
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from pydantic import JsonValue

from litellm.types.llms.anthropic import ANTHROPIC_OAUTH_TOKEN_PREFIX
from tests.integration._support.anthropic_thinking import (
    answer,
    identity,
    marker_of,
    message_body,
    message_events,
    prompt,
    stream_reply,
    streams,
    text_events,
)
from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.database import read_rows
from tests.integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from tests.integration._support.wire import Reply, Request, Wire, wire_server

T = TypeVar("T")

SOURCES: Final = ("token_file", "secret_reference", "internal_issuer", "keycloak", "environment")
CLIENTS: Final = ("chat", "messages", "messages_stream")

_CLOSE: Final = MappingProxyType({"Connection": "close"})
_PROVIDER: Final = MappingProxyType({"custom_llm_provider": "anthropic"})
_SENSITIVE: Final = (
    "authorization",
    "token",
    "key",
    "secret",
    "vertex_credentials",
    "credentials",
    "password",
    "passwd",
)
_DEFAULT_TOKEN_FILE: Final = "/var/run/secrets/integration/anthropic-identity-token"
_DEFAULT_KEYCLOAK_URL: Final = "https://keycloak.integration.invalid/realms/integration/protocol/openid-connect/token"
_KEYCLOAK_TARGET: Final = "/realms/integration/protocol/openid-connect/token"
_KEYCLOAK_ASSERTION: Final = "keycloak-scripted-assertion"
_ISSUER: Final = "https://issuer.integration.invalid"
_SUBJECT: Final = "workload-integration"
_AUDIENCE: Final = "https://api.anthropic.com"
_TTL_SECONDS: Final = 900
_IDENTITY_TOKEN_VARIABLE: Final = "INTEGRATION_WIF_IDENTITY_TOKEN"
_SIGNING_KEY_VARIABLE: Final = "INTEGRATION_WIF_SIGNING_KEY"
_KEYCLOAK_SECRET_VARIABLE: Final = "INTEGRATION_WIF_KEYCLOAK_SECRET"
_JWT_BEARER: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_MODEL: Final = "anthropic/claude-haiku-4-5"
_CREDENTIAL_QUERY: Final = (
    'SELECT credential_values, credential_info FROM "LiteLLM_CredentialsTable" WHERE credential_name = %s'
)


def _ids(rule_id: JsonValue) -> dict[str, JsonValue]:
    return {
        "anthropic_federation_rule_id": rule_id,
        "anthropic_organization_id": "org-integration",
        "anthropic_service_account_id": "svac-integration",
        "anthropic_federation_workspace_id": "wrkspc-integration",
    }


def _shape(
    source: str,
    rule_id: JsonValue,
    *,
    token_file: str = _DEFAULT_TOKEN_FILE,
    keycloak_token_url: str = _DEFAULT_KEYCLOAK_URL,
) -> dict[str, JsonValue]:
    match source:
        case "token_file":
            return {**_ids(rule_id), "anthropic_identity_token_file": token_file}
        case "secret_reference":
            return {**_ids(rule_id), "anthropic_identity_token": f"oidc/env/{_IDENTITY_TOKEN_VARIABLE}"}
        case "internal_issuer":
            return {
                **_ids(rule_id),
                "anthropic_identity_source": "internal_issuer",
                "anthropic_issuer_url": _ISSUER,
                "anthropic_issuer_subject": _SUBJECT,
                "anthropic_issuer_audience": _AUDIENCE,
                "anthropic_issuer_ttl_seconds": _TTL_SECONDS,
                "anthropic_issuer_signing_key_ref": f"os.environ/{_SIGNING_KEY_VARIABLE}",
            }
        case "keycloak":
            return {
                **_ids(rule_id),
                "anthropic_identity_source": "keycloak",
                "anthropic_keycloak_token_url": keycloak_token_url,
                "anthropic_keycloak_client_id": "litellm-integration",
                "anthropic_keycloak_client_secret_ref": f"os.environ/{_KEYCLOAK_SECRET_VARIABLE}",
                "anthropic_keycloak_auth_method": "client_secret_post",
                "anthropic_keycloak_scope": "openid",
            }
        case "environment":
            return _ids(rule_id)
        case _:
            pytest.fail(f"unknown identity source {source!r}")


def _sensitive(key: str) -> bool:
    return any(word in key.lower() for word in _SENSITIVE)


def _masked(value: JsonValue) -> JsonValue:
    if not isinstance(value, str):
        return value
    return "*****" if len(value) <= 4 else f"{value[:2]}****{value[-2:]}"


def _masked_view(values: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: _masked(value) if _sensitive(key) else value for key, value in values.items()}


def _credential_name() -> str:
    return f"federation-{uuid.uuid4().hex}"


def _create_body(name: str, values: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"credential_name": name, "credential_values": dict(values), "credential_info": dict(_PROVIDER)}


def _create(gateway: Gateway, scenario: Scenario, values: Mapping[str, JsonValue]) -> str:
    name: Final = _credential_name()
    scenario.cleanups.callback(_delete_if_present, gateway, name)
    gateway.post("/credentials", _create_body(name, values))
    return name


def _delete_if_present(gateway: Gateway, name: str) -> None:
    response: Final = gateway.request("DELETE", f"/credentials/{name}")
    assert response.status_code in (200, 404), response.text


def _patch(
    gateway: Gateway,
    name: str,
    values: Mapping[str, JsonValue],
    delete: Sequence[str] = (),
    *,
    key: str | None = None,
) -> httpx.Response:
    return gateway.request(
        "PATCH",
        f"/credentials/{name}",
        {
            "credential_name": name,
            "credential_values": dict(values),
            "credential_info": dict(_PROVIDER),
            "credential_values_to_delete": list(delete),
        },
        key=key,
    )


def _stored_values(gateway: Gateway, name: str, *, key: str | None = None) -> tuple[int, JsonValue]:
    response: Final = gateway.request("GET", f"/credentials/by_name/{name}", key=key, headers=_CLOSE)
    if response.status_code != 200:
        return response.status_code, response.text
    return 200, JSON_OBJECT.validate_json(response.content)["credential_values"]


def _listed_values(gateway: Gateway, name: str) -> JsonValue:
    response: Final = gateway.request("GET", "/credentials", headers=_CLOSE)
    assert response.status_code == 200, response.text
    listed: Final = JSON_OBJECT.validate_json(response.content)["credentials"]
    assert isinstance(listed, list), listed
    matching: Final = tuple(object_value(entry) for entry in listed if object_value(entry)["credential_name"] == name)
    return matching[0]["credential_values"] if matching else None


def _stable(read: Callable[[], T], satisfied: Callable[[T], bool], *, seconds: float = 70) -> T:
    def spread_over_workers() -> tuple[T, ...]:
        with ThreadPoolExecutor(max_workers=10) as pool:
            return tuple(pool.map(lambda _: read(), range(10)))

    return eventually(
        spread_over_workers,
        lambda samples: all(satisfied(sample) for sample in samples),
        seconds=seconds,
    )[-1]


def _converged(gateway: Gateway, name: str, expected: Mapping[str, JsonValue], *, key: str | None = None) -> None:
    masked: Final = _masked_view(expected)
    _stable(partial(_stored_values, gateway, name, key=key), lambda sample: sample == (200, masked))


def _db_row(name: str) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    rows: Final = read_rows(_CREDENTIAL_QUERY, (name,))
    assert len(rows) == 1, rows
    return object_value(rows[0]["credential_values"]), object_value(rows[0]["credential_info"])


def _assert_encrypted_at_rest(name: str, submitted: Mapping[str, JsonValue]) -> None:
    stored, info = _db_row(name)
    assert info == dict(_PROVIDER), info
    assert stored.keys() == submitted.keys(), stored
    dumped: Final = json.dumps(stored)
    for key, value in submitted.items():
        if not isinstance(value, str):
            assert stored[key] == value, (key, stored[key])
            continue
        assert stored[key] != value, key
        if _sensitive(key):
            assert value not in dumped, key


def _without(values: Mapping[str, JsonValue], *keys: str) -> dict[str, JsonValue]:
    return {key: value for key, value in values.items() if key not in keys}


def _listed_models(gateway: Gateway) -> frozenset[str]:
    response: Final = gateway.request("GET", "/model/info", headers=_CLOSE)
    assert response.status_code == 200, response.text
    entries: Final = JSON_OBJECT.validate_json(response.content)["data"]
    assert isinstance(entries, list), entries
    return frozenset(string_value(object_value(entry)["model_name"]) for entry in entries)


def _deployments_visible(gateway: Gateway, names: Sequence[str]) -> None:
    wanted: Final = frozenset(names)
    _stable(partial(_listed_models, gateway), lambda listed: wanted <= listed)


def _federated_deployment(gateway: Gateway, scenario: Scenario, credential: str, api_base: str) -> str:
    name: Final = f"integration-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": _MODEL, "api_base": api_base, "litellm_credential_name": credential},
            "model_info": {},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return name


def _chat(gateway: Gateway, model: str, marker: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt(marker)}]},
        headers=_CLOSE,
    )


def _chat_outcome(gateway: Gateway, model: str) -> tuple[int, bool]:
    marker: Final = uuid.uuid4().hex
    response: Final = _chat(gateway, model, marker)
    if response.status_code != 200:
        return response.status_code, False
    choices: Final = JSON_OBJECT.validate_json(response.content)["choices"]
    assert isinstance(choices, list), choices
    return 200, object_value(object_value(choices[0])["message"])["content"] == answer(marker)


@pytest.mark.parametrize("source", SOURCES)
def test_federation_shape_round_trips(gateway: Gateway, source: str) -> None:
    with gateway.scenario() as scenario:
        values: Final = _shape(source, f"fdrl-{uuid.uuid4().hex}")
        name: Final = _create(gateway, scenario, values)
        _converged(gateway, name, values)
        _stable(partial(_listed_values, gateway, name), lambda listed: listed == _masked_view(values))
        _assert_encrypted_at_rest(name, values)


def test_patch_sets_and_deletes_values(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        values: Final = _shape("token_file", f"fdrl-{uuid.uuid4().hex}")
        name: Final = _create(gateway, scenario, values)
        _converged(gateway, name, values)
        response: Final = _patch(
            gateway,
            name,
            {"anthropic_federation_workspace_id": "wrkspc-updated"},
            ("anthropic_service_account_id",),
        )
        assert response.status_code == 200, response.text
        expected: Final = _without(
            {**values, "anthropic_federation_workspace_id": "wrkspc-updated"}, "anthropic_service_account_id"
        )
        _converged(gateway, name, expected)
        _assert_encrypted_at_rest(name, expected)


def test_patch_with_empty_values_only_deletes(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        values: Final = _shape("token_file", f"fdrl-{uuid.uuid4().hex}")
        name: Final = _create(gateway, scenario, values)
        _converged(gateway, name, values)
        response: Final = _patch(gateway, name, {}, ("anthropic_federation_workspace_id",))
        assert response.status_code == 200, response.text
        expected: Final = _without(values, "anthropic_federation_workspace_id")
        _converged(gateway, name, expected)
        _assert_encrypted_at_rest(name, expected)


def test_patch_overlapping_set_and_delete_is_refused(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        values: Final = _shape("token_file", f"fdrl-{uuid.uuid4().hex}")
        name: Final = _create(gateway, scenario, values)
        _converged(gateway, name, values)
        before: Final = _db_row(name)
        response: Final = _patch(
            gateway,
            name,
            {"anthropic_federation_workspace_id": "wrkspc-overlap"},
            ("anthropic_federation_workspace_id",),
        )
        assert response.status_code == 400, response.text
        assert "credential_values_to_delete overlaps credential_values for key(s)" in response.text, response.text
        assert "anthropic_federation_workspace_id" in response.text, response.text
        assert _db_row(name) == before
        _converged(gateway, name, values)


_MALFORMED: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"string": "fdrl-malformed", "list": ["fdrl-malformed"], "empty": {}, "null": None}
)


@pytest.mark.parametrize("shape", tuple(_MALFORMED))
def test_malformed_credential_values_are_refused(gateway: Gateway, shape: str) -> None:
    name: Final = _credential_name()
    with gateway.scenario() as scenario:
        scenario.cleanups.callback(_delete_if_present, gateway, name)
        response: Final = gateway.request(
            "POST",
            "/credentials",
            {"credential_name": name, "credential_values": _MALFORMED[shape], "credential_info": dict(_PROVIDER)},
        )
        assert response.status_code == 422, response.text
        assert read_rows(_CREDENTIAL_QUERY, (name,)) == []
        assert _stored_values(gateway, name)[0] == 404


def test_oversized_and_non_string_ids_round_trip(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        oversized: Final = _shape("environment", "f" * 5120)
        numeric: Final = _shape("environment", 424242)
        oversized_name: Final = _create(gateway, scenario, oversized)
        numeric_name: Final = _create(gateway, scenario, numeric)
        _converged(gateway, oversized_name, oversized)
        _converged(gateway, numeric_name, numeric)
        _assert_encrypted_at_rest(oversized_name, oversized)
        _assert_encrypted_at_rest(numeric_name, numeric)


def test_duplicate_create_conflicts_and_repeated_patch_is_idempotent(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        values: Final = _shape("token_file", f"fdrl-{uuid.uuid4().hex}")
        name: Final = _create(gateway, scenario, values)
        duplicate: Final = gateway.request(
            "POST", "/credentials", _create_body(name, _shape("token_file", f"fdrl-{uuid.uuid4().hex}"))
        )
        assert duplicate.status_code == 409, duplicate.text
        assert (
            f"Credential '{name}' already exists. Update it with PATCH /credentials/{name}, or delete it first."
            in duplicate.text
        ), duplicate.text
        _converged(gateway, name, values)
        _assert_encrypted_at_rest(name, values)
        statuses: Final = tuple(
            _patch(gateway, name, {"anthropic_federation_workspace_id": "wrkspc-twice"}).status_code for _ in range(2)
        )
        assert statuses == (200, 200), statuses
        expected: Final = {**values, "anthropic_federation_workspace_id": "wrkspc-twice"}
        _converged(gateway, name, expected)
        _assert_encrypted_at_rest(name, expected)


@pytest.mark.parametrize("grant", ("routed", "plain"))
def test_non_admin_cannot_write_federation_fields(gateway: Gateway, grant: str) -> None:
    with gateway.scenario() as scenario:
        values: Final = _shape("token_file", f"fdrl-{uuid.uuid4().hex}")
        name: Final = _create(gateway, scenario, values)
        _converged(gateway, name, values)
        before: Final = _db_row(name)
        model: Final = scenario.model()
        user_id: Final = scenario.user(user_role="internal_user")
        key: Final = (
            scenario.key(user_id=user_id, models=[model], allowed_routes=["/credentials*", "/v1/chat/completions"])
            if grant == "routed"
            else scenario.key(user_id=user_id, models=[model])
        )
        intruder: Final = _credential_name()
        scenario.cleanups.callback(_delete_if_present, gateway, intruder)
        attempts: Final = (
            (
                "create",
                gateway.request(
                    "POST",
                    "/credentials",
                    _create_body(intruder, _shape("token_file", f"fdrl-{uuid.uuid4().hex}")),
                    key=key,
                ),
            ),
            ("set", _patch(gateway, name, {"anthropic_federation_rule_id": "fdrl-hijacked"}, key=key)),
            ("unset", _patch(gateway, name, {}, ("anthropic_identity_token_file",), key=key)),
            ("delete", gateway.request("DELETE", f"/credentials/{name}", key=key)),
            ("jwks", gateway.request("GET", f"/credentials/{name}/jwks", key=key)),
        )
        refused: Final = 403 if grant == "routed" else 401
        assert tuple((label, response.status_code) for label, response in attempts) == tuple(
            (label, refused) for label, _ in attempts
        ), tuple((label, response.text) for label, response in attempts)
        if grant == "routed":
            assert all("Only proxy admins" in response.text for _, response in attempts), tuple(
                response.text for _, response in attempts
            )
        listing: Final = gateway.request("GET", "/credentials", key=key, headers=_CLOSE)
        assert listing.status_code == (200 if grant == "routed" else 401), listing.text
        assert _DEFAULT_TOKEN_FILE not in listing.text, listing.text
        if grant == "routed":
            _converged(gateway, name, values, key=key)
        else:
            assert _stored_values(gateway, name, key=key)[0] == 401
        assert _db_row(name) == before
        assert read_rows(_CREDENTIAL_QUERY, (intruder,)) == []
        _converged(gateway, name, values)
        gateway.chat(model, key=key)


def test_untrusted_exchange_host_is_refused_before_any_exchange(gateway: Gateway) -> None:
    with wire_server(lambda request: Reply(status=500, body=b'{"error": "never reached"}')) as wire:
        with gateway.scenario() as scenario:
            values: Final = _shape("token_file", f"fdrl-{uuid.uuid4().hex}")
            name: Final = _create(gateway, scenario, values)
            model: Final = _federated_deployment(gateway, scenario, name, wire.url)
            _deployments_visible(gateway, (model,))
            response: Final = _chat(gateway, model, uuid.uuid4().hex)
            assert response.status_code == 401, response.text
            port: Final = httpx.URL(wire.url).port
            assert "refused to use host '" in response.text, response.text
            assert f":{port}'" in response.text, response.text
            assert "LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS" in response.text, response.text
            assert wire.drain() == ()
            assert wire.connections() == 0


def test_concurrent_credential_writes_converge_across_workers(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        names: Final = tuple(_credential_name() for _ in range(10))
        for name in names:
            scenario.cleanups.callback(_delete_if_present, gateway, name)

        def lane(name: str) -> tuple[int, int, int, int]:
            created: Final = gateway.request(
                "POST", "/credentials", _create_body(name, _shape("token_file", f"fdrl-{name}"))
            )
            updated: Final = _patch(gateway, name, {"anthropic_federation_workspace_id": f"wrkspc-{name}"})
            trimmed: Final = _patch(gateway, name, {}, ("anthropic_service_account_id",))
            read: Final = gateway.request("GET", f"/credentials/by_name/{name}", headers=_CLOSE)
            return created.status_code, updated.status_code, trimmed.status_code, read.status_code

        with ThreadPoolExecutor(max_workers=10) as pool:
            outcomes: Final = tuple(pool.map(lane, names))
        assert all(outcome[:3] == (200, 200, 200) for outcome in outcomes), outcomes
        assert all(outcome[3] in (200, 404) for outcome in outcomes), outcomes
        for name in names:
            expected: Final = _without(
                {**_shape("token_file", f"fdrl-{name}"), "anthropic_federation_workspace_id": f"wrkspc-{name}"},
                "anthropic_service_account_id",
            )
            _converged(gateway, name, expected)
            _assert_encrypted_at_rest(name, expected)


@dataclass(frozen=True, slots=True)
class _Peer:
    wire: Wire
    seen: list[Request]

    def requests(self) -> tuple[Request, ...]:
        self.seen.extend(self.wire.drain())
        return tuple(self.seen)


def _exchange_reply(request: Request) -> Reply:
    grant: Final = JSON_OBJECT.validate_json(request.body)
    minted: Final = {
        "access_token": f"{ANTHROPIC_OAUTH_TOKEN_PREFIX}01-exchanged-{string_value(grant['federation_rule_id'])}",
        "token_type": "Bearer",
        "expires_in": 3600,
    }
    return Reply(body=json.dumps(minted).encode())


def _federation_peer(request: Request) -> Reply:
    if request.target == "/v1/oauth/token":
        return _exchange_reply(request)
    if request.target == _KEYCLOAK_TARGET:
        return Reply(
            body=json.dumps({"access_token": _KEYCLOAK_ASSERTION, "token_type": "Bearer", "expires_in": 60}).encode()
        )
    marker: Final = marker_of(request)
    if streams(request):
        return stream_reply(request, message_events(marker, (text_events(0, answer(marker)),)))
    return Reply(body=message_body(marker))


def _es256_private_key_pem() -> str:
    return (
        ec.generate_private_key(ec.SECP256R1())
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )


@dataclass(frozen=True, slots=True)
class _Secrets:
    allowed_dir: Path
    token_file: Path
    identity_token: str
    environment_token: str
    keycloak_secret: str
    signing_key_pem: str

    def overrides(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS": "127.0.0.1",
                "LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS": str(self.allowed_dir),
                "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2",
                _IDENTITY_TOKEN_VARIABLE: self.identity_token,
                _SIGNING_KEY_VARIABLE: self.signing_key_pem,
                _KEYCLOAK_SECRET_VARIABLE: self.keycloak_secret,
                "ANTHROPIC_IDENTITY_TOKEN": self.environment_token,
            }
        )


_REMOVED_ENVIRONMENT: Final = (
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_API_BASE",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
)


def _secrets(directory: Path) -> _Secrets:
    allowed: Final = directory / "secrets"
    allowed.mkdir()
    token_file: Final = allowed / "anthropic-identity-token"
    token_file.write_text(f"file-token-{uuid.uuid4().hex}")
    return _Secrets(
        allowed_dir=allowed,
        token_file=token_file,
        identity_token=f"env-token-{uuid.uuid4().hex}",
        environment_token=f"ambient-token-{uuid.uuid4().hex}",
        keycloak_secret=f"keycloak-secret-{uuid.uuid4().hex}",
        signing_key_pem=_es256_private_key_pem(),
    )


@dataclass(frozen=True, slots=True)
class FederationRig:
    owned: OwnedProxy
    peer: _Peer
    secrets: _Secrets
    rule_ids: Mapping[str, str]
    credentials: Mapping[str, str]
    deployments: Mapping[str, str]


@pytest.fixture(scope="module")
def federation(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FederationRig]:
    directory: Final = tmp_path_factory.mktemp("federation").resolve()
    secrets: Final = _secrets(directory)
    with gateway_from_environment() as gateway, wire_server(_federation_peer) as wire:
        with owned_proxy_process(
            gateway, directory, secrets.overrides(), remove_environment=_REMOVED_ENVIRONMENT, workers=2
        ) as owned:
            with owned.gateway.scenario() as scenario:
                rule_ids: Final = {source: f"fdrl-{source}-{uuid.uuid4().hex}" for source in SOURCES}
                credentials: Final = {
                    source: _create(
                        owned.gateway,
                        scenario,
                        _shape(
                            source,
                            rule_ids[source],
                            token_file=str(secrets.token_file),
                            keycloak_token_url=f"{wire.url}{_KEYCLOAK_TARGET}",
                        ),
                    )
                    for source in SOURCES
                }
                deployments: Final = {
                    source: _federated_deployment(owned.gateway, scenario, credentials[source], wire.url)
                    for source in SOURCES
                }
                _deployments_visible(owned.gateway, tuple(deployments.values()))
                yield FederationRig(
                    owned=owned,
                    peer=_Peer(wire, []),
                    secrets=secrets,
                    rule_ids=MappingProxyType(rule_ids),
                    credentials=MappingProxyType(credentials),
                    deployments=MappingProxyType(deployments),
                )


def _call(rig: FederationRig, client: str, model: str, marker: str) -> None:
    base_url: Final = str(rig.owned.gateway.client.base_url)
    key: Final = rig.owned.gateway.key
    match client:
        case "chat":
            completion: Final = openai.OpenAI(
                base_url=base_url + "/v1", api_key=key, max_retries=0
            ).chat.completions.create(model=model, messages=[{"role": "user", "content": prompt(marker)}])
            assert completion.choices[0].message.content == answer(marker), completion
        case "messages":
            message: Final = anthropic.Anthropic(base_url=base_url, api_key=key, max_retries=0).messages.create(
                model=model, max_tokens=64, messages=[{"role": "user", "content": prompt(marker)}]
            )
            assert message.id == identity(marker), message
            assert [block.text for block in message.content if block.type == "text"] == [answer(marker)], message
        case "messages_stream":
            with anthropic.Anthropic(base_url=base_url, api_key=key, max_retries=0).messages.stream(
                model=model, max_tokens=64, messages=[{"role": "user", "content": prompt(marker)}]
            ) as stream:
                final: Final = stream.get_final_message()
            assert final.id == identity(marker), final
            assert [block.text for block in final.content if block.type == "text"] == [answer(marker)], final
        case _:
            pytest.fail(f"unknown client {client!r}")


def _assert_assertion(rig: FederationRig, source: str, assertion: str, upstream: Sequence[Request]) -> None:
    match source:
        case "token_file":
            assert assertion == rig.secrets.token_file.read_text()
        case "secret_reference":
            assert assertion == rig.secrets.identity_token
        case "environment":
            assert assertion == rig.secrets.environment_token
        case "keycloak":
            assert assertion == _KEYCLOAK_ASSERTION
            grants: Final = tuple(request for request in upstream if request.target == _KEYCLOAK_TARGET)
            assert grants, upstream
            for grant in grants:
                assert grant.headers.get("content-type") == "application/x-www-form-urlencoded", grant.headers
                assert "authorization" not in grant.headers, grant.headers
                assert parse_qs(grant.body.decode()) == {
                    "grant_type": ["client_credentials"],
                    "client_id": ["litellm-integration"],
                    "client_secret": [rig.secrets.keycloak_secret],
                    "scope": ["openid"],
                }, grant.body
        case "internal_issuer":
            exported: Final = rig.owned.gateway.get(f"/credentials/{rig.credentials[source]}/jwks")["keys"]
            assert isinstance(exported, list) and len(exported) == 1, exported
            jwk: Final = object_value(exported[0])
            header: Final = jwt.get_unverified_header(assertion)
            assert (header["alg"], header["kid"]) == ("ES256", jwk["kid"]), (header, jwk)
            claims: Final = jwt.decode(
                assertion,
                jwt.PyJWK(dict(jwk)).key,
                algorithms=["ES256"],
                audience=_AUDIENCE,
                issuer=_ISSUER,
                options={"verify_exp": False},
            )
            assert claims["sub"] == _SUBJECT, claims
            assert claims["exp"] - claims["iat"] == _TTL_SECONDS, claims
            assert claims["jti"], claims
        case _:
            pytest.fail(f"unknown identity source {source!r}")


@pytest.mark.timeout(240)
@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("source", SOURCES)
def test_federated_exchange(federation: FederationRig, source: str, client: str) -> None:
    marker: Final = uuid.uuid4().hex
    rule_id: Final = federation.rule_ids[source]
    _call(federation, client, federation.deployments[source], marker)
    upstream: Final = federation.peer.requests()
    sent: Final = tuple(
        request for request in upstream if request.target == "/v1/messages" and marker in request.body.decode()
    )
    assert len(sent) == 1, upstream
    assert sent[0].headers.get("authorization") == f"Bearer {ANTHROPIC_OAUTH_TOKEN_PREFIX}01-exchanged-{rule_id}", sent[
        0
    ].headers
    assert "oauth-2025-04-20" in sent[0].headers.get("anthropic-beta", ""), sent[0].headers
    assert "x-api-key" not in sent[0].headers, sent[0].headers
    exchanges: Final = tuple(
        JSON_OBJECT.validate_json(request.body) for request in upstream if request.target == "/v1/oauth/token"
    )
    mine: Final = tuple(grant for grant in exchanges if grant["federation_rule_id"] == rule_id)
    assert mine, exchanges
    for grant in mine:
        assert grant["grant_type"] == _JWT_BEARER, grant
        assert (grant["organization_id"], grant["service_account_id"], grant["workspace_id"]) == (
            "org-integration",
            "svac-integration",
            "wrkspc-integration",
        ), grant
        _assert_assertion(federation, source, string_value(grant["assertion"]), upstream)


@pytest.mark.timeout(240)
def test_token_file_outside_allowed_dirs_is_refused_before_any_exchange(
    federation: FederationRig, tmp_path: Path
) -> None:
    stray: Final = tmp_path.resolve() / "anthropic-identity-token"
    stray.write_text("stray-token")
    rule_id: Final = f"fdrl-stray-{uuid.uuid4().hex}"
    marker: Final = uuid.uuid4().hex
    owned: Final = federation.owned
    with owned.gateway.scenario() as scenario:
        name: Final = _create(owned.gateway, scenario, _shape("token_file", rule_id, token_file=str(stray)))
        model: Final = _federated_deployment(owned.gateway, scenario, name, federation.peer.wire.url)
        _deployments_visible(owned.gateway, (model,))
        response: Final = _chat(owned.gateway, model, marker)
        assert response.status_code == 401, response.text
        assert "LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS" in response.text, response.text
        upstream: Final = federation.peer.requests()
        assert not any(rule_id in request.body.decode() for request in upstream), upstream
        assert not any(marker in request.body.decode() for request in upstream), upstream
        assert owned.gateway.request("GET", "/health/liveliness").status_code == 200
        control: Final = scenario.model()
        assert _chat_outcome(owned.gateway, control)[0] == 200


def _is_worker(child: psutil.Process) -> bool:
    try:
        return "spawn_main" in " ".join(child.cmdline()) and child.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _workers(owned: OwnedProxy) -> tuple[psutil.Process, ...]:
    return tuple(child for child in psutil.Process(owned.process.pid).children() if _is_worker(child))


@pytest.mark.timeout(240)
def test_worker_kill_mid_credential_burst(gateway: Gateway, tmp_path: Path) -> None:
    directory: Final = tmp_path.resolve()
    secrets: Final = _secrets(directory)
    with wire_server(_federation_peer) as wire:
        with owned_proxy_process(
            gateway, directory, secrets.overrides(), remove_environment=_REMOVED_ENVIRONMENT, workers=2
        ) as owned:
            with owned.gateway.scenario() as scenario:
                rule_id: Final = f"fdrl-burst-{uuid.uuid4().hex}"
                credential: Final = _create(
                    owned.gateway, scenario, _shape("token_file", rule_id, token_file=str(secrets.token_file))
                )
                model: Final = _federated_deployment(owned.gateway, scenario, credential, wire.url)
                _deployments_visible(owned.gateway, (model,))
                assert _chat_outcome(owned.gateway, model) == (200, True)
                names: Final = tuple(_credential_name() for _ in range(24))
                for name in names:
                    scenario.cleanups.callback(_delete_if_present, owned.gateway, name)

                def create(name: str) -> tuple[str, int | str]:
                    try:
                        response: Final = owned.gateway.request(
                            "POST", "/credentials", _create_body(name, _shape("token_file", f"fdrl-{name}"))
                        )
                    except httpx.TransportError as error:
                        return name, type(error).__name__
                    return name, response.status_code

                victim: Final = eventually(lambda: _workers(owned), lambda workers: len(workers) == 2)[0]
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures: Final = tuple(pool.submit(create, name) for name in names)
                    victim.kill()
                    outcomes: Final = tuple(future.result() for future in futures)
                assert all(status == 200 or isinstance(status, str) for _, status in outcomes), outcomes
                landed: Final = tuple(name for name, status in outcomes if status == 200)
                for name in landed:
                    assert len(read_rows(_CREDENTIAL_QUERY, (name,))) == 1, name
                respawned: Final = eventually(
                    lambda: frozenset(worker.pid for worker in _workers(owned)),
                    lambda pids: len(pids) == 2 and victim.pid not in pids,
                    seconds=30,
                )
                assert f"Child process [{victim.pid}] died" in owned.log.read_text(), respawned
                with httpx.Client(
                    base_url=owned.gateway.client.base_url,
                    timeout=15,
                    trust_env=False,
                    limits=httpx.Limits(max_keepalive_connections=0),
                ) as fresh:
                    survivor: Final = Gateway(fresh, owned.gateway.key, owned.gateway.upstream_url)
                    for name in landed:
                        _converged(survivor, name, _shape("token_file", f"fdrl-{name}"))
                    _stable(partial(_chat_outcome, survivor, model), lambda outcome: outcome == (200, True))


_FAIL_CLOSED_HINT: Final = "Settings > Workload identity"
_REFUSAL_CLIENTS: Final = (
    "chat",
    "chat_stream",
    "chat_async",
    "messages",
    "messages_stream",
    "responses",
    "responses_stream",
)
_OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120


@dataclass(frozen=True, slots=True)
class _Misconfiguration:
    reference: str
    missing: tuple[str, ...]
    blank: bool
    expected: str

    def values(self, rig: FederationRig, tag: str) -> dict[str, JsonValue]:
        token: Final = (
            str(rig.secrets.token_file)
            if self.reference == "anthropic_identity_token_file"
            else f"oidc/env/{_IDENTITY_TOKEN_VARIABLE}"
        )
        ids: Final = {"anthropic_federation_rule_id": f"fdrl-{tag}", "anthropic_organization_id": f"org-{tag}"}
        kept: Final = {key: value for key, value in ids.items() if key not in self.missing}
        blanked: Final = {key: "" for key in self.missing} if self.blank else {}
        return {self.reference: token, **kept, **blanked}


_MISCONFIGURED: Final[Mapping[str, _Misconfiguration]] = MappingProxyType(
    {
        "token_file_without_org": _Misconfiguration(
            "anthropic_identity_token_file",
            ("anthropic_organization_id",),
            False,
            "anthropic_identity_token_file is set, but anthropic_organization_id is not set",
        ),
        "token_file_without_rule": _Misconfiguration(
            "anthropic_identity_token_file",
            ("anthropic_federation_rule_id",),
            False,
            "anthropic_identity_token_file is set, but anthropic_federation_rule_id is not set",
        ),
        "inline_token_without_org": _Misconfiguration(
            "anthropic_identity_token",
            ("anthropic_organization_id",),
            False,
            "anthropic_identity_token is set, but anthropic_organization_id is not set",
        ),
        "inline_token_without_rule": _Misconfiguration(
            "anthropic_identity_token",
            ("anthropic_federation_rule_id",),
            False,
            "anthropic_identity_token is set, but anthropic_federation_rule_id is not set",
        ),
        "token_file_without_ids": _Misconfiguration(
            "anthropic_identity_token_file",
            ("anthropic_federation_rule_id", "anthropic_organization_id"),
            False,
            "anthropic_identity_token_file is set, but anthropic_federation_rule_id and anthropic_organization_id"
            " are not set",
        ),
        "token_file_blank_org": _Misconfiguration(
            "anthropic_identity_token_file",
            ("anthropic_organization_id",),
            True,
            "anthropic_identity_token_file is set, but anthropic_organization_id is not set",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class _Misconfigured:
    tag: str
    credential: str
    model: str
    shape: _Misconfiguration


def _misconfigured(rig: FederationRig, scenario: Scenario, shape: _Misconfiguration) -> _Misconfigured:
    tag: Final = uuid.uuid4().hex
    credential: Final = _create(rig.owned.gateway, scenario, shape.values(rig, tag))
    model: Final = _federated_deployment(rig.owned.gateway, scenario, credential, rig.peer.wire.url)
    return _Misconfigured(tag=tag, credential=credential, model=model, shape=shape)


async def _async_chat(base_url: str, key: str, model: str, marker: str) -> None:
    await openai.AsyncOpenAI(base_url=base_url + "/v1", api_key=key, max_retries=0).chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt(marker)}]
    )


def _attempt(rig: FederationRig, client: str, model: str, marker: str) -> None:
    base_url: Final = str(rig.owned.gateway.client.base_url)
    key: Final = rig.owned.gateway.key
    chat: Final = openai.OpenAI(base_url=base_url + "/v1", api_key=key, max_retries=0)
    messages: Final = anthropic.Anthropic(base_url=base_url, api_key=key, max_retries=0)
    match client:
        case "chat":
            chat.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt(marker)}])
        case "chat_stream":
            for _ in chat.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt(marker)}], stream=True
            ):
                pass
        case "chat_async":
            asyncio.run(_async_chat(base_url, key, model, marker))
        case "messages":
            messages.messages.create(model=model, max_tokens=64, messages=[{"role": "user", "content": prompt(marker)}])
        case "messages_stream":
            with messages.messages.stream(
                model=model, max_tokens=64, messages=[{"role": "user", "content": prompt(marker)}]
            ) as stream:
                stream.get_final_message()
        case "responses":
            chat.responses.create(model=model, input=prompt(marker))
        case "responses_stream":
            for _ in chat.responses.create(model=model, input=prompt(marker), stream=True):
                pass
        case _:
            pytest.fail(f"unknown client {client!r}")


def _refusal(rig: FederationRig, client: str, model: str, marker: str) -> tuple[int, str]:
    try:
        _attempt(rig, client, model, marker)
    except (openai.APIStatusError, anthropic.APIStatusError) as error:
        return error.status_code, str(error)
    pytest.fail(f"{client} succeeded against the misconfigured deployment {model}")


def _assert_names_missing_id(message: str, shape: _Misconfiguration) -> None:
    assert shape.expected in message, message
    assert _FAIL_CLOSED_HINT in message, message


def _assert_fails_closed(outcome: tuple[int, str], shape: _Misconfiguration) -> None:
    status, message = outcome
    assert status == 401, message
    _assert_names_missing_id(message, shape)


def _assert_peer_untouched(rig: FederationRig, *fragments: str) -> None:
    bodies: Final = tuple(request.body.decode() for request in rig.peer.requests())
    touched: Final = tuple(body for body in bodies if any(fragment in body for fragment in fragments))
    assert touched == (), touched


@pytest.mark.timeout(240)
@pytest.mark.parametrize("client", _REFUSAL_CLIENTS)
@pytest.mark.parametrize("shape", tuple(_MISCONFIGURED))
def test_legacy_reference_without_ids_fails_closed_before_any_exchange(
    federation: FederationRig, shape: str, client: str
) -> None:
    marker: Final = uuid.uuid4().hex
    with federation.owned.gateway.scenario() as scenario:
        misconfigured: Final = _misconfigured(federation, scenario, _MISCONFIGURED[shape])
        _deployments_visible(federation.owned.gateway, (misconfigured.model,))
        _assert_fails_closed(_refusal(federation, client, misconfigured.model, marker), misconfigured.shape)
        _assert_peer_untouched(federation, misconfigured.tag, marker)


@pytest.mark.timeout(240)
def test_file_upload_fails_closed_naming_the_missing_id(federation: FederationRig) -> None:
    note: Final = uuid.uuid4().hex
    with federation.owned.gateway.scenario() as scenario:
        misconfigured: Final = _misconfigured(federation, scenario, _MISCONFIGURED["token_file_without_org"])
        _deployments_visible(federation.owned.gateway, (misconfigured.model,))
        response: Final = federation.owned.gateway.request_multipart(
            "/v1/files",
            {"purpose": "user_data", "model": misconfigured.model},
            {"file": ("notes.jsonl", json.dumps({"note": note}).encode(), "application/jsonl")},
        )
        _assert_fails_closed((response.status_code, response.text), misconfigured.shape)
        _assert_peer_untouched(federation, misconfigured.tag, note)


@pytest.mark.timeout(240)
def test_skills_listing_fails_closed_naming_the_missing_id(federation: FederationRig) -> None:
    with federation.owned.gateway.scenario() as scenario:
        misconfigured: Final = _misconfigured(federation, scenario, _MISCONFIGURED["inline_token_without_rule"])
        _deployments_visible(federation.owned.gateway, (misconfigured.model,))
        response: Final = federation.owned.gateway.request(
            "GET", "/v1/skills", params={"beta": "true", "model": misconfigured.model}, headers=_CLOSE
        )
        _assert_fails_closed((response.status_code, response.text), misconfigured.shape)
        upstream: Final = federation.peer.requests()
        listed: Final = tuple(request for request in upstream if request.target.startswith("/v1/skills"))
        assert listed == (), listed
        _assert_peer_untouched(federation, misconfigured.tag)


@pytest.mark.timeout(240)
def test_health_report_names_the_missing_id(federation: FederationRig) -> None:
    with federation.owned.gateway.scenario() as scenario:
        misconfigured: Final = _misconfigured(federation, scenario, _MISCONFIGURED["token_file_without_rule"])
        _deployments_visible(federation.owned.gateway, (misconfigured.model,))
        response: Final = federation.owned.gateway.request(
            "GET", "/health", params={"model": misconfigured.model}, headers=_CLOSE
        )
        assert response.status_code == 503, response.text
        unhealthy: Final = JSON_OBJECT.validate_json(response.content)["unhealthy_endpoints"]
        assert isinstance(unhealthy, list) and len(unhealthy) == 1, response.text
        _assert_names_missing_id(string_value(object_value(unhealthy[0])["error"]), misconfigured.shape)
        _assert_peer_untouched(federation, misconfigured.tag)


@pytest.mark.timeout(240)
def test_connection_probe_names_the_missing_id(federation: FederationRig) -> None:
    with federation.owned.gateway.scenario() as scenario:
        misconfigured: Final = _misconfigured(federation, scenario, _MISCONFIGURED["inline_token_without_org"])
        _deployments_visible(federation.owned.gateway, (misconfigured.model,))
        response: Final = federation.owned.gateway.request(
            "POST",
            "/health/test_connection",
            {
                "litellm_params": {
                    "model": _MODEL,
                    "api_base": federation.peer.wire.url,
                    "litellm_credential_name": misconfigured.credential,
                },
                "mode": "chat",
            },
            headers=_CLOSE,
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["status"] == "error", response.text
        _assert_names_missing_id(response.text, misconfigured.shape)
        _assert_peer_untouched(federation, misconfigured.tag)


@pytest.mark.timeout(240)
def test_static_key_beside_a_stray_token_reference_still_wins(federation: FederationRig) -> None:
    tag: Final = uuid.uuid4().hex
    marker: Final = uuid.uuid4().hex
    api_key: Final = f"sk-ant-api03-{tag}"
    owned: Final = federation.owned
    with owned.gateway.scenario() as scenario:
        credential: Final = _create(
            owned.gateway,
            scenario,
            {
                "api_key": api_key,
                "anthropic_identity_token_file": str(federation.secrets.token_file),
                "anthropic_federation_rule_id": f"fdrl-{tag}",
            },
        )
        model: Final = _federated_deployment(owned.gateway, scenario, credential, federation.peer.wire.url)
        _deployments_visible(owned.gateway, (model,))
        _call(federation, "chat", model, marker)
        upstream: Final = federation.peer.requests()
        sent: Final = tuple(
            request for request in upstream if request.target == "/v1/messages" and marker in request.body.decode()
        )
        assert len(sent) == 1, upstream
        assert sent[0].headers.get("x-api-key") == api_key, sent[0].headers
        assert "authorization" not in sent[0].headers, sent[0].headers
        _assert_peer_untouched(federation, tag)


@pytest.mark.timeout(240)
def test_blank_token_file_reference_is_unset_and_the_ambient_token_federates(federation: FederationRig) -> None:
    rule_id: Final = f"fdrl-blank-{uuid.uuid4().hex}"
    marker: Final = uuid.uuid4().hex
    owned: Final = federation.owned
    with owned.gateway.scenario() as scenario:
        credential: Final = _create(owned.gateway, scenario, {**_ids(rule_id), "anthropic_identity_token_file": ""})
        model: Final = _federated_deployment(owned.gateway, scenario, credential, federation.peer.wire.url)
        _deployments_visible(owned.gateway, (model,))
        _call(federation, "chat", model, marker)
        grants: Final = tuple(
            JSON_OBJECT.validate_json(request.body)
            for request in federation.peer.requests()
            if request.target == "/v1/oauth/token"
        )
        mine: Final = tuple(grant for grant in grants if grant["federation_rule_id"] == rule_id)
        assert mine, grants
        assert all(grant["assertion"] == federation.secrets.environment_token for grant in mine), mine


def test_unauthenticated_call_is_refused_before_the_credential_is_read(federation: FederationRig) -> None:
    owned: Final = federation.owned
    marker: Final = uuid.uuid4().hex
    with owned.gateway.scenario() as scenario:
        entry: Final = _misconfigured(federation, scenario, _MISCONFIGURED["token_file_without_org"])
        _deployments_visible(owned.gateway, (entry.model,))
        refused: Final = owned.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": entry.model, "messages": [{"role": "user", "content": prompt(marker)}]},
            key=f"sk-not-a-key-{marker}",
            headers=_CLOSE,
        )
        assert refused.status_code == 401, refused.text
        assert _FAIL_CLOSED_HINT not in refused.text, refused.text
        _assert_fails_closed(_refusal(federation, "chat", entry.model, uuid.uuid4().hex), entry.shape)
        _assert_peer_untouched(federation, entry.tag, marker)


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_environment_organization_id_completes_a_legacy_reference(gateway: Gateway, tmp_path: Path) -> None:
    directory: Final = tmp_path.resolve()
    secrets: Final = _secrets(directory)
    tag: Final = uuid.uuid4().hex
    organization: Final = f"org-env-{tag}"
    rule_id: Final = f"fdrl-{tag}"
    marker: Final = uuid.uuid4().hex
    with wire_server(_federation_peer) as wire:
        with owned_proxy_process(
            gateway,
            directory,
            {**secrets.overrides(), "ANTHROPIC_ORGANIZATION_ID": organization},
            remove_environment=_REMOVED_ENVIRONMENT,
            workers=2,
        ) as owned:
            with owned.gateway.scenario() as scenario:
                credential: Final = _create(
                    owned.gateway,
                    scenario,
                    {"anthropic_identity_token_file": str(secrets.token_file), "anthropic_federation_rule_id": rule_id},
                )
                model: Final = _federated_deployment(owned.gateway, scenario, credential, wire.url)
                _deployments_visible(owned.gateway, (model,))
                response: Final = _chat(owned.gateway, model, marker)
                assert response.status_code == 200, response.text
                upstream: Final = wire.drain()
                grants: Final = tuple(
                    JSON_OBJECT.validate_json(request.body)
                    for request in upstream
                    if request.target == "/v1/oauth/token"
                )
                mine: Final = tuple(grant for grant in grants if grant["federation_rule_id"] == rule_id)
                assert mine, upstream
                assert all(grant["organization_id"] == organization for grant in mine), mine
                assert all(grant["assertion"] == secrets.token_file.read_text() for grant in mine), mine


@pytest.mark.timeout(240)
def test_mixed_burst_fails_closed_without_disturbing_healthy_federation(federation: FederationRig) -> None:
    owned: Final = federation.owned
    shapes: Final = tuple(_MISCONFIGURED.values())
    healthy: Final = tuple(product(SOURCES, CLIENTS))
    with owned.gateway.scenario() as scenario:
        misconfigured: Final = tuple(
            _misconfigured(federation, scenario, shapes[index % len(shapes)]) for index in range(12)
        )
        _deployments_visible(owned.gateway, tuple(entry.model for entry in misconfigured))
        control: Final = scenario.model()
        serial_markers: Final = tuple(uuid.uuid4().hex for _ in _REFUSAL_CLIENTS)
        for client, marker in zip(_REFUSAL_CLIENTS, serial_markers, strict=True):
            _assert_fails_closed(_refusal(federation, client, misconfigured[0].model, marker), misconfigured[0].shape)
        refused_markers: Final = tuple(uuid.uuid4().hex for _ in misconfigured)
        healthy_markers: Final = tuple(uuid.uuid4().hex for _ in healthy)

        def refuse(index: int) -> tuple[int, str]:
            client: Final = _REFUSAL_CLIENTS[index % len(_REFUSAL_CLIENTS)]
            return _refusal(federation, client, misconfigured[index].model, refused_markers[index])

        def federate(index: int) -> None:
            source, client = healthy[index]
            _call(federation, client, federation.deployments[source], healthy_markers[index])

        with ThreadPoolExecutor(max_workers=16) as pool:
            refusals: Final = tuple(pool.submit(refuse, index) for index in range(len(misconfigured)))
            federations: Final = tuple(pool.submit(federate, index) for index in range(len(healthy)))
            controls: Final = tuple(pool.submit(_chat_outcome, owned.gateway, control) for _ in range(4))
            outcomes: Final = tuple(future.result() for future in refusals)
            for future in federations:
                future.result()
            assert tuple(future.result()[0] for future in controls) == (200,) * 4
        for entry, outcome in zip(misconfigured, outcomes, strict=True):
            _assert_fails_closed(outcome, entry.shape)
        sent: Final = tuple(
            request.body.decode() for request in federation.peer.requests() if request.target == "/v1/messages"
        )
        for marker in healthy_markers:
            assert sum(marker in body for body in sent) == 1, marker
        _assert_peer_untouched(federation, *serial_markers, *refused_markers, *(entry.tag for entry in misconfigured))
        assert owned.gateway.request("GET", "/health/liveliness").status_code == 200
