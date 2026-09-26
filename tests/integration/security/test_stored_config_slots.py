"""Stored-config slots: credentials the proxy holds in its env, config or database reach only their owner.

Slots: A1 (virtual key raw value), A2 (master key), B2 (deployment ``api_key`` via ``/model/new``),
B3 (``/credentials`` entry named by ``litellm_credential_name``), B4 (deployment
``aws_secret_access_key``), B4v and B4t (Vertex service-account JSON and the access token minted
for it), B5 (team ``model_config`` credential override), E1 (guardrail ``api_key`` from config),
G1 and G1b (sink credentials from env).

Every test sends one ``/v1/chat/completions`` request (success, then provider 4xx) and then:

- positive control: the double that owns the canary received it (the provider's bearer, a valid
  SigV4 signature, the guardrail's ``x-api-key``, the sink's own auth header), or, for A1 and A2,
  the proxy accepted it as the caller's or the admin's key;
- at-rest control: where the slot is stored, the column is non-empty and does not hold the
  canary (a hash for A1, ciphertext for B2 to B5), so a clean S1 is not clean because nothing
  was stored;
- sensitivity control: the marker sent in the message is reported where stored prompts belong;
- the detail routes for the ids the test created are filled into S2 and called;
- no sweep finds the canary anywhere else.

Tests whose slot is created through the API share one module proxy (fresh canaries per test);
tests whose slot lives in env or config boot their own proxy so every run holds a fresh core.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from datetime import UTC, datetime
from typing import Final
from urllib.parse import parse_qs

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.sigv4 import encoded_path, signature
from integration._support.wire import Reply, Request, wire_server
from integration.security._canary import MARKER, Canary, canary, find_canary
from integration.security._sinks import (
    CONFIG_MODEL,
    GENERIC_SINK,
    PROVIDER_4XX,
    Caller,
    Recorder,
    Rig,
    canary_rig,
    settle,
)
from integration.security._sweeps import (
    assert_marker_seen,
    assert_no_hits,
    SweepReport,
    record_route_sweep,
    sweep_all,
    sweep_sink,
)

OUTCOMES: Final = ("success", "provider_4xx")
BEDROCK_MODEL: Final = "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0"
AWS_ACCESS_KEY: Final = "AKIACANARYINTEGRATION"
GUARDRAIL_PATH: Final = "/beta/litellm_basic_guardrail_api"
GUARDRAIL_SINK: Final = "guardrail"
LANGFUSE_SINK: Final = "langfuse"
LANGFUSE_PUBLIC_KEY: Final = "pk-lf-canary-integration"
VERTEX_BACKEND: Final = "gemini-2.0-flash"
TOKEN_PATH: Final = "/_oauth/token"
VERTEX_PROJECT: Final = "canary-project"
VERTEX_LOCATION: Final = "us-central1"
VERTEX_MODEL_PATH: Final = (
    f"/v1/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/publishers/google/models/{VERTEX_BACKEND}"
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    """Shared proxy for slots created through the API, with team model_config overrides on."""

    def configure(config: dict[str, object], _: str) -> None:
        settings: Final = config["litellm_settings"]
        assert isinstance(settings, dict)
        settings["enable_model_config_credential_overrides"] = True

    with canary_rig(tmp_path_factory.mktemp("canary-stored-config"), configure=configure) as value:
        yield value


def _caller(
    scenario: Scenario,
    *,
    models: Sequence[str],
    key: str | None = None,
    team_metadata: Mapping[str, object] | None = None,
) -> Caller:
    """A team, an internal user on it and that user's key on the team, allowed ``models``."""
    team: Final = scenario.team(**({"metadata": dict(team_metadata)} if team_metadata is not None else {}))
    user: Final = scenario.user(user_role="internal_user")
    scenario.gateway.post("/team/member_add", {"team_id": team, "member": {"user_id": user, "role": "user"}})
    fields: Final = {"team_id": team, "user_id": user, "models": list(models), **({"key": key} if key else {})}
    return Caller(team, user, scenario.key(**fields))


def _model(scenario: Scenario, litellm_params: Mapping[str, object]) -> tuple[str, str]:
    """A database deployment created through ``/model/new``; returns (model name, model id)."""
    name: Final = f"canary-{uuid.uuid4().hex}"
    created: Final = scenario.gateway.post(
        "/model/new", {"model_name": name, "litellm_params": dict(litellm_params), "model_info": {}}
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return name, identity


def _credential(scenario: Scenario, values: Mapping[str, str]) -> str:
    name: Final = f"canary-credential-{uuid.uuid4().hex}"
    scenario.gateway.post(
        "/credentials", {"credential_name": name, "credential_values": dict(values), "credential_info": {}}
    )

    def delete() -> None:
        response: Final = scenario.gateway.request("DELETE", f"/credentials/{name}")
        assert response.status_code == 200, response.text

    scenario.cleanups.callback(delete)
    return name


def _chat(
    gateway: Gateway, key: str, model: str, slot: str, marker: Canary, outcome: str
) -> tuple[httpx.Response, str]:
    """One chat request; returns the response and the spend-log request id."""
    text: Final = f"slot {slot} {marker.value}" + (f" {PROVIDER_4XX}" if outcome == "provider_4xx" else "")
    response: Final = gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}]}, key=key
    )
    assert response.status_code == (200 if outcome == "success" else 400), response.text
    request_id: Final = (
        string_value(response.json()["id"]) if outcome == "success" else response.headers["x-litellm-call-id"]
    )
    return response, request_id


def _reads(gateway: Gateway, paths: Mapping[str, Mapping[str, str]]) -> tuple[httpx.Response, ...]:
    """Admin detail reads that take their id as a query parameter, which S2 does not fill."""
    responses: Final = tuple(gateway.request("GET", path, params=dict(params)) for path, params in paths.items())
    assert all(response.status_code == 200 for response in responses), [
        (response.request.url.path, response.status_code, response.text[:200]) for response in responses
    ]
    return responses


def _assert_stored_without_canary(query: str, parameters: tuple[str, ...], secret: Canary) -> None:
    """At-rest control: the stored value exists, is non-trivial, and does not hold the canary."""
    rows: Final = read_rows(query, parameters)
    assert len(rows) == 1, rows
    stored: Final = next(iter(rows[0].values()))
    assert isinstance(stored, str) and len(stored) >= 32, f"Nothing stored for slot {secret.slot}: {stored!r}"
    assert stored != secret.value and find_canary(stored, (secret,)) == (), f"Slot {secret.slot} stored in plaintext"


def _finish(
    rig: Rig,
    gateway: Gateway,
    request: pytest.FixtureRequest,
    *,
    secrets: Sequence[Canary],
    marker: Canary,
    response: httpx.Response,
    request_id: str,
    caller: Caller,
    ids: Mapping[str, str],
    detail_routes: Sequence[str],
    reads: Sequence[httpx.Response] = (),
    extra_sinks: Mapping[str, Callable[[], Sequence[Request]]] | None = None,
    extra_callers: Mapping[str, str] | None = None,
    own_headers: Mapping[str, tuple[str, str]] | None = None,
    since: datetime,
    context: str,
) -> SweepReport:
    settle(rig, request_id, marker)
    sinks: Final = {
        **{name: sink.requests() for name, sink in rig.sinks.items()},
        **{name: read() for name, read in (extra_sinks or {}).items()},
    }
    report: Final = sweep_all(
        gateway,
        (marker, *secrets),
        responses=(response, *reads),
        sinks=sinks,
        ids={"request_id": request_id, "team_id": caller.team_id, "user_id": caller.user_id, **ids},
        callers={"admin": gateway.key, "internal_user": caller.key, **(extra_callers or {})},
        own_headers={**rig.own_headers, **(own_headers or {})},
        since=since,
    )
    record_route_sweep(report.routes, request.node.nodeid)
    unswept: Final = tuple(route for route in detail_routes if f"admin {route}" not in report.routes.called)
    assert not unswept, f"S2 never called the scenario's detail routes: {unswept}"
    unfound: Final = tuple(
        (route, status) for route in detail_routes if (status := gateway.request("GET", route).status_code) != 200
    )
    assert not unfound, f"The scenario's detail routes did not resolve its ids: {unfound}"
    assert_marker_seen(
        report,
        {
            "S1": "LiteLLM_SpendLogs.proxy_server_request",
            "S2": f"GET /spend/logs/ui/{request_id} as admin -> 200",
            "S4": f"{GENERIC_SINK}[",
        },
    )
    assert_marker_seen(report, {"S2": f"GET /spend/logs?request_id={request_id} as admin -> 200"})
    assert_no_hits(report.credential_hits(), context)
    return report


def _bearer(rig: Rig, marker: Canary, secret: Canary) -> None:
    """Positive control: the provider double received the scenario's request with the slot's bearer."""
    delivered: Final = rig.provider.carrying(marker.value)
    assert [entry.headers.get("authorization") for entry in delivered] == [f"Bearer {secret.value}"], (
        f"Positive control: the provider double never received the {secret.slot} canary"
    )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as three callers at most
@pytest.mark.parametrize("outcome", OUTCOMES)
def test_virtual_key_raw_value_authenticates_and_is_stored_only_as_a_hash(
    rig: Rig, outcome: str, request: pytest.FixtureRequest
) -> None:
    started: Final = datetime.now(UTC)
    a1: Final = canary("A1")
    marker: Final = canary(MARKER)
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(scenario, models=[CONFIG_MODEL], key=a1.value)
        assert caller.key == a1.value
        digest: Final = hashlib.sha256(a1.value.encode()).hexdigest()
        _assert_stored_without_canary('SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,), a1)
        response, request_id = _chat(rig.proxy, a1.value, CONFIG_MODEL, "A1", marker, outcome)
        assert len(rig.provider.carrying(marker.value)) == 1, "Positive control: the A1 key did not authenticate"
        spend: Final = eventually(
            lambda: read_rows('SELECT api_key FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        assert spend[0]["api_key"] == digest
        _finish(
            rig,
            rig.proxy,
            request,
            secrets=(a1,),
            marker=marker,
            response=response,
            request_id=request_id,
            caller=caller,
            ids={"model": CONFIG_MODEL},
            detail_routes=(f"/team/{caller.team_id}/members/me",),
            reads=_reads(rig.proxy, {"/key/info": {"key": digest}, "/team/info": {"team_id": caller.team_id}}),
            context=f"slot A1, {outcome}",
            since=started,
        )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as three callers at most
@pytest.mark.parametrize("outcome", OUTCOMES)
def test_master_key_from_env_authorizes_admin_calls_only(
    tmp_path: Path, outcome: str, request: pytest.FixtureRequest
) -> None:
    started: Final = datetime.now(UTC)
    a2: Final = canary("A2")
    marker: Final = canary(MARKER)
    with canary_rig(tmp_path, environment={"LITELLM_MASTER_KEY": a2.value}) as owned:
        admin: Final = Gateway(owned.proxy.client, a2.value, owned.proxy.upstream_url)
        with admin.scenario() as scenario:
            caller: Final = _caller(scenario, models=[CONFIG_MODEL])
            assert admin.request("GET", "/key/list").status_code == 200, "Positive control: A2 is not the admin key"
            response, request_id = _chat(admin, caller.key, CONFIG_MODEL, "A2", marker, outcome)
            assert len(owned.provider.carrying(marker.value)) == 1
            _finish(
                owned,
                admin,
                request,
                secrets=(a2,),
                marker=marker,
                response=response,
                request_id=request_id,
                caller=caller,
                ids={"model": CONFIG_MODEL},
                detail_routes=(f"/team/{caller.team_id}/members/me",),
                reads=_reads(admin, {"/team/info": {"team_id": caller.team_id}}),
                context=f"slot A2, {outcome}",
                since=started,
            )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as three callers at most
@pytest.mark.parametrize("outcome", OUTCOMES)
def test_model_api_key_added_through_the_api_reaches_only_the_provider(
    rig: Rig, outcome: str, request: pytest.FixtureRequest
) -> None:
    started: Final = datetime.now(UTC)
    b2: Final = canary("B2")
    marker: Final = canary(MARKER)
    with rig.proxy.scenario() as scenario:
        model, model_id = _model(
            scenario, {"model": "openai/gpt-4o-mini", "api_base": rig.provider.url + "/v1", "api_key": b2.value}
        )
        _assert_stored_without_canary(
            """SELECT litellm_params->>'api_key' FROM "LiteLLM_ProxyModelTable" WHERE model_id=%s""", (model_id,), b2
        )
        caller: Final = _caller(scenario, models=[model])
        response, request_id = _chat(rig.proxy, caller.key, model, "B2", marker, outcome)
        _bearer(rig, marker, b2)
        _finish(
            rig,
            rig.proxy,
            request,
            secrets=(b2,),
            marker=marker,
            response=response,
            request_id=request_id,
            caller=caller,
            ids={"model_id": model_id, "model": model},
            detail_routes=(f"/credentials/by_model/{model_id}",),
            reads=_reads(rig.proxy, {"/model/info": {"litellm_model_id": model_id}}),
            context=f"slot B2, {outcome}",
            since=started,
        )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as three callers at most
@pytest.mark.parametrize("outcome", OUTCOMES)
def test_named_credential_reaches_only_the_provider(rig: Rig, outcome: str, request: pytest.FixtureRequest) -> None:
    started: Final = datetime.now(UTC)
    b3: Final = canary("B3")
    marker: Final = canary(MARKER)
    with rig.proxy.scenario() as scenario:
        credential: Final = _credential(scenario, {"api_key": b3.value})
        _assert_stored_without_canary(
            """SELECT credential_values->>'api_key' FROM "LiteLLM_CredentialsTable" WHERE credential_name=%s""",
            (credential,),
            b3,
        )
        model, model_id = _model(
            scenario,
            {
                "model": "openai/gpt-4o-mini",
                "api_base": rig.provider.url + "/v1",
                "litellm_credential_name": credential,
            },
        )
        caller: Final = _caller(scenario, models=[model])
        response, request_id = _chat(rig.proxy, caller.key, model, "B3", marker, outcome)
        _bearer(rig, marker, b3)
        _finish(
            rig,
            rig.proxy,
            request,
            secrets=(b3,),
            marker=marker,
            response=response,
            request_id=request_id,
            caller=caller,
            ids={"model_id": model_id, "model": model, "credential_name": credential},
            detail_routes=(f"/credentials/by_name/{credential}", f"/credentials/by_model/{model_id}"),
            reads=_reads(rig.proxy, {"/model/info": {"litellm_model_id": model_id}}),
            context=f"slot B3, {outcome}",
            since=started,
        )


def _converse(request: Request) -> Reply:
    if PROVIDER_4XX.encode() in request.body:
        return Reply(
            status=400,
            body=json.dumps({"message": "rejected"}).encode(),
            headers={"x-amzn-errortype": "ValidationException"},
        )
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "bedrock canary control"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def _signed_with(request: Request, secret: str) -> bool:
    """Whether ``request`` carries a SigV4 signature for ``AWS_ACCESS_KEY`` made with ``secret``."""
    authorization: Final = request.headers.get("authorization", "")
    if not authorization.startswith("AWS4-HMAC-SHA256 "):
        return False
    fields: Final = dict(part.split("=", 1) for part in authorization.removeprefix("AWS4-HMAC-SHA256 ").split(", "))
    access, scope = fields["Credential"].split("/", 1)
    expected: Final = signature(
        request.method,
        encoded_path(request.target),
        request.headers,
        fields["SignedHeaders"],
        request.body,
        secret,
        scope,
    )[1]
    return access == AWS_ACCESS_KEY and hmac.compare_digest(expected, fields["Signature"])


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as three callers at most
@pytest.mark.parametrize("outcome", OUTCOMES)
def test_aws_secret_key_signs_the_provider_request_and_stays_encrypted(
    rig: Rig, outcome: str, request: pytest.FixtureRequest
) -> None:
    started: Final = datetime.now(UTC)
    b4: Final = canary("B4")
    marker: Final = canary(MARKER)
    with wire_server(_converse) as wire, rig.proxy.scenario() as scenario:
        bedrock: Final = Recorder(wire)
        model, model_id = _model(
            scenario,
            {
                "model": BEDROCK_MODEL,
                "aws_access_key_id": AWS_ACCESS_KEY,
                "aws_secret_access_key": b4.value,
                "aws_region_name": "us-east-1",
                "aws_bedrock_runtime_endpoint": wire.url,
            },
        )
        _assert_stored_without_canary(
            """SELECT litellm_params->>'aws_secret_access_key' FROM "LiteLLM_ProxyModelTable" WHERE model_id=%s""",
            (model_id,),
            b4,
        )
        caller: Final = _caller(scenario, models=[model])
        response, request_id = _chat(rig.proxy, caller.key, model, "B4", marker, outcome)
        delivered: Final = bedrock.carrying(marker.value)
        assert len(delivered) == 1 and _signed_with(delivered[0], b4.value), (
            "Positive control: the Bedrock double never received a request signed with the B4 canary"
        )
        _finish(
            rig,
            rig.proxy,
            request,
            secrets=(b4,),
            marker=marker,
            response=response,
            request_id=request_id,
            caller=caller,
            ids={"model_id": model_id, "model": model},
            detail_routes=(f"/credentials/by_model/{model_id}",),
            reads=_reads(rig.proxy, {"/model/info": {"litellm_model_id": model_id}}),
            extra_sinks={"bedrock": bedrock.requests},
            context=f"slot B4, {outcome}",
            since=started,
        )


def _service_account(token_url: str, key_id: Canary) -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": VERTEX_PROJECT,
            "private_key_id": key_id.value,
            "private_key": private_key,
            "client_email": f"canary@{VERTEX_PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{token_url}/_oauth/authorize",
            "token_uri": token_url + TOKEN_PATH,
        }
    )


def _vertex(token: Canary) -> Callable[[Request], Reply]:
    """Token endpoint and Gemini ``generateContent`` double; the token endpoint mints ``token``."""

    def respond(request: Request) -> Reply:
        if request.target == TOKEN_PATH:
            return Reply(
                body=json.dumps({"access_token": token.value, "expires_in": 3600, "token_type": "Bearer"}).encode()
            )
        assert request.target == f"{VERTEX_MODEL_PATH}:generateContent", request.target
        if PROVIDER_4XX.encode() in request.body:
            return Reply(
                status=400,
                body=json.dumps({"error": {"code": 400, "message": "rejected", "status": "INVALID_ARGUMENT"}}).encode(),
            )
        return Reply(
            body=json.dumps(
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "vertex canary control"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3, "totalTokenCount": 10},
                    "modelVersion": VERTEX_BACKEND,
                }
            ).encode()
        )

    return respond


def _assertion_key_id(request: Request) -> str:
    """The ``kid`` header of the JWT bearer assertion a token request carries."""
    assertion: Final = parse_qs(request.body.decode())["assertion"][0]
    header: Final = assertion.split(".", 1)[0]
    return string_value(json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))["kid"])


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as three callers at most
@pytest.mark.parametrize("outcome", OUTCOMES)
def test_vertex_service_account_and_its_token_reach_only_the_token_endpoint_and_provider(
    rig: Rig, outcome: str, request: pytest.FixtureRequest
) -> None:
    started: Final = datetime.now(UTC)
    b4v: Final = canary("B4v")
    b4t: Final = canary("B4t")
    marker: Final = canary(MARKER)
    with wire_server(_vertex(b4t)) as wire, rig.proxy.scenario() as scenario:
        vertex: Final = Recorder(wire)
        model, model_id = _model(
            scenario,
            {
                "model": f"vertex_ai/{VERTEX_BACKEND}",
                "api_base": wire.url + VERTEX_MODEL_PATH,
                "vertex_project": VERTEX_PROJECT,
                "vertex_location": VERTEX_LOCATION,
                "vertex_credentials": _service_account(wire.url, b4v),
            },
        )
        _assert_stored_without_canary(
            """SELECT litellm_params->>'vertex_credentials' FROM "LiteLLM_ProxyModelTable" WHERE model_id=%s""",
            (model_id,),
            b4v,
        )
        caller: Final = _caller(scenario, models=[model])
        response, request_id = _chat(rig.proxy, caller.key, model, "B4v", marker, outcome)
        minted: Final = tuple(entry for entry in vertex.requests() if entry.target == TOKEN_PATH)
        assert minted and {_assertion_key_id(entry) for entry in minted} == {b4v.value}, (
            "Positive control: the token endpoint never received an assertion signed for the B4v service account"
        )
        delivered: Final = vertex.carrying(marker.value)
        assert [entry.headers.get("authorization") for entry in delivered] == [f"Bearer {b4t.value}"], (
            "Positive control: the Vertex double never received the B4t access token"
        )
        assert_no_hits(sweep_sink("vertex token endpoint", minted, (b4t,)), f"slot B4t, {outcome}")
        _finish(
            rig,
            rig.proxy,
            request,
            secrets=(b4v, b4t),
            marker=marker,
            response=response,
            request_id=request_id,
            caller=caller,
            ids={"model_id": model_id, "model": model},
            detail_routes=(f"/credentials/by_model/{model_id}",),
            reads=_reads(rig.proxy, {"/model/info": {"litellm_model_id": model_id}}),
            extra_sinks={"vertex": lambda: tuple(entry for entry in vertex.requests() if entry.target != TOKEN_PATH)},
            own_headers={"vertex": ("authorization", "B4t")},
            context=f"slots B4v and B4t, {outcome}",
            since=started,
        )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as three callers at most
@pytest.mark.parametrize("outcome", OUTCOMES)
def test_team_model_config_credential_override_reaches_only_the_provider(
    rig: Rig, outcome: str, request: pytest.FixtureRequest
) -> None:
    started: Final = datetime.now(UTC)
    b5: Final = canary("B5")
    b1: Final = rig.canaries["B1"]
    marker: Final = canary(MARKER)
    with rig.proxy.scenario() as scenario:
        credential: Final = _credential(scenario, {"api_key": b5.value})
        _assert_stored_without_canary(
            """SELECT credential_values->>'api_key' FROM "LiteLLM_CredentialsTable" WHERE credential_name=%s""",
            (credential,),
            b5,
        )
        caller: Final = _caller(
            scenario,
            models=[CONFIG_MODEL],
            team_metadata={"model_config": {CONFIG_MODEL: {"openai": {"litellm_credentials": credential}}}},
        )
        response, request_id = _chat(rig.proxy, caller.key, CONFIG_MODEL, "B5", marker, outcome)
        _bearer(rig, marker, b5)
        _finish(
            rig,
            rig.proxy,
            request,
            secrets=(b5, b1),
            marker=marker,
            response=response,
            request_id=request_id,
            caller=caller,
            ids={"model": CONFIG_MODEL, "credential_name": credential},
            detail_routes=(f"/credentials/by_name/{credential}",),
            reads=_reads(rig.proxy, {"/team/info": {"team_id": caller.team_id}}),
            context=f"slot B5, {outcome}",
            since=started,
        )


def _guardrail(request: Request) -> Reply:
    assert request.target == GUARDRAIL_PATH, request.target
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def _guardrail_params(url: str, secret: Canary) -> dict[str, object]:
    return {
        "guardrail": "generic_guardrail_api",
        "mode": "pre_call",
        "default_on": True,
        "api_base": url,
        "api_key": secret.value,
    }


def _guardrail_delivered(guardrail: Recorder, marker: Canary, secret: Canary) -> None:
    delivered: Final = guardrail.carrying(marker.core)
    assert [entry.headers.get("x-api-key") for entry in delivered] == [secret.value], (
        "Positive control: the guardrail double never received the E1 canary"
    )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as three callers at most
@pytest.mark.parametrize("outcome", OUTCOMES)
def test_config_guardrail_api_key_reaches_only_the_guardrail(
    tmp_path: Path, outcome: str, request: pytest.FixtureRequest
) -> None:
    started: Final = datetime.now(UTC)
    e1: Final = canary("E1")
    marker: Final = canary(MARKER)
    name: Final = f"canary-guardrail-{uuid.uuid4().hex}"
    with wire_server(_guardrail) as wire:
        guardrail: Final = Recorder(wire)

        def configure(config: dict[str, object], _: str) -> None:
            config["guardrails"] = [{"guardrail_name": name, "litellm_params": _guardrail_params(wire.url, e1)}]

        with canary_rig(tmp_path, configure=configure) as owned, owned.proxy.scenario() as scenario:
            caller: Final = _caller(scenario, models=[CONFIG_MODEL])
            response, request_id = _chat(owned.proxy, caller.key, CONFIG_MODEL, "E1", marker, outcome)
            _guardrail_delivered(guardrail, marker, e1)
            listed: Final = owned.proxy.get("/v2/guardrails/list")["guardrails"]
            assert isinstance(listed, list)
            guardrail_id: Final = next(
                string_value(object_value(entry)["guardrail_id"])
                for entry in listed
                if object_value(entry)["guardrail_name"] == name
            )
            _finish(
                owned,
                owned.proxy,
                request,
                secrets=(e1,),
                marker=marker,
                response=response,
                request_id=request_id,
                caller=caller,
                ids={"model": CONFIG_MODEL, "guardrail_id": guardrail_id},
                detail_routes=(f"/guardrails/{guardrail_id}/info", f"/guardrails/{guardrail_id}"),
                reads=_reads(owned.proxy, {"/guardrails/list": {}, "/v2/guardrails/list": {}}),
                extra_sinks={GUARDRAIL_SINK: guardrail.requests},
                own_headers={GUARDRAIL_SINK: ("x-api-key", "E1")},
                context=f"slot E1 (config), {outcome}",
                since=started,
            )


def _langfuse(request: Request) -> Reply:
    if request.method == "GET" and request.target.startswith("/api/public/projects"):
        return Reply(body=json.dumps({"data": [{"id": "canary-project", "name": "canary"}]}).encode())
    return Reply(body=b"", content_type="application/x-protobuf")


def _assert_callback_secrets_gated(gateway: Gateway, internal_user: str, viewer: str) -> None:
    """The callback settings route refuses internal users and redacts sink secrets for admin viewers."""
    refused: Final = gateway.request("GET", "/get/config/callbacks", key=internal_user)
    assert refused.status_code == 401, refused.text
    shown: Final = gateway.request("GET", "/get/config/callbacks", key=viewer)
    assert shown.status_code == 200, shown.text
    secrets: Final = {
        name: value
        for entry in shown.json()["callbacks"]
        for name, value in entry["variables"].items()
        if name in ("GENERIC_LOGGER_HEADERS", "LANGFUSE_SECRET_KEY")
    }
    assert secrets == {"GENERIC_LOGGER_HEADERS": "REDACTED", "LANGFUSE_SECRET_KEY": "REDACTED"}, secrets


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as three callers at most
@pytest.mark.parametrize("outcome", OUTCOMES)
def test_sink_credentials_from_env_reach_only_their_sink(
    tmp_path: Path, outcome: str, request: pytest.FixtureRequest
) -> None:
    started: Final = datetime.now(UTC)
    g1: Final = canary("G1")
    g1b: Final = canary("G1b")
    marker: Final = canary(MARKER)

    def configure(config: dict[str, object], _: str) -> None:
        settings: Final = config["litellm_settings"]
        assert isinstance(settings, dict)
        settings.update({"success_callback": ["langfuse"], "failure_callback": ["langfuse"]})

    with wire_server(_langfuse) as wire:
        langfuse: Final = Recorder(wire)
        environment: Final = {
            "LANGFUSE_HOST": wire.url,
            "LANGFUSE_PUBLIC_KEY": LANGFUSE_PUBLIC_KEY,
            "LANGFUSE_SECRET_KEY": g1b.value,
            "LANGFUSE_FLUSH_INTERVAL": "1",
        }
        with (
            canary_rig(tmp_path, configure=configure, environment=environment, sink_token=g1) as owned,
            owned.proxy.scenario() as scenario,
        ):
            caller: Final = _caller(scenario, models=[CONFIG_MODEL])
            response, request_id = _chat(owned.proxy, caller.key, CONFIG_MODEL, "G1", marker, outcome)
            generic: Final = eventually(lambda: owned.sinks[GENERIC_SINK].carrying(marker.core), bool, seconds=30)
            assert {entry.headers.get("authorization") for entry in generic} == {f"Bearer {g1.value}"}, (
                "Positive control: the generic_api double never received the G1 canary"
            )
            basic: Final = "Basic " + base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{g1b.value}".encode()).decode()
            traced: Final = eventually(lambda: langfuse.carrying(marker.core), bool, seconds=30)
            assert {entry.headers.get("authorization") for entry in traced} == {basic}, (
                "Positive control: the Langfuse double never received the G1b canary"
            )
            viewer: Final = scenario.key(user_id=scenario.user(user_role="proxy_admin_viewer"))
            _assert_callback_secrets_gated(owned.proxy, caller.key, viewer)
            report: Final = _finish(
                owned,
                owned.proxy,
                request,
                secrets=(g1, g1b),
                marker=marker,
                response=response,
                request_id=request_id,
                caller=caller,
                ids={"model": CONFIG_MODEL},
                detail_routes=(f"/team/{caller.team_id}/members/me",),
                extra_sinks={LANGFUSE_SINK: langfuse.requests},
                own_headers={LANGFUSE_SINK: ("authorization", "G1b")},
                extra_callers={"proxy_admin_viewer": viewer},
                context=f"slots G1 and G1b, {outcome}",
                since=started,
            )
            assert {(hit.slot, hit.location) for hit in report.routes.allowed} == {
                (slot, "GET /get/config/callbacks as admin -> 200") for slot in ("G1", "G1b")
            }, report.routes.allowed
