import base64
import hashlib
import json
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs

import httpx
import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, generate_private_key
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.compatibility._a2a_peers import Peer, legacy_send, peer, sse_frames
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.timeout(900)

_KID: Final = "a2a-obo-signer"
_EXCHANGE_GRANT: Final = "urn:ietf:params:oauth:grant-type:token-exchange"
_JWT_BEARER_GRANT: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_CLIENT_SECRET: Final = "obo-client-secret-" + uuid.uuid4().hex
_CLAIMS: Final = '{"access_token":{"acrs":{"essential":true,"value":"c1"}}}'
_PINNED: Final = {"a2a_protocol_version": "0.3"}
_FORM: Final = TypeAdapter(dict[str, list[str]])


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    idp: Wire
    signer: RSAPrivateKey

    def caller(self) -> str:
        now: Final = int(time.time())
        return jwt.encode(
            {"sub": "obo-user-" + uuid.uuid4().hex, "iat": now, "exp": now + 600},
            self.signer,
            algorithm="RS256",
            headers={"kid": _KID},
        )

    def exchanges(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {name: values[0] for name, values in _FORM.validate_python(parse_qs(item.body.decode())).items()}
            for item in self.idp.drain()
        )


def _exchanged(subject: str, target: str) -> str:
    return "exchanged-" + hashlib.sha256(f"{subject}|{target}".encode()).hexdigest()[:24]


def _idp(request: Request) -> Reply:
    form: Final = {name: values[0] for name, values in parse_qs(request.body.decode()).items()}
    if form.get("client_id") == "claims-client":
        refusal: Final = {"error": "invalid_grant", "error_description": "AADSTS50076 internal", "claims": _CLAIMS}
        return Reply(status=400, body=json.dumps(refusal).encode())
    if form.get("client_id") == "broken-client":
        return Reply(status=401, body=json.dumps({"error": "invalid_client"}).encode())
    subject: Final = form.get("subject_token") or form.get("assertion", "")
    target: Final = form.get("audience") or form.get("scope", "")
    token: Final = {"access_token": _exchanged(subject, target), "token_type": "Bearer", "expires_in": 3600}
    return Reply(body=json.dumps(token).encode())


def _config(directory: Path) -> Path:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    general: Final = {
        **object_value(stock["general_settings"]),
        "enable_jwt_auth": True,
        "litellm_jwtauth": {"user_id_jwt_field": "sub", "user_id_upsert": True, "virtual_key_claim_field": "sub"},
    }
    path: Final = directory / "a2a_obo.yaml"
    path.write_text(yaml.safe_dump({**stock, "general_settings": general}))
    return path


def _base64url(value: int) -> str:
    return base64.urlsafe_b64encode(value.to_bytes((value.bit_length() + 7) // 8, "big")).rstrip(b"=").decode()


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    signer: Final = generate_private_key(public_exponent=65537, key_size=2048)
    numbers: Final = signer.public_key().public_numbers()
    public: Final = {"kty": "RSA", "alg": "RS256", "use": "sig", "n": _base64url(numbers.n), "e": _base64url(numbers.e)}
    jwks: Final = json.dumps({"keys": [{**public, "kid": _KID}]}).encode()
    directory: Final = tmp_path_factory.mktemp("a2a-obo")
    with (
        wire_server(lambda request: Reply(body=jwks)) as keys,
        wire_server(_idp) as idp,
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, {"JWT_PUBLIC_KEY_URL": keys.url}, config=_config(directory)) as proxy,
    ):
        yield _Rig(proxy, idp, signer)


def _block(rig: _Rig, client_id: str = "a2a-obo-client", **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "token_exchange_endpoint": rig.idp.url + "/token",
        "client_id": client_id,
        "client_secret": _CLIENT_SECRET,
        **fields,
    }


def _agent(
    rig: _Rig,
    scenario: Scenario,
    agent: Peer,
    litellm_params: Mapping[str, JsonValue],
    **fields: JsonValue,
) -> str:
    created: Final = rig.proxy.request(
        "POST",
        "/v1/agents",
        {
            "agent_name": agent.marker,
            "agent_card_params": {
                "protocolVersion": "0.3",
                "name": agent.marker,
                "description": "Synthetic OBO agent",
                "version": "1.0.0",
                "url": agent.url,
                "capabilities": {"streaming": True},
                "defaultInputModes": ["text"],
                "defaultOutputModes": ["text"],
                "skills": [],
            },
            "litellm_params": {**_PINNED, **litellm_params},
            **fields,
        },
    )
    assert created.status_code == 200, created.text
    identity: Final = str(created.json()["agent_id"])
    scenario.cleanups.callback(rig.proxy.request, "DELETE", f"/v1/agents/{identity}")
    return identity


def _send(
    rig: _Rig, identity: str, marker: str, headers: Mapping[str, str], method: str = "message/send"
) -> httpx.Response:
    return rig.proxy.client.post(f"/a2a/{identity}", json=legacy_send(marker, method=method), headers=dict(headers))


def _agent_authorizations(agent: Peer) -> tuple[str | None, ...]:
    assert agent.wire is not None
    return tuple(item.headers.get("authorization") for item in agent.wire.drain() if item.method == "POST")


def _assert_answered(response: httpx.Response, marker: str) -> None:
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["id"] == marker and "error" not in body, response.text


def _assert_refused(response: httpx.Response, status: int, *secrets: str) -> None:
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == -32000, response.text
    exposed: Final = response.text + json.dumps(dict(response.headers))
    assert not [secret for secret in (_CLIENT_SECRET, *secrets) if secret in exposed], response.text


def test_a_jwt_caller_reaches_the_agent_with_only_its_exchanged_token(rig: _Rig) -> None:
    with peer("obo", "langgraph") as agent, peer("obo2", "langgraph") as other, rig.proxy.scenario() as scenario:
        identity: Final = _agent(
            rig,
            scenario,
            agent,
            {"token_exchange": _block(rig, audience="api://agent-a")},
            static_headers={"Authorization": "Bearer static-admin-token"},
            extra_headers=["authorization"],
        )
        second: Final = _agent(rig, scenario, other, {"token_exchange": _block(rig, audience="api://agent-b")})
        caller: Final = rig.caller()
        rig.exchanges()
        sent: Final = _send(
            rig,
            identity,
            "obo-send",
            {"Authorization": f"Bearer {caller}", f"x-a2a-{identity}-authorization": "Bearer convention-token"},
        )
        _assert_answered(sent, "obo-send")
        first: Final = rig.exchanges()
        assert len(first) == 1, first
        assert first[0]["grant_type"] == _EXCHANGE_GRANT and first[0]["subject_token"] == caller, first
        assert first[0]["audience"] == "api://agent-a" and first[0]["client_id"] == "a2a-obo-client", first
        assert first[0]["client_secret"] == _CLIENT_SECRET, first
        expected: Final = f"Bearer {_exchanged(caller, 'api://agent-a')}"
        assert _agent_authorizations(agent) == (expected,)

        streamed: Final = _send(rig, identity, "obo-stream", {"Authorization": f"Bearer {caller}"}, "message/stream")
        assert streamed.status_code == 200, streamed.text
        assert sse_frames(streamed.text) and not [f for f in sse_frames(streamed.text) if "error" in f], streamed.text
        assert _agent_authorizations(agent) == (expected,)
        assert rig.exchanges() == ()

        _assert_answered(
            _send(rig, second, "obo-other-agent", {"Authorization": f"Bearer {caller}"}), "obo-other-agent"
        )
        assert [form["audience"] for form in rig.exchanges()] == ["api://agent-b"]
        assert _agent_authorizations(other) == (f"Bearer {_exchanged(caller, 'api://agent-b')}",)

        another: Final = rig.caller()
        _assert_answered(_send(rig, identity, "obo-another", {"Authorization": f"Bearer {another}"}), "obo-another")
        assert [form["subject_token"] for form in rig.exchanges()] == [another]
        assert _agent_authorizations(agent) == (f"Bearer {_exchanged(another, 'api://agent-a')}",)


def test_a_jwt_mapped_to_a_virtual_key_is_exchanged_with_the_entra_obo_grant(rig: _Rig) -> None:
    with peer("oboentra", "langgraph") as agent, rig.proxy.scenario() as scenario:
        scope: Final = "api://agent-entra/.default"
        identity: Final = _agent(
            rig, scenario, agent, {"token_exchange": _block(rig, profile="entra_obo", scopes=[scope])}
        )
        now: Final = int(time.time())
        user: Final = "obo-mapped-" + uuid.uuid4().hex
        caller: Final = jwt.encode(
            {"sub": user, "iat": now, "exp": now + 600}, rig.signer, algorithm="RS256", headers={"kid": _KID}
        )
        key: Final = scenario.key()
        mapping: Final = rig.proxy.post(
            "/jwt/key/mapping/new", {"jwt_claim_name": "sub", "jwt_claim_value": user, "key": key}
        )
        scenario.cleanups.callback(rig.proxy.post, "/jwt/key/mapping/delete", {"id": mapping["id"]})
        rig.exchanges()
        _assert_answered(_send(rig, identity, "obo-entra", {"Authorization": f"Bearer {caller}"}), "obo-entra")
        forms: Final = rig.exchanges()
        assert len(forms) == 1, forms
        assert forms[0]["grant_type"] == _JWT_BEARER_GRANT and forms[0]["assertion"] == caller, forms
        assert forms[0]["requested_token_use"] == "on_behalf_of" and forms[0]["scope"] == scope, forms
        assert _agent_authorizations(agent) == (f"Bearer {_exchanged(caller, scope)}",)


def test_a_virtual_key_or_a_missing_bearer_is_refused_before_the_idp_and_the_agent(rig: _Rig) -> None:
    with peer("obokey", "langgraph") as agent, rig.proxy.scenario() as scenario:
        identity: Final = _agent(rig, scenario, agent, {"token_exchange": _block(rig)})
        key: Final = scenario.key(max_parallel_requests=1)
        rig.exchanges()
        for headers in ({"Authorization": f"Bearer {key}"}, {"x-litellm-api-key": key}, {"x-litellm-api-key": key}):
            refused = _send(rig, identity, "obo-key", headers)
            _assert_refused(refused, 401, key)
            assert refused.headers["www-authenticate"] == 'Bearer error="invalid_request"', refused.headers
        subject: Final = "opaque-subject-" + uuid.uuid4().hex
        _assert_answered(
            _send(rig, identity, "obo-key-ok", {"x-litellm-api-key": key, "Authorization": f"Bearer {subject}"}),
            "obo-key-ok",
        )
        caller: Final = rig.caller()
        _assert_answered(_send(rig, identity, "obo-after", {"Authorization": f"Bearer {caller}"}), "obo-after")
        assert [form["subject_token"] for form in rig.exchanges()] == [subject, caller]
        assert _agent_authorizations(agent) == (
            f"Bearer {_exchanged(subject, '')}",
            f"Bearer {_exchanged(caller, '')}",
        )


def test_an_idp_refusal_becomes_a_json_rpc_error_and_never_reaches_the_agent(rig: _Rig) -> None:
    with wire_server(lambda request: Reply()) as closed:
        unreachable: Final = closed.url + "/token"
    with peer("oboclaims", "langgraph") as agent, rig.proxy.scenario() as scenario:
        claims_agent: Final = _agent(rig, scenario, agent, {"token_exchange": _block(rig, "claims-client")})
        broken_agent: Final = _agent(
            rig, scenario, agent, {"token_exchange": _block(rig, "broken-client")}, agent_name=agent.marker + "-broken"
        )
        down_agent: Final = _agent(
            rig,
            scenario,
            agent,
            {"token_exchange": {**_block(rig), "token_exchange_endpoint": unreachable}},
            agent_name=agent.marker + "-down",
        )
        good_agent: Final = _agent(
            rig, scenario, agent, {"token_exchange": _block(rig)}, agent_name=agent.marker + "-ok"
        )
        caller: Final = rig.caller()
        headers: Final = {"Authorization": f"Bearer {caller}"}
        stepped_up: Final = _send(rig, claims_agent, "obo-claims", headers)
        _assert_refused(stepped_up, 401, caller, "AADSTS50076")
        challenge: Final = stepped_up.headers["www-authenticate"]
        assert 'error="insufficient_claims"' in challenge, challenge
        assert f'claims="{base64.b64encode(_CLAIMS.encode()).decode()}"' in challenge, challenge
        _assert_refused(_send(rig, broken_agent, "obo-broken", headers), 500, caller)
        _assert_refused(_send(rig, down_agent, "obo-down", headers), 503, caller)
        _assert_answered(_send(rig, good_agent, "obo-good", headers), "obo-good")
        assert [form["client_id"] for form in rig.exchanges()] == ["claims-client", "broken-client", "a2a-obo-client"]
        assert _agent_authorizations(agent) == (f"Bearer {_exchanged(caller, '')}",)


def test_the_a2a_model_route_refuses_an_obo_agent_without_calling_it(rig: _Rig) -> None:
    with peer("obomodel", "langgraph") as agent, rig.proxy.scenario() as scenario:
        _agent(rig, scenario, agent, {"token_exchange": _block(rig)})
        bridge: Final = _agent(
            rig,
            scenario,
            agent,
            {"token_exchange": _block(rig), "custom_llm_provider": "databricks_agent", "model": "databricks/agent"},
            agent_name=agent.marker + "-bridge",
        )
        plain: Final = _agent(rig, scenario, agent, {}, agent_name=agent.marker + "-plain")
        caller: Final = rig.caller()
        rig.exchanges()
        for name in (agent.marker, agent.marker + "-bridge"):
            refused = rig.proxy.client.post(
                "/v1/chat/completions",
                json={"model": f"a2a/{name}", "messages": [{"role": "user", "content": "synthetic ping"}]},
                headers={"Authorization": f"Bearer {caller}"},
            )
            assert refused.status_code == 400, refused.text
            assert "/a2a/{agent_id}" in refused.text, refused.text
        bridged: Final = _send(rig, bridge, "obo-bridge", {"Authorization": f"Bearer {caller}"})
        _assert_refused(bridged, 400, caller)
        assert "reached by URL" in bridged.text, bridged.text
        _assert_answered(_send(rig, plain, "obo-plain", {"Authorization": f"Bearer {caller}"}), "obo-plain")
        assert rig.exchanges() == ()
        assert _agent_authorizations(agent) == (None,)


def test_an_agent_without_token_exchange_keeps_its_static_authorization(rig: _Rig) -> None:
    with peer("obostatic", "langgraph") as agent, rig.proxy.scenario() as scenario:
        identity: Final = _agent(
            rig, scenario, agent, {}, static_headers={"Authorization": "Bearer static-admin-token"}
        )
        rig.exchanges()
        caller: Final = rig.caller()
        _assert_answered(_send(rig, identity, "obo-static", {"Authorization": f"Bearer {caller}"}), "obo-static")
        assert _agent_authorizations(agent) == ("Bearer static-admin-token",)
        assert rig.exchanges() == ()


def _task_call(rig: _Rig, identity: str, method: str, headers: Mapping[str, str]) -> httpx.Response:
    return rig.proxy.client.post(
        f"/a2a/{identity}",
        json={"jsonrpc": "2.0", "id": method, "method": method, "params": {"id": "obo-task"}},
        headers=dict(headers),
    )


def test_tasks_methods_reach_an_obo_agent_with_only_the_exchanged_bearer(rig: _Rig) -> None:
    with peer("obotasks", "langgraph") as agent, rig.proxy.scenario() as scenario:
        identity: Final = _agent(
            rig,
            scenario,
            agent,
            {"token_exchange": _block(rig, audience="api://agent-tasks")},
            static_headers={"Authorization": "Bearer static-admin-token"},
        )
        caller: Final = rig.caller()
        key: Final = scenario.key()
        rig.exchanges()
        for method in ("tasks/get", "tasks/cancel"):
            refused = _task_call(rig, identity, method, {"Authorization": f"Bearer {key}"})
            _assert_refused(refused, 401, key)
        assert rig.exchanges() == ()
        assert _agent_authorizations(agent) == ()
        for method in ("tasks/get", "tasks/cancel"):
            answered = _task_call(rig, identity, method, {"Authorization": f"Bearer {caller}"})
            assert answered.status_code == 200, answered.text
            result = object_value(JSON_OBJECT.validate_json(answered.content)["result"])
            assert result["id"] == agent.marker + "-task", answered.text
        assert [form["subject_token"] for form in rig.exchanges()] == [caller]
        expected: Final = f"Bearer {_exchanged(caller, 'api://agent-tasks')}"
        assert _agent_authorizations(agent) == (expected, expected)


def _assert_rejected(response: httpx.Response, status: int, error_type: str, prefix: str, *secrets: str) -> None:
    _assert_refused(response, status, *secrets)
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert string_value(error["message"]).startswith(prefix), response.text
    assert object_value(error["data"]) == {"type": error_type, "param": None, "code": str(status)}, response.text


def test_an_agent_rejecting_the_exchanged_token_answers_the_caller_with_its_status_and_re_exchanges(
    rig: _Rig,
) -> None:
    with peer("oborejected", "langgraph") as agent, rig.proxy.scenario() as scenario:
        identity: Final = _agent(rig, scenario, agent, {"token_exchange": _block(rig)})
        caller: Final = rig.caller()
        headers: Final = {"Authorization": f"Bearer {caller}"}
        exchanged: Final = _exchanged(caller, "")
        rig.exchanges()
        agent.post_status = 401
        rejected: Final = _send(rig, identity, "obo-401", headers)
        _assert_rejected(rejected, 401, "authentication_error", "unauthorized: ", caller, exchanged)
        agent.post_status = 200
        _assert_answered(_send(rig, identity, "obo-after-401", headers), "obo-after-401")
        assert [form["subject_token"] for form in rig.exchanges()] == [caller, caller]

        agent.post_status = 401
        streamed: Final = _send(rig, identity, "obo-stream-401", headers, "message/stream")
        frames: Final = sse_frames(streamed.text)
        assert len(frames) == 1, streamed.text
        stream_error: Final = frames[0]["error"]
        assert isinstance(stream_error, dict) and stream_error["code"] == -32000, streamed.text
        assert stream_error["data"] == {"type": "authentication_error", "param": None, "code": "401"}, streamed.text
        assert caller not in streamed.text and exchanged not in streamed.text and _CLIENT_SECRET not in streamed.text
        rejected_task: Final = _task_call(rig, identity, "tasks/get", headers)
        _assert_rejected(rejected_task, 401, "authentication_error", "unauthorized: ", caller, exchanged)
        assert [form["subject_token"] for form in rig.exchanges()] == [caller]

        agent.post_status = 403
        forbidden: Final = _send(rig, identity, "obo-403", headers)
        _assert_rejected(forbidden, 403, "permission_error", "forbidden: ", caller, exchanged)
        assert len(rig.exchanges()) == 1
        assert _agent_authorizations(agent) == (f"Bearer {exchanged}",) * 5


def _assert_rejected_frame(response: httpx.Response, status: int, error_type: str, *secrets: str) -> None:
    assert response.status_code == 200, response.text
    frames: Final = sse_frames(response.text)
    assert len(frames) == 1, response.text
    error: Final = object_value(frames[0]["error"])
    assert error["code"] == -32000, response.text
    assert object_value(error["data"]) == {"type": error_type, "param": None, "code": str(status)}, response.text
    assert not [secret for secret in (_CLIENT_SECRET, *secrets) if secret in response.text], response.text


def test_an_agent_rejecting_the_exchanged_token_on_resubscribe_streams_the_refusal_and_re_exchanges(
    rig: _Rig,
) -> None:
    with peer("oboresubscribe", "langgraph") as agent, rig.proxy.scenario() as scenario:
        identity: Final = _agent(rig, scenario, agent, {"token_exchange": _block(rig)})
        caller: Final = rig.caller()
        headers: Final = {"Authorization": f"Bearer {caller}"}
        exchanged: Final = _exchanged(caller, "")
        rig.exchanges()
        agent.post_status = 401
        unauthorized: Final = _task_call(rig, identity, "tasks/resubscribe", headers)
        _assert_rejected_frame(unauthorized, 401, "authentication_error", caller, exchanged)
        agent.post_status = 403
        forbidden: Final = _task_call(rig, identity, "tasks/resubscribe", headers)
        _assert_rejected_frame(forbidden, 403, "permission_error", caller, exchanged)
        assert [form["subject_token"] for form in rig.exchanges()] == [caller, caller]
        assert _agent_authorizations(agent) == (f"Bearer {exchanged}",) * 2


def test_a_refused_exchange_counts_against_the_keys_rpm_limit(rig: _Rig) -> None:
    with peer("oborpm", "langgraph") as agent, rig.proxy.scenario() as scenario:
        identity: Final = _agent(rig, scenario, agent, {"token_exchange": _block(rig, "claims-client")})
        key: Final = scenario.key(rpm_limit=2)
        headers: Final = {"x-litellm-api-key": key, "Authorization": f"Bearer garbage-{uuid.uuid4().hex}"}
        rig.exchanges()
        for marker in ("obo-rpm-1", "obo-rpm-2"):
            refused = _send(rig, identity, marker, headers)
            _assert_refused(refused, 401, key)
            assert 'error="insufficient_claims"' in refused.headers["www-authenticate"], refused.headers
        assert len(rig.exchanges()) == 2
        _assert_refused(_send(rig, identity, "obo-rpm-3", headers), 429, key)
        assert rig.exchanges() == ()
        assert _agent_authorizations(agent) == ()


def test_a_refused_exchange_releases_the_keys_parallel_slot(rig: _Rig) -> None:
    with wire_server(lambda request: Reply()) as closed:
        unreachable: Final = closed.url + "/token"
    with peer("oboslot", "langgraph") as agent, rig.proxy.scenario() as scenario:
        claims_agent: Final = _agent(rig, scenario, agent, {"token_exchange": _block(rig, "claims-client")})
        down_agent: Final = _agent(
            rig,
            scenario,
            agent,
            {"token_exchange": {**_block(rig), "token_exchange_endpoint": unreachable}},
            agent_name=agent.marker + "-down",
        )
        bridge_agent: Final = _agent(
            rig,
            scenario,
            agent,
            {"token_exchange": _block(rig), "custom_llm_provider": "databricks_agent", "model": "databricks/agent"},
            agent_name=agent.marker + "-bridge",
        )
        good_agent: Final = _agent(
            rig, scenario, agent, {"token_exchange": _block(rig)}, agent_name=agent.marker + "-ok"
        )
        key: Final = scenario.key(max_parallel_requests=1)
        subject: Final = "opaque-subject-" + uuid.uuid4().hex
        headers: Final = {"x-litellm-api-key": key, "Authorization": f"Bearer {subject}"}
        rig.exchanges()
        _assert_refused(_send(rig, claims_agent, "obo-slot-claims", headers), 401, key, subject)
        _assert_refused(_send(rig, down_agent, "obo-slot-down", headers), 503, key, subject)
        _assert_refused(_send(rig, bridge_agent, "obo-slot-bridge", headers), 400, key, subject)
        _assert_refused(_send(rig, good_agent, "obo-slot-no-subject", {"Authorization": f"Bearer {key}"}), 401, key)
        _assert_answered(_send(rig, good_agent, "obo-slot-ok", headers), "obo-slot-ok")
        assert [form["client_id"] for form in rig.exchanges()] == ["claims-client", "a2a-obo-client"]
        assert _agent_authorizations(agent) == (f"Bearer {_exchanged(subject, '')}",)
