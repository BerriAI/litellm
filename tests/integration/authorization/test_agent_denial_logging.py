import hashlib
import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    held,
    object_value,
    string_value,
)
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue

pytestmark = pytest.mark.timeout(900)

_PROBE: Final = "integration._support.a2a_denial_probe."
_SETTLE: Final = 3.0
_CONVERGE: Final = 30.0
_NO_SETTINGS: Final[Mapping[str, JsonValue]] = MappingProxyType({})
_ROWS: Final = (
    'SELECT request_id, call_type, status, model, agent_id, spend, api_key, "user", team_id, metadata '
    'FROM "LiteLLM_SpendLogs" WHERE api_key = %s'
)
_ONE_FAILURE: Final = ("async_failure", "post_call_failure")
_STREAM_METHODS: Final = frozenset({"message/stream", "SendStreamingMessage"})


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    hook: str
    call_type: str | None = None
    litellm_call_id: str | None = None
    model: str | None = None
    status: str | None = None
    key_alias: str | None = None
    key_hash: str | None = None
    user_id: str | None = None
    team_id: str | None = None
    agent_id: str | None = None
    metadata_agent_id: str | None = None
    status_code: int | None = None


class _Events:
    def __init__(self, sink: Wire) -> None:
        self.sink: Final = sink
        self.seen: tuple[_Event, ...] = ()

    def of(self, alias: str) -> tuple[_Event, ...]:
        self.seen = (*self.seen, *(_Event.model_validate_json(request.body) for request in self.sink.drain()))
        return tuple(event for event in self.seen if event.key_alias == alias)


@dataclass(frozen=True, slots=True)
class _Agent:
    agent_id: str
    name: str


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    events: _Events
    upstream: Wire
    allowed: _Agent
    denied: _Agent
    denied_legacy: _Agent
    bridge: _Agent


@dataclass(frozen=True, slots=True)
class _Caller:
    key: str
    alias: str
    user_id: str
    team_id: str | None

    @property
    def hashed(self) -> str:
        return hashlib.sha256(self.key.encode()).hexdigest()


def _compact(content: Mapping[str, JsonValue]) -> bytes:
    return json.dumps(content, separators=(",", ":")).encode()


def _refusal(agent: str, response: httpx.Response | None = None) -> bytes:
    message: Final = f"Agent '{agent}' is not allowed for your key/team. Contact proxy admin for access."
    if response is None:
        return _compact({"detail": message})
    return _compact(
        {
            "jsonrpc": "2.0",
            "id": JSON_OBJECT.validate_json(response.request.content)["id"],
            "error": {
                "code": -32000,
                "message": message,
                "data": {"type": "permission_error", "param": None, "code": "403"},
            },
        }
    )


def _not_found(request_id: JsonValue, agent: str) -> bytes:
    return _compact(
        {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": f"Agent '{agent}' not found"}}
    )


def _card(url: str, name: str, version: str) -> dict[str, JsonValue]:
    card: Final[dict[str, JsonValue]] = {
        "protocolVersion": version,
        "name": name,
        "description": "Synthetic peer",
        "version": "1.0.0",
        "url": url + "/",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }
    if version == "0.3":
        return card
    return {**card, "supportedInterfaces": [{"url": url + "/", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}]}


def _peer(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=_compact(_card("http://" + request.headers["host"], "peer", "0.3")))
    body: Final = JSON_OBJECT.validate_json(request.body)
    if body.get("method") == "message/stream":
        frame: Final = {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "kind": "status-update",
                "taskId": "t1",
                "contextId": "c1",
                "status": {"state": "completed"},
                "final": True,
            },
        }
        return Reply(content_type="text/event-stream", chunks=(f"data: {json.dumps(frame)}\n\n".encode(),))
    return Reply(
        body=_compact(
            {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "kind": "message",
                    "role": "agent",
                    "messageId": "peer-message",
                    "parts": [{"kind": "text", "text": "granted"}],
                },
            }
        )
    )


def _register(proxy: Gateway, scenario: Scenario, body: Mapping[str, JsonValue]) -> _Agent:
    created: Final = proxy.request("POST", "/v1/agents", body)
    assert created.status_code == 200, created.text
    agent: Final = _Agent(string_value(JSON_OBJECT.validate_json(created.content)["agent_id"]), str(body["agent_name"]))

    def cleanup() -> None:
        deleted: Final = proxy.request("DELETE", f"/v1/agents/{agent.agent_id}")
        assert deleted.status_code == 200, deleted.text

    scenario.cleanups.callback(cleanup)
    return agent


def _config(directory: Path, callbacks: tuple[str, ...], general: Mapping[str, JsonValue]) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "general_settings": {**object_value(base["general_settings"]), **general},
        "litellm_settings": {
            **object_value(base["litellm_settings"]),
            "callbacks": [_PROBE + "probe", *(_PROBE + name for name in callbacks)],
        },
    }
    path: Final = directory / "a2a-denials.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _owned_rig(
    directory: Path,
    callbacks: tuple[str, ...] = (),
    general: Mapping[str, JsonValue] = _NO_SETTINGS,
    environment: Mapping[str, str] = MappingProxyType({}),
) -> Iterator[_Rig]:
    marker: Final = "a2adeny" + uuid.uuid4().hex[:10]
    with (
        wire_server(_peer) as upstream,
        wire_server(lambda request: Reply()) as sink,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"A2A_DENIAL_PROBE_URL": sink.url, **environment},
            config=_config(directory, callbacks, general),
        ) as proxy,
        proxy.scenario() as scenario,
    ):

        def agent(suffix: str, version: str, **extra: JsonValue) -> _Agent:
            name: Final = f"{marker}-{suffix}"
            return _register(
                proxy, scenario, {"agent_name": name, "agent_card_params": _card(upstream.url, name, version), **extra}
            )

        yield _Rig(
            proxy,
            _Events(sink),
            upstream,
            allowed=agent("allowed", "0.3"),
            denied=agent("denied", "1.0"),
            denied_legacy=agent("legacy", "0.3"),
            bridge=agent(
                "bridge",
                "0.3",
                litellm_params={
                    "custom_llm_provider": "openai",
                    "model": "openai/gpt-5.4-mini",
                    "api_base": upstream.url + "/v1",
                    "api_key": "synthetic-bridge-key",
                },
            ),
        )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    with _owned_rig(tmp_path_factory.mktemp("a2a-denials")) as owned:
        yield owned


@pytest.fixture(scope="module", params=["raising_failure_logger,returning_teapot_logger", "raising_teapot_logger"])
def hostile(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    callbacks: Final = tuple(str(request.param).split(","))
    with _owned_rig(tmp_path_factory.mktemp("a2a-denials-hostile"), callbacks) as owned:
        yield owned


@pytest.fixture(scope="module", params=["disable_pre_routing_error_logs", "disable_error_logs"])
def flagged(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    with _owned_rig(tmp_path_factory.mktemp("a2a-denials-flagged"), general={str(request.param): True}) as owned:
        yield owned


@pytest.fixture(scope="module")
def alerting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[_Rig, Wire]]:
    with wire_server(lambda request: Reply(body=b"ok", content_type="text/plain")) as alerts:
        with _owned_rig(
            tmp_path_factory.mktemp("a2a-denials-alerting"),
            general={"alerting": ["slack"], "alert_types": ["llm_exceptions"]},
            environment={"SLACK_WEBHOOK_URL": alerts.url, "DEFAULT_FLUSH_INTERVAL_SECONDS": "1"},
        ) as owned:
            yield owned, alerts


def _caller(rig: _Rig, scenario: Scenario, grant: str = "team") -> _Caller:
    alias: Final = "a2adeny-" + uuid.uuid4().hex[:12]
    user: Final = scenario.user()
    allowed: Final[list[JsonValue]] = [rig.allowed.agent_id]
    if grant == "personal":
        return _Caller(
            scenario.key(user_id=user, key_alias=alias, object_permission={"agents": allowed}), alias, user, None
        )
    if grant == "access group":
        created: Final = rig.proxy.request(
            "POST", "/v1/access_group", {"access_group_name": alias, "access_agent_ids": allowed}
        )
        assert created.status_code == 201, created.text
        group: Final = string_value(JSON_OBJECT.validate_json(created.content)["access_group_id"])
        scenario.cleanups.callback(rig.proxy.request, "DELETE", f"/v1/access_group/{group}")
        return _Caller(scenario.key(user_id=user, key_alias=alias, access_group_ids=[group]), alias, user, None)
    team: Final = scenario.team(object_permission={"agents": allowed})
    return _Caller(scenario.key(user_id=user, team_id=team, key_alias=alias), alias, user, team)


def _params(method: str, message_id: str, version: str | None) -> dict[str, JsonValue]:
    if method.startswith("tasks/pushNotificationConfig"):
        return {"taskId": "t1", "pushNotificationConfig": {"url": "https://callback.example.com/hook"}}
    if method in {"tasks/get", "tasks/cancel", "GetTask", "CancelTask"}:
        return {"id": "t1"}
    if version == "1.0":
        return {"message": {"messageId": message_id, "role": "ROLE_USER", "parts": [{"text": "x"}]}}
    return {
        "message": {
            "kind": "message",
            "messageId": message_id,
            "role": "user",
            "parts": [{"kind": "text", "text": "x"}],
        }
    }


def _send(
    rig: _Rig,
    key: str,
    path: str,
    *,
    method: str = "SendMessage",
    version: str | None = "1.0",
    request_id: JsonValue = None,
    message_id: str | None = None,
    extra: Mapping[str, JsonValue] = _NO_SETTINGS,
) -> httpx.Response:
    headers: Final = {
        "Authorization": f"Bearer {key}",
        **({"A2A-Version": version} if version is not None else {}),
        **({"Accept": "text/event-stream"} if method in _STREAM_METHODS else {}),
    }
    body: Final = {
        "jsonrpc": "2.0",
        "id": "r-" + uuid.uuid4().hex if request_id is None else request_id,
        "method": method,
        "params": _params(method, message_id or uuid.uuid4().hex, version),
        **extra,
    }
    return rig.proxy.client.post(path, headers=headers, json=body)


def _assert_never_dispatched(rig: _Rig, scenario: Scenario, denied_message: str) -> None:
    control_key: Final = scenario.key(object_permission={"agents": [rig.allowed.agent_id]})
    control_message: Final = "control-" + uuid.uuid4().hex
    granted: Final = _send(
        rig,
        control_key,
        f"/a2a/{rig.allowed.agent_id}",
        method="message/send",
        version=None,
        message_id=control_message,
    )
    assert granted.status_code == 200, granted.text
    received: Final = tuple(request.body.decode() for request in rig.upstream.drain())
    assert any(control_message in body for body in received), received
    assert not any(denied_message in body for body in received), received


def _metadata(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    value: Final = row["metadata"]
    return JSON_OBJECT.validate_json(value) if isinstance(value, str) else object_value(value)


def _rows(rig: _Rig, caller: _Caller) -> list[dict[str, JsonValue]]:
    return rig.proxy.rows(_ROWS, (caller.hashed,))


def _hooks(events: tuple[_Event, ...]) -> tuple[str, ...]:
    return tuple(sorted(event.hook for event in events))


def _spend(rig: _Rig, caller: _Caller) -> tuple[JsonValue, ...]:
    key: Final = object_value(rig.proxy.get("/key/info", {"key": caller.key})["info"])["spend"]
    user: Final = object_value(rig.proxy.get("/user/info", {"user_id": caller.user_id})["user_info"])["spend"]
    if caller.team_id is None:
        return key, user
    team: Final = object_value(rig.proxy.get("/team/info", {"team_id": caller.team_id})["team_info"])["spend"]
    return key, user, team


def _assert_one_failure(rig: _Rig, caller: _Caller, *, status: int, agent_id: str, model: str) -> dict[str, JsonValue]:
    events: Final = held(
        lambda: rig.events.of(caller.alias),
        lambda found: _hooks(found) == _ONE_FAILURE,
        holding=_SETTLE,
        seconds=_CONVERGE,
    )
    logged: Final = next(event for event in events if event.hook == "async_failure")
    assert (logged.call_type, logged.status, logged.model) == ("asend_message", "failure", model), logged
    assert (logged.key_hash, logged.user_id, logged.team_id) == (caller.hashed, caller.user_id, caller.team_id), logged
    hooked: Final = next(event for event in events if event.hook == "post_call_failure")
    assert (hooked.status_code, hooked.agent_id, hooked.key_hash) == (status, agent_id, caller.hashed), hooked
    assert (hooked.user_id, hooked.team_id) == (caller.user_id, caller.team_id), hooked
    rows: Final = held(lambda: _rows(rig, caller), lambda found: len(found) == 1, holding=_SETTLE, seconds=_CONVERGE)
    row: Final = rows[0]
    observed: Final = (row["status"], row["call_type"], row["agent_id"], row["api_key"], row["user"], row["spend"])
    assert observed == ("failure", "asend_message", agent_id, caller.hashed, caller.user_id, 0.0), row
    assert (row["team_id"] or None) == caller.team_id, row
    assert object_value(_metadata(row)["error_information"])["error_code"] == str(status), row
    assert all(spend == 0.0 for spend in _spend(rig, caller)), _spend(rig, caller)
    return row


@pytest.mark.parametrize(
    ("route", "method", "version", "legacy"),
    [
        pytest.param("/a2a/{}", "SendMessage", "1.0", False, id="regression-send-1.0"),
        pytest.param("/a2a/{}/message/send", "SendMessage", "1.0", False, id="regression-send-alias"),
        pytest.param("/v1/a2a/{}/message/send", "SendMessage", "1.0", False, id="regression-send-v1-alias"),
        pytest.param("/a2a/{}", "SendStreamingMessage", "1.0", False, id="regression-stream-1.0"),
        pytest.param("/a2a/{}", "GetTask", "1.0", False, id="regression-get-task-1.0"),
        pytest.param("/a2a/{}", "CancelTask", "1.0", False, id="regression-cancel-task-1.0"),
        pytest.param("/a2a/{}", "message/send", None, True, id="regression-send-0.3"),
        pytest.param("/a2a/{}", "message/stream", None, True, id="regression-stream-0.3"),
        pytest.param("/a2a/{}", "tasks/get", None, True, id="regression-tasks-get-0.3"),
        pytest.param("/a2a/{}", "tasks/cancel", None, True, id="regression-tasks-cancel-0.3"),
        pytest.param("/a2a/{}", "tasks/pushNotificationConfig/set", None, True, id="regression-push-set-0.3"),
    ],
)
def test_denied_agent_keeps_its_403_logs_one_zero_spend_failure_for_the_caller_and_never_dispatches(
    rig: _Rig, route: str, method: str, version: str | None, legacy: bool
) -> None:
    target: Final = rig.denied_legacy if legacy else rig.denied
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(rig, scenario)
        message: Final = "denied-" + uuid.uuid4().hex
        response: Final = _send(
            rig, caller.key, route.format(target.agent_id), method=method, version=version, message_id=message
        )
        assert (response.status_code, response.content) == (403, _refusal(target.agent_id, response))
        assert response.headers["content-type"] == "application/json"
        _assert_never_dispatched(rig, scenario, message)
        _assert_one_failure(rig, caller, status=403, agent_id=target.agent_id, model=f"a2a_agent/{target.name}")


@pytest.mark.parametrize("grant", ["personal", "team", "access group"], ids=lambda grant: "regression-" + grant)
def test_denied_agent_failure_is_attributed_to_the_authenticated_key_for_every_grant_kind(
    rig: _Rig, grant: str
) -> None:
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(rig, scenario, grant)
        granted: Final = _send(rig, caller.key, f"/a2a/{rig.allowed.agent_id}", method="message/send", version=None)
        assert granted.status_code == 200, granted.text
        response: Final = _send(rig, caller.key, f"/a2a/{rig.denied.agent_id}")
        assert (response.status_code, response.content) == (403, _refusal(rig.denied.agent_id, response))
        events: Final = held(
            lambda: rig.events.of(caller.alias),
            lambda found: _hooks(found) == tuple(sorted((*_ONE_FAILURE, "async_success"))),
            holding=_SETTLE,
            seconds=_CONVERGE,
        )
        failure: Final = next(event for event in events if event.hook == "async_failure")
        assert (failure.key_hash, failure.user_id, failure.team_id) == (caller.hashed, caller.user_id, caller.team_id)
        rows: Final = held(
            lambda: _rows(rig, caller), lambda found: len(found) == 2, holding=_SETTLE, seconds=_CONVERGE
        )
        failed: Final = tuple(row for row in rows if row["status"] == "failure")
        assert [(row["agent_id"], row["user"], row["spend"]) for row in failed] == [
            (rig.denied.agent_id, caller.user_id, 0.0)
        ], rows


def test_denied_completion_bridge_agent_logs_one_failure_and_never_reaches_the_bridged_model(rig: _Rig) -> None:
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(rig, scenario)
        message: Final = "bridge-" + uuid.uuid4().hex
        response: Final = _send(
            rig, caller.key, f"/a2a/{rig.bridge.agent_id}", method="message/send", version=None, message_id=message
        )
        assert (response.status_code, response.content) == (403, _refusal(rig.bridge.agent_id, response))
        _assert_never_dispatched(rig, scenario, message)
        _assert_one_failure(rig, caller, status=403, agent_id=rig.bridge.agent_id, model=f"a2a_agent/{rig.bridge.name}")


@pytest.mark.parametrize(
    ("route", "method", "version"),
    [
        pytest.param("/a2a/{}", "SendMessage", "1.0", id="regression-send"),
        pytest.param("/a2a/{}/message/send", "SendMessage", "1.0", id="regression-send-alias"),
        pytest.param("/v1/a2a/{}/message/send", "SendMessage", "1.0", id="regression-send-v1-alias"),
        pytest.param("/a2a/{}", "message/stream", None, id="regression-stream-0.3"),
    ],
)
def test_unknown_agent_keeps_its_jsonrpc_404_and_logs_one_zero_spend_failure_for_the_caller(
    rig: _Rig, route: str, method: str, version: str | None
) -> None:
    unknown: Final = "no-such-" + uuid.uuid4().hex
    request_id: Final = "u-" + uuid.uuid4().hex
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(rig, scenario)
        message: Final = "unknown-" + uuid.uuid4().hex
        response: Final = _send(
            rig,
            caller.key,
            route.format(unknown),
            method=method,
            version=version,
            request_id=request_id,
            message_id=message,
        )
        assert (response.status_code, response.content) == (404, _not_found(request_id, unknown))
        assert response.headers["content-type"] == "application/json"
        _assert_never_dispatched(rig, scenario, message)
        _assert_one_failure(rig, caller, status=404, agent_id=unknown, model=f"a2a_agent/{unknown}")


@pytest.mark.parametrize("request_id", [0, 7, "abc"], ids=lambda value: f"regression-id-{value!r}")
def test_unknown_agent_echoes_the_jsonrpc_id_with_its_type_and_logs_one_failure(
    rig: _Rig, request_id: JsonValue
) -> None:
    unknown: Final = "no-such-" + uuid.uuid4().hex
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(rig, scenario)
        response: Final = _send(rig, caller.key, f"/a2a/{unknown}", request_id=request_id)
        assert (response.status_code, response.content) == (404, _not_found(request_id, unknown))
        assert type(JSON_OBJECT.validate_json(response.content)["id"]) is type(request_id)
        _assert_one_failure(rig, caller, status=404, agent_id=unknown, model=f"a2a_agent/{unknown}")


@pytest.mark.parametrize("outcome", ["denied", "unknown"], ids=lambda outcome: "regression-" + outcome)
def test_client_supplied_attribution_cannot_rewrite_a_denial_failure_row(rig: _Rig, outcome: str) -> None:
    unknown: Final = "no-such-" + uuid.uuid4().hex
    target: Final = rig.denied.agent_id if outcome == "denied" else unknown
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(rig, scenario)
        other: Final = _caller(rig, scenario)
        success_id: Final = "s-" + uuid.uuid4().hex
        granted: Final = _send(
            rig, other.key, f"/a2a/{rig.allowed.agent_id}", method="message/send", version=None, request_id=success_id
        )
        assert granted.status_code == 200, granted.text
        success_rows: Final = eventually(lambda: _rows(rig, other), lambda found: len(found) == 1, seconds=_CONVERGE)
        success_request: Final = success_rows[0]["request_id"]
        request_id: Final = "spoof-" + uuid.uuid4().hex
        spoof: Final[dict[str, JsonValue]] = {
            "agent_id": "spoof",
            "metadata": {
                "agent_id": "spoof",
                "user_api_key_alias": "spoof",
                "user_api_key_team_id": "spoof",
                "tags": ["x"],
            },
            "litellm_call_id": success_request,
        }
        body: Final = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "SendMessage",
            "params": {**_params("SendMessage", uuid.uuid4().hex, "1.0"), "litellm_call_id": "spoof2"},
            **spoof,
        }
        response: Final = rig.proxy.client.post(
            f"/a2a/{target}", headers={"Authorization": f"Bearer {caller.key}", "A2A-Version": "1.0"}, json=body
        )
        expected: Final = _refusal(target, response) if outcome == "denied" else _not_found(request_id, unknown)
        assert (response.status_code, response.content) == (403 if outcome == "denied" else 404, expected)
        model: Final = f"a2a_agent/{rig.denied.name}" if outcome == "denied" else f"a2a_agent/{unknown}"
        row: Final = _assert_one_failure(rig, caller, status=response.status_code, agent_id=target, model=model)
        assert row["request_id"] not in {success_request, "spoof2", request_id}, row
        metadata: Final = _metadata(row)
        assert (metadata["user_api_key_alias"], metadata["user_api_key_team_id"]) == (caller.alias, caller.team_id)
        assert rig.events.of("spoof") == ()
        after: Final = _rows(rig, other)
        assert [(item["request_id"], item["status"]) for item in after] == [(success_request, "success")], after


def test_two_keys_reusing_one_jsonrpc_id_get_one_denial_row_each(rig: _Rig) -> None:
    shared_id: Final = "same-" + uuid.uuid4().hex
    with rig.proxy.scenario() as scenario:
        callers: Final = (_caller(rig, scenario), _caller(rig, scenario))
        for caller in callers:
            response = _send(rig, caller.key, f"/a2a/{rig.denied.agent_id}", request_id=shared_id)
            assert (response.status_code, response.content) == (403, _refusal(rig.denied.agent_id, response))
        rows: Final = tuple(
            _assert_one_failure(
                rig, caller, status=403, agent_id=rig.denied.agent_id, model=f"a2a_agent/{rig.denied.name}"
            )
            for caller in callers
        )
        assert rows[0]["request_id"] != rows[1]["request_id"], rows


@pytest.mark.parametrize("outcome", ["denied", "unknown"])
def test_failing_or_transforming_failure_callbacks_never_change_the_denial_response(
    hostile: _Rig, outcome: str
) -> None:
    unknown: Final = "no-such-" + uuid.uuid4().hex
    request_id: Final = "h-" + uuid.uuid4().hex
    target: Final = hostile.denied.agent_id if outcome == "denied" else unknown
    with hostile.proxy.scenario() as scenario:
        caller: Final = _caller(hostile, scenario)
        response: Final = _send(hostile, caller.key, f"/a2a/{target}", request_id=request_id)
        expected: Final = (403, _refusal(target, response)) if outcome == "denied" else (404, _not_found(request_id, unknown))
        assert (response.status_code, response.content) == expected


@pytest.mark.parametrize("outcome", ["denied", "unknown"], ids=lambda outcome: "regression-" + outcome)
def test_failing_or_transforming_failure_callbacks_do_not_stop_the_denial_from_being_logged(
    hostile: _Rig, outcome: str
) -> None:
    unknown: Final = "no-such-" + uuid.uuid4().hex
    target: Final = hostile.denied.agent_id if outcome == "denied" else unknown
    with hostile.proxy.scenario() as scenario:
        caller: Final = _caller(hostile, scenario)
        response: Final = _send(hostile, caller.key, f"/a2a/{target}")
        assert response.status_code == (403 if outcome == "denied" else 404), response.text
        model: Final = f"a2a_agent/{hostile.denied.name}" if outcome == "denied" else f"a2a_agent/{unknown}"
        _assert_one_failure(hostile, caller, status=response.status_code, agent_id=target, model=model)


def test_denials_follow_the_model_denied_chat_logging_under_error_log_flags(flagged: _Rig) -> None:
    unknown: Final = "no-such-" + uuid.uuid4().hex
    request_id: Final = "f-" + uuid.uuid4().hex
    with flagged.proxy.scenario() as scenario:
        chat_caller: Final = _caller(flagged, scenario)
        denied_caller: Final = _caller(flagged, scenario)
        unknown_caller: Final = _caller(flagged, scenario)
        chat_alias: Final = "a2adeny-chat-" + uuid.uuid4().hex[:8]
        chat_key: Final = scenario.key(
            user_id=chat_caller.user_id, team_id=chat_caller.team_id, key_alias=chat_alias, models=["openai/gpt-5-mini"]
        )
        chat: Final = flagged.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": "openai/gpt-5.4-mini", "messages": [{"role": "user", "content": "x"}]},
            key=chat_key,
        )
        assert chat.status_code in {401, 403}, chat.text
        denied: Final = _send(flagged, denied_caller.key, f"/a2a/{flagged.denied.agent_id}")
        assert (denied.status_code, denied.content) == (403, _refusal(flagged.denied.agent_id, denied))
        missing: Final = _send(flagged, unknown_caller.key, f"/a2a/{unknown}", request_id=request_id)
        assert (missing.status_code, missing.content) == (404, _not_found(request_id, unknown))
        _assert_never_dispatched(flagged, scenario, "never-sent")
        chat_hooks: Final = held(
            lambda: _hooks(flagged.events.of(chat_alias)),
            lambda found: len(found) > 0,
            holding=_SETTLE,
            seconds=_CONVERGE,
        )
        assert _hooks(flagged.events.of(denied_caller.alias)) == chat_hooks
        assert _hooks(flagged.events.of(unknown_caller.alias)) == chat_hooks
        control: Final = _caller(flagged, scenario)
        granted: Final = _send(
            flagged, control.key, f"/a2a/{flagged.allowed.agent_id}", method="message/send", version=None
        )
        assert granted.status_code == 200, granted.text
        eventually(lambda: _rows(flagged, control), lambda found: len(found) == 1, seconds=_CONVERGE)
        assert (_rows(flagged, denied_caller), _rows(flagged, unknown_caller)) == ([], [])


def test_client_error_denials_never_raise_an_llm_exceptions_alert(alerting: tuple[_Rig, Wire]) -> None:
    rig, alerts = alerting
    unknown: Final = "no-such-" + uuid.uuid4().hex
    marker: Final = "boom-" + uuid.uuid4().hex

    def failing(request: Request) -> Reply:
        return Reply(
            status=500, body=_compact({"error": {"message": f"upstream exploded {marker}", "type": "server_error"}})
        )

    with rig.proxy.scenario() as scenario, wire_server(failing) as broken:
        caller: Final = _caller(rig, scenario)
        assert _send(rig, caller.key, f"/a2a/{rig.denied.agent_id}").status_code == 403
        assert _send(rig, caller.key, f"/a2a/{unknown}").status_code == 404
        model: Final = scenario.model(api_base=broken.url + "/v1", num_retries=0)
        failed: Final = rig.proxy.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "x"}]}
        )
        assert failed.status_code >= 500, failed.text
        texts: Final = eventually(
            lambda: tuple(JSON_OBJECT.validate_json(request.body).get("text") for request in alerts.drain()),
            lambda found: any(marker in str(text) for text in found),
            seconds=_CONVERGE,
        )
        assert not any(rig.denied.agent_id in str(text) or unknown in str(text) for text in texts), texts


def test_guard_model_denied_chat_logs_one_acompletion_failure(rig: _Rig) -> None:
    with rig.proxy.scenario() as scenario:
        user: Final = scenario.user()
        alias: Final = "a2adeny-chat-" + uuid.uuid4().hex[:8]
        key: Final = scenario.key(user_id=user, key_alias=alias, models=["openai/gpt-5-mini"])
        caller: Final = _Caller(key, alias, user, None)
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": "openai/gpt-5.4-mini", "messages": [{"role": "user", "content": "x"}]},
            key=key,
        )
        assert response.status_code == 403, response.text
        assert "key_model_access_denied" in response.text
        events: Final = held(
            lambda: rig.events.of(alias),
            lambda found: _hooks(found) == _ONE_FAILURE,
            holding=_SETTLE,
            seconds=_CONVERGE,
        )
        assert {event.status_code for event in events if event.hook == "post_call_failure"} == {403}
        rows: Final = held(
            lambda: _rows(rig, caller), lambda found: len(found) == 1, holding=_SETTLE, seconds=_CONVERGE
        )
        assert (rows[0]["status"], rows[0]["call_type"], rows[0]["spend"]) == ("failure", "acompletion", 0.0), rows


@pytest.mark.parametrize(
    ("method", "call_type"), [("message/send", "asend_message"), ("message/stream", "asend_message_streaming")]
)
def test_guard_granted_agent_succeeds_with_one_success_row_and_no_failure(
    rig: _Rig, method: str, call_type: str
) -> None:
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(rig, scenario)
        request_id: Final = "g-" + uuid.uuid4().hex
        response: Final = _send(
            rig, caller.key, f"/a2a/{rig.allowed.agent_id}", method=method, version=None, request_id=request_id
        )
        assert response.status_code == 200, response.text
        if method == "message/stream":
            assert response.text.startswith("data: ") and request_id in response.text, response.text
        else:
            assert object_value(JSON_OBJECT.validate_json(response.content)["result"])["parts"] == [
                {"kind": "text", "text": "granted"}
            ]
        held(
            lambda: rig.events.of(caller.alias),
            lambda found: _hooks(found) == ("async_success",),
            holding=_SETTLE,
            seconds=_CONVERGE,
        )
        rows: Final = held(
            lambda: _rows(rig, caller), lambda found: len(found) == 1, holding=_SETTLE, seconds=_CONVERGE
        )
        assert (rows[0]["status"], rows[0]["call_type"], rows[0]["agent_id"]) == (
            "success",
            call_type,
            rig.allowed.agent_id,
        ), rows


def test_guard_zero_budget_key_is_refused_once_without_double_logging(rig: _Rig) -> None:
    with rig.proxy.scenario() as scenario:
        user: Final = scenario.user()
        alias: Final = "a2adeny-budget-" + uuid.uuid4().hex[:8]
        key: Final = scenario.key(
            user_id=user, key_alias=alias, max_budget=0, object_permission={"agents": [rig.allowed.agent_id]}
        )
        caller: Final = _Caller(key, alias, user, None)
        message: Final = "budget-" + uuid.uuid4().hex
        response: Final = _send(
            rig, key, f"/a2a/{rig.allowed.agent_id}", method="message/send", version=None, message_id=message
        )
        assert response.status_code == 422, response.text
        assert "Budget has been exceeded" in response.text
        _assert_never_dispatched(rig, scenario, message)
        held(
            lambda: rig.events.of(alias),
            lambda found: _hooks(found) == _ONE_FAILURE,
            holding=_SETTLE,
            seconds=_CONVERGE,
        )
        rows: Final = held(
            lambda: _rows(rig, caller), lambda found: len(found) == 1, holding=_SETTLE, seconds=_CONVERGE
        )
        assert (rows[0]["status"], rows[0]["call_type"]) == ("failure", "asend_message"), rows


def test_guard_agent_card_discovery_keeps_its_refusals_and_writes_no_failure_row(rig: _Rig) -> None:
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(rig, scenario)
        headers: Final = {"Authorization": f"Bearer {caller.key}"}
        denied_card: Final = rig.proxy.client.get(
            f"/a2a/{rig.denied.agent_id}/.well-known/agent-card.json", headers=headers
        )
        assert (denied_card.status_code, denied_card.content) == (403, _refusal(rig.denied.agent_id))
        missing: Final = rig.proxy.client.get("/a2a/no-such-card/.well-known/agent-card.json", headers=headers)
        assert (missing.status_code, missing.content) == (404, _compact({"detail": "Agent 'no-such-card' not found"}))
        granted: Final = _send(rig, caller.key, f"/a2a/{rig.allowed.agent_id}", method="message/send", version=None)
        assert granted.status_code == 200, granted.text
        rows: Final = held(
            lambda: _rows(rig, caller), lambda found: len(found) == 1, holding=_SETTLE, seconds=_CONVERGE
        )
        assert [row["status"] for row in rows] == ["success"], rows
        assert _hooks(rig.events.of(caller.alias)) == ("async_success",)


def test_guard_jsonrpc_1_0_envelope_keeps_its_invalid_request_body(rig: _Rig) -> None:
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(rig, scenario)
        request_id: Final = "v-" + uuid.uuid4().hex
        response: Final = _send(
            rig,
            caller.key,
            f"/a2a/{rig.allowed.agent_id}",
            method="message/send",
            version=None,
            request_id=request_id,
            extra={"jsonrpc": "1.0"},
        )
        assert (response.status_code, response.content) == (
            400,
            _compact(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32600, "message": "Invalid Request: jsonrpc must be '2.0'"},
                }
            ),
        )
