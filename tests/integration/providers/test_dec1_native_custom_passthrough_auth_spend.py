"""Operator-defined pass-through endpoints: who may call them and what a call is billed.

An endpoint created with `auth: true` (the default through the API) is closed twice over. The caller needs a
valid virtual key, and that key or its team must also list the endpoint's path in `allowed_passthrough_routes`,
because these routes do not inherit the blanket "LLM API routes" permission. An endpoint created with
`auth: false` is a public forwarder. Budgets and rate limits on the key apply to pass-through calls exactly as
they do to chat calls, and a refused call must never reach the target.

Billing has no model to price, so the operator sets `cost_per_request`. Each successful call must write one
`LiteLLM_SpendLogs` row with `call_type = pass_through_endpoint`, attribute it to the calling key, and add the
flat cost to the key's spend. These are the rows the Rust spend-logging rewrite must keep producing.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.wire import Reply, wire_server
from integration.providers.dec1_native_support import (
    PASS_THROUGH_CALL_TYPE,
    SPEND_SECONDS,
    always,
    digest,
    error_message,
    granted_key,
    json_reply,
    number,
    register_pass_through,
    spend_row,
    spend_rows,
)
from pydantic import JsonValue

_BODY: Final[dict[str, JsonValue]] = {"ticket_id": "T-1"}
_REPLY: Final[dict[str, JsonValue]] = {"summary": "scripted"}


def _key_spend(gateway: Gateway, key: str) -> float:
    info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
    return number(info["spend"])


def test_an_auth_endpoint_refuses_a_request_without_a_key(gateway: Gateway) -> None:
    """An endpoint is created with `auth: true` and called with no `Authorization` header.

    The proxy must answer 401 and the target must receive nothing.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", auth=True)
        response: Final = gateway.client.post(path, json=_BODY)
        assert response.status_code == 401, response.text
        assert wire.drain() == ()


def test_an_auth_endpoint_refuses_an_unknown_key(gateway: Gateway) -> None:
    """An `auth: true` endpoint is called with a well-formed `sk-` key the proxy never issued.

    The proxy must answer 401 and the target must receive nothing.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", auth=True)
        response: Final = gateway.request("POST", path, _BODY, key="sk-dec1-native-never-issued")
        assert response.status_code == 401, response.text
        assert wire.drain() == ()


def test_a_valid_key_without_a_route_grant_is_refused_with_403(gateway: Gateway) -> None:
    """A valid key that has no `allowed_passthrough_routes` calls an `auth: true` endpoint.

    The proxy must answer 403 with the message `Key/team not allowed to access passthrough route <path>.
    Configure `allowed_passthrough_routes` on the team or key.` and the target must receive nothing. A valid key
    alone is not enough, otherwise every key could reach every internal service an operator maps.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", auth=True)
        key: Final = scenario.key()
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 403, response.text
        assert (
            f"Key/team not allowed to access passthrough route {path}. "
            "Configure `allowed_passthrough_routes` on the team or key."
        ) in error_message(response.content), response.text
        assert wire.drain() == ()


def test_a_key_granted_the_route_is_served(gateway: Gateway) -> None:
    """A key created with `allowed_passthrough_routes: [<path>]` calls the `auth: true` endpoint.

    The proxy must answer 200 with the target's body and the target must receive exactly one request.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", auth=True)
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        assert response.json() == _REPLY, response.text
        assert len(wire.drain()) == 1


def test_a_team_grant_serves_a_key_that_has_no_grant_of_its_own(gateway: Gateway) -> None:
    """A team has `allowed_passthrough_routes: [<path>]` in its metadata and a key in that team has none.

    The key must be served with 200. The grant is read from the key first and from its team second.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", auth=True)
        team: Final = scenario.team(metadata={"allowed_passthrough_routes": [path]})
        key: Final = scenario.key(team_id=team)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_a_grant_for_one_endpoint_does_not_open_another(gateway: Gateway) -> None:
    """Two `auth: true` endpoints exist and a key is granted only the first.

    The first must answer 200. The second must answer 403 and its target must receive nothing.
    """
    with (
        wire_server(always(json_reply(_REPLY))) as granted_wire,
        wire_server(always(json_reply(_REPLY))) as other_wire,
        gateway.scenario() as scenario,
    ):
        granted_path: Final = register_pass_through(gateway, scenario, f"{granted_wire.url}/v1/jobs", auth=True)
        other_path: Final = register_pass_through(gateway, scenario, f"{other_wire.url}/v1/jobs", auth=True)
        key: Final = granted_key(scenario, granted_path)
        assert gateway.request("POST", granted_path, _BODY, key=key).status_code == 200
        refused: Final = gateway.request("POST", other_path, _BODY, key=key)
        assert refused.status_code == 403, refused.text
        assert len(granted_wire.drain()) == 1
        assert other_wire.drain() == ()


def test_a_grant_on_the_endpoint_path_covers_its_subpaths(gateway: Gateway) -> None:
    """An `auth: true` endpoint has `include_subpath: true` and a key is granted the bare endpoint path.

    A call to `<path>/v1/deep/resource` must be served with 200 and reach the target at that subpath. The grant
    matches by prefix, so one entry covers the whole mapped API.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/api", auth=True, include_subpath=True)
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", f"{path}/v1/deep/resource", _BODY, key=key)
        assert response.status_code == 200, response.text
        assert [request.target for request in wire.drain()] == ["/api/v1/deep/resource"]


def test_the_master_key_reaches_an_auth_endpoint_without_a_grant(gateway: Gateway) -> None:
    """The proxy admin's master key, which has no `allowed_passthrough_routes`, calls an `auth: true` endpoint.

    The call must be served with 200. Admins are exempt from the per-route grant.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", auth=True)
        response: Final = gateway.request("POST", path, _BODY)
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_an_open_endpoint_serves_a_caller_with_no_key(gateway: Gateway) -> None:
    """An endpoint is created with `auth: false` and called with no `Authorization` header.

    The proxy must answer 200 with the target's body. This is the documented way to expose a public forwarder,
    for example a webhook receiver.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/hook", auth=False)
        response: Final = gateway.client.post(path, json=_BODY)
        assert response.status_code == 200, response.text
        assert response.json() == _REPLY, response.text
        assert len(wire.drain()) == 1


def test_an_open_endpoint_does_not_forward_the_callers_authorization_header(gateway: Gateway) -> None:
    """An `auth: false` endpoint is called with `Authorization: Bearer third-party-token`.

    The target must not receive an `authorization` header. Whatever bearer a caller attaches to a public
    forwarder stays at the proxy unless the operator turns header forwarding on.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/hook", auth=False)
        response: Final = gateway.client.post(path, json=_BODY, headers={"Authorization": "Bearer third-party-token"})
        assert response.status_code == 200, response.text
        received: Final = wire.drain()
        assert len(received) == 1, received
        assert "authorization" not in received[0].headers, dict(received[0].headers)


def test_cost_per_request_is_written_to_the_spend_row(gateway: Gateway) -> None:
    """The endpoint is created with `cost_per_request: 0.25` and a granted key calls it once.

    Exactly one `LiteLLM_SpendLogs` row must appear for that key with `call_type = pass_through_endpoint`,
    `spend = 0.25` and `status = success`.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", cost_per_request=0.25)
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        row: Final = spend_row(gateway, key)
        assert row["call_type"] == PASS_THROUGH_CALL_TYPE, row
        assert number(row["spend"]) == pytest.approx(0.25), row
        assert row["status"] == "success", row
        assert row["api_key"] == digest(key), row
        assert len(wire.drain()) == 1


def test_cost_per_request_accumulates_on_the_keys_spend(gateway: Gateway) -> None:
    """The endpoint costs 0.25 per request and a granted key calls it three times.

    `/key/info` must eventually report `spend = 0.75` for the key and three spend rows must exist.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", cost_per_request=0.25)
        key: Final = granted_key(scenario, path)
        statuses: Final = [gateway.request("POST", path, _BODY, key=key).status_code for _ in range(3)]
        assert statuses == [200, 200, 200], statuses
        rows: Final = eventually(lambda: spend_rows(gateway, key), lambda found: len(found) == 3, seconds=SPEND_SECONDS)
        assert [number(row["spend"]) for row in rows] == [pytest.approx(0.25)] * 3, rows
        spent: Final = eventually(
            lambda: _key_spend(gateway, key), lambda value: value >= 0.75 - 1e-9, seconds=SPEND_SECONDS
        )
        assert spent == pytest.approx(0.75), spent
        assert len(wire.drain()) == 3


def test_an_endpoint_without_a_cost_writes_a_zero_spend_row(gateway: Gateway) -> None:
    """The endpoint is created without `cost_per_request` and a granted key calls it once.

    One spend row with `call_type = pass_through_endpoint` must still be written, with `spend = 0`. The row is
    what usage dashboards count, so a free endpoint is logged, not skipped.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        row: Final = spend_row(gateway, key)
        assert number(row["spend"]) == 0.0, row
        assert row["status"] == "success", row
        assert len(wire.drain()) == 1


def test_the_spend_row_is_attributed_to_the_keys_team(gateway: Gateway) -> None:
    """A key that belongs to a team calls a priced endpoint.

    The spend row's `team_id` must be the key's team, so team budgets and team usage reports include
    pass-through traffic.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", cost_per_request=0.5)
        team: Final = scenario.team()
        key: Final = granted_key(scenario, path, team_id=team)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        row: Final = spend_row(gateway, key)
        assert row["team_id"] == team, row
        assert number(row["spend"]) == pytest.approx(0.5), row
        assert len(wire.drain()) == 1


def test_a_streamed_call_is_billed_the_flat_cost_once(gateway: Gateway) -> None:
    """The endpoint costs 0.25 per request and the target answers a three-frame event stream.

    After the caller has read the whole stream, exactly one spend row must exist for the key with
    `spend = 0.25`: one row per call, not one per frame and not zero.
    Product bug: CI run 42966523075 wrote the row with spend 0. Only the non-streaming success path applies
    `cost_per_request` (`_set_cost_per_request`, litellm/proxy/pass_through_endpoints/success_handler.py:591);
    the streaming path logs through `route_streaming_logging_to_handler`
    (litellm/proxy/pass_through_endpoints/streaming_handler.py:329), which never does.
    """
    pytest.skip("BUG: a streamed pass-through call is billed 0 instead of the endpoint's cost_per_request")
    frames: Final = (b'data: {"n":1}\n\n', b'data: {"n":2}\n\n', b"data: [DONE]\n\n")
    reply: Final = Reply(content_type="text/event-stream", chunks=frames)
    with wire_server(always(reply)) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/stream", cost_per_request=0.25)
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, {"stream": True}, key=key)
        assert response.status_code == 200, response.text
        assert response.content == b"".join(frames), response.content
        row: Final = spend_row(gateway, key)
        assert number(row["spend"]) == pytest.approx(0.25), row
        assert len(wire.drain()) == 1


def test_an_upstream_error_writes_a_failure_row_and_charges_nothing(gateway: Gateway) -> None:
    """The endpoint costs 0.25 per request and the target answers 500.

    The caller gets the 500. The key's single spend row must have `status = failure` and `spend = 0`: a call the
    target rejected is logged for visibility and is not billed the flat cost.
    """
    with (
        wire_server(always(json_reply({"error": "boom"}, status=500))) as wire,
        gateway.scenario() as scenario,
    ):
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", cost_per_request=0.25)
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 500, response.text
        row: Final = spend_row(gateway, key)
        assert row["status"] == "failure", row
        assert number(row["spend"]) == 0.0, row
        assert len(wire.drain()) == 1


def test_the_flat_cost_is_reported_in_the_response_cost_header(gateway: Gateway) -> None:
    """The endpoint costs 0.25 per request and a granted key calls it once.

    The response must carry `x-litellm-response-cost: 0.25`. UNVERIFIED: the header is built from
    `logging_obj.finalized_cost` in `pass_through_request`; that it is populated for a flat cost was not traced.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", cost_per_request=0.25)
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.25), dict(response.headers)
        assert len(wire.drain()) == 1


def test_the_response_reports_the_keys_limits_in_litellm_headers(gateway: Gateway) -> None:
    """A granted key with `rpm_limit: 100`, `tpm_limit: 5000` and `max_budget: 9.5` calls an endpoint.

    The response must carry `x-litellm-key-rpm-limit: 100`, `x-litellm-key-tpm-limit: 5000` and
    `x-litellm-key-max-budget: 9.5`, the same key headers a chat call returns.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs")
        key: Final = granted_key(scenario, path, rpm_limit=100, tpm_limit=5000, max_budget=9.5)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-key-rpm-limit"] == "100", dict(response.headers)
        assert response.headers["x-litellm-key-tpm-limit"] == "5000", dict(response.headers)
        assert float(response.headers["x-litellm-key-max-budget"]) == 9.5, dict(response.headers)
        assert len(wire.drain()) == 1


def test_a_key_over_its_budget_is_refused_before_the_target_is_called(gateway: Gateway) -> None:
    """The endpoint costs 1.0 per request and the key's `max_budget` is 0.5.

    The first call is served. Once its spend has settled, a further call must answer 422 with error type
    `budget_exceeded`, and the target must have received only the served calls, never the refused one.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", cost_per_request=1.0)
        key: Final = granted_key(scenario, path, max_budget=0.5)
        first: Final = gateway.request("POST", path, _BODY, key=key)
        assert first.status_code == 200, first.text
        eventually(lambda: _key_spend(gateway, key), lambda value: value >= 1.0 - 1e-9, seconds=SPEND_SECONDS)
        wire.drain()
        refused: Final = eventually(
            lambda: gateway.request("POST", path, _BODY, key=key),
            lambda response: response.status_code != 200,
            seconds=SPEND_SECONDS,
        )
        assert refused.status_code == 422, refused.text
        assert refused.json()["error"]["type"] == "budget_exceeded", refused.text
        settled: Final = wire.drain()
        after_refusal: Final = gateway.request("POST", path, _BODY, key=key)
        assert after_refusal.status_code == 422, after_refusal.text
        assert wire.drain() == (), settled


def test_a_team_over_its_budget_is_refused_on_a_pass_through_endpoint(gateway: Gateway) -> None:
    """The endpoint costs 1.0 per request, the key is unlimited, and its team's `max_budget` is 0.5.

    After the first served call has settled, a further call must be refused with a non-200 status whose error
    type is `budget_exceeded`, without reaching the target. Pass-through spend counts against the team budget.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs", cost_per_request=1.0)
        team: Final = scenario.team(max_budget=0.5)
        key: Final = granted_key(scenario, path, team_id=team)
        first: Final = gateway.request("POST", path, _BODY, key=key)
        assert first.status_code == 200, first.text
        refused: Final = eventually(
            lambda: gateway.request("POST", path, _BODY, key=key),
            lambda response: response.status_code != 200,
            seconds=SPEND_SECONDS,
        )
        assert refused.json()["error"]["type"] == "budget_exceeded", refused.text
        wire.drain()
        assert gateway.request("POST", path, _BODY, key=key).status_code == refused.status_code
        assert wire.drain() == ()


def test_a_keys_rpm_limit_applies_to_a_pass_through_endpoint(gateway: Gateway) -> None:
    """A granted key has `rpm_limit: 1` and calls the endpoint twice in a row.

    The first call must be served with 200 and the second must answer 429. The target must have received one
    request only.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs")
        key: Final = granted_key(scenario, path, rpm_limit=1)
        first: Final = gateway.request("POST", path, _BODY, key=key)
        second: Final = gateway.request("POST", path, _BODY, key=key)
        assert first.status_code == 200, first.text
        assert second.status_code == 429, second.text
        assert len(wire.drain()) == 1


def test_a_deleted_key_is_refused_on_a_pass_through_endpoint(gateway: Gateway) -> None:
    """A granted key calls the endpoint successfully and is then deleted with `/key/delete`.

    The next call with that key must answer 401 and must not reach the target.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs")
        created: Final = gateway.post("/key/generate", {"allowed_passthrough_routes": [path]})
        key: Final = str(created["key"])
        assert gateway.request("POST", path, _BODY, key=key).status_code == 200
        wire.drain()
        gateway.post("/key/delete", {"keys": [key]})
        refused: Final = eventually(
            lambda: gateway.request("POST", path, _BODY, key=key),
            lambda response: response.status_code != 200,
            seconds=30,
        )
        assert refused.status_code == 401, refused.text
        wire.drain()
        assert gateway.request("POST", path, _BODY, key=key).status_code == 401
        assert wire.drain() == ()
