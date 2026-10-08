from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pytest
import yaml
from _logging_only_scope_support import (
    BASE_DEFAULT_CACHE_HIT_DIRECTIONS,
    BASE_DEFAULT_NORMAL_DIRECTIONS,
    BASE_DEFAULT_UPSTREAM_FAILURE_DIRECTIONS,
    JSON_OBJECT,
    ChaosCall,
    ClientKind,
    Direction,
    Endpoint,
    _assert_response_id,
    _cache_hit,
    _call_cache_client,
    _call_client,
    _configuration,
    _database_guardrail,
    _direction,
    _directions_for_audit_leg,
    _directions_for_scope,
    _drain_upstream,
    _empty_proxy_configuration,
    _guardrail_entries,
    _guardrail_mode_status_pairs,
    _is_base_audit_leg,
    _json_contains_exact_string,
    _policy_call_id,
    _policy_call_id_matches,
    _provider_response,
    _response_body_without_ids,
    _response_text,
    _spend_row_for_call_id,
    _spend_row_for_response_id,
    _spend_rows,
    _spend_rows_for_calls,
    _spend_rows_matching_call,
    wire_server,
)
from _logging_only_scope_support import (
    _record_audit_properties as _record_audit_properties,
)
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.process import owned_proxy
from integration._support.upstream import delete_scenario, register_scenario
from integration._support.wire import Reply, Request
from pydantic import JsonValue

from tests.integration.cost_calculation.cost_tracking_case import JsonResponse


@pytest.mark.parametrize(
    ("row_id", "endpoint", "stream", "client_kind", "scope", "include_scope", "block_directions"),
    (
        pytest.param("A1", "chat", False, "openai_sync", "input", True, (), id="A1-chat-input"),
        pytest.param("A2", "chat", False, "openai_sync", "output", True, (), id="A2-chat-output"),
        pytest.param("A3", "chat", False, "openai_sync", "both", True, (), id="A3-chat-both"),
        pytest.param("A4", "chat", False, "httpx", None, False, (), id="A4-chat-missing-scope"),
        pytest.param("A5", "chat", False, "httpx", None, True, (), id="A5-chat-null-scope"),
        pytest.param("A6", "chat", True, "openai_async", "input", True, (), id="A6-chat-stream-async-input"),
        pytest.param("A7", "chat", True, "openai_async", "output", True, (), id="A7-chat-stream-async-output"),
        pytest.param("A8", "messages", False, "anthropic_sync", "input", True, (), id="A8-messages-input"),
        pytest.param("A9", "messages", False, "anthropic_sync", "output", True, (), id="A9-messages-output"),
        pytest.param(
            "A10",
            "messages",
            True,
            "anthropic_async",
            "output",
            True,
            (),
            id="A10-messages-stream-async-output",
        ),
        pytest.param(
            "A11",
            "messages",
            True,
            "anthropic_async",
            "input",
            True,
            (),
            id="A11-messages-stream-async-input",
        ),
        pytest.param("A12", "responses", False, "openai_async", "input", True, (), id="A12-responses-async-input"),
        pytest.param("A13", "responses", False, "openai_async", "output", True, (), id="A13-responses-async-output"),
        pytest.param("A14", "responses", True, "openai_sync", "output", True, (), id="A14-responses-stream-output"),
        pytest.param("A15", "responses", True, "openai_sync", "input", True, (), id="A15-responses-stream-input"),
        pytest.param("B1", "chat", False, "openai_sync", "input", True, ("request",), id="B1-logging-block-input"),
        pytest.param("B2", "chat", False, "openai_sync", "output", True, ("response",), id="B2-logging-block-output"),
        pytest.param(
            "B3",
            "chat",
            False,
            "openai_sync",
            "both",
            True,
            ("request",),
            id="B3-logging-block-both",
        ),
    ),
)
def test_runtime_directional_scope_matches_client_call_and_spend_log(
    gateway: Gateway,
    tmp_path: Path,
    row_id: str,
    endpoint: Endpoint,
    stream: bool,
    client_kind: ClientKind,
    scope: str | None,
    include_scope: bool,
    block_directions: tuple[Direction, ...],
) -> None:
    identity: Final = f"logging-scope-{row_id.lower()}-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic request {identity}"
    reply: Final = f"synthetic response {identity}"
    scenario_id: Final = f"phase12-{row_id.lower()}-{uuid.uuid4().hex}"
    baseline_call_id: Final = f"{scenario_id}-baseline"
    guarded_call_id: Final = f"{scenario_id}-guarded"
    provider_response: Final = _provider_response(endpoint, scenario_id, reply, stream)
    upstream_handle: Final = register_scenario(scenario_id, provider_response)

    def policy(request: Request) -> Reply:
        payload: Final = JSON_OBJECT.validate_json(request.body)
        direction: Final = _direction(payload)
        verdict: Final = (
            {"action": "BLOCKED", "blocked_reason": "synthetic logging-only denial"}
            if direction in block_directions
            else {"action": "NONE"}
        )
        return Reply(body=json.dumps(verdict).encode())

    try:
        with gateway.scenario() as scenario:
            api_base: Final = (
                upstream_handle.api_base() if endpoint == "messages" else f"{upstream_handle.api_base()}/v1"
            )
            model: Final = scenario.model(
                model={
                    "chat": "openai/gpt-4o-mini",
                    "messages": "anthropic/claude-3-7-sonnet-20250219",
                    "responses": "openai/gpt-4.1-mini",
                }[endpoint],
                api_base=api_base,
                api_key="synthetic-provider-key",
            )
            with wire_server(policy) as guardrail:
                config: Final = _configuration(tmp_path, identity, guardrail.url, scope, include_scope=include_scope)
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    baseline: Final = _call_client(
                        client_kind, endpoint, gateway, model, prompt, stream, baseline_call_id
                    )
                    baseline_upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert baseline.status == 200, baseline.body
                    assert baseline.text == reply, baseline
                    assert len(baseline_upstream) == 1, baseline_upstream
                    result: Final = _call_client(
                        client_kind, endpoint, candidate, model, prompt, stream, guarded_call_id
                    )
                    assert (result.status, _response_body_without_ids(result.body)) == (
                        baseline.status,
                        _response_body_without_ids(baseline.body),
                    ), (result, baseline)
                    assert result.text == reply, result
                    observed_upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(observed_upstream) == 1, observed_upstream
                    assert prompt in json.dumps(observed_upstream[0]["body"]), observed_upstream
                    base_default_directions: Final = BASE_DEFAULT_NORMAL_DIRECTIONS[(endpoint, stream)]
                    expected_directions: Final = _directions_for_audit_leg(base_default_directions, scope)
                    directions_to_collect: Final = _directions_for_audit_leg(base_default_directions, scope)
                    eventually(
                        lambda: guardrail.received.qsize(),
                        lambda count: count >= len(directions_to_collect),
                        seconds=20,
                    )
                    calls: Final = guardrail.drain()
                    payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in calls)
                    observed_directions: Final = tuple(_direction(payload) for payload in payloads)
                    assert all(_policy_call_id_matches(payload, guarded_call_id) for payload in payloads), payloads
                    assert tuple(payload["texts"] for payload in payloads) == tuple(
                        [prompt] if direction == "request" else [reply] for direction in observed_directions
                    ), payloads
                    rows: Final = _spend_rows(model, 2)
                    guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                    assert len(guarded_rows) == 1, rows
                    _assert_response_id(endpoint, str(guarded_rows[0]["request_id"]), result.response_id, scenario_id)
                    entries: Final = _guardrail_entries(guarded_rows[0])
                    observed_entries: Final = tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    )
                    expected_entries: Final = tuple(
                        (
                            identity,
                            "logging_only",
                            "guardrail_intervened" if direction in block_directions else "success",
                        )
                        for direction in expected_directions
                    )
                    assert (
                        tuple(sorted(observed_directions)),
                        tuple(sorted(observed_entries)),
                    ) == (tuple(sorted(expected_directions)), tuple(sorted(expected_entries))), (
                        payloads,
                        entries,
                        rows,
                    )
    finally:
        delete_scenario(upstream_handle)


@pytest.mark.parametrize(
    ("endpoint", "stream", "client_kind"),
    (
        pytest.param("chat", False, "openai_sync", id="A-default-chat-nonstream"),
        pytest.param("chat", True, "openai_async", id="A-default-chat-stream"),
        pytest.param("messages", False, "anthropic_sync", id="A-default-messages-nonstream"),
        pytest.param("messages", True, "anthropic_async", id="A-default-messages-stream"),
        pytest.param("responses", False, "openai_async", id="A-default-responses-nonstream"),
        pytest.param("responses", True, "openai_sync", id="A-default-responses-stream"),
    ),
)
def test_A_unset_scope_matches_measured_endpoint_stream_default(
    gateway: Gateway,
    tmp_path: Path,
    endpoint: Endpoint,
    stream: bool,
    client_kind: ClientKind,
) -> None:
    identity: Final = f"logging-scope-a-default-{endpoint}-{stream}-{uuid.uuid4().hex}"
    scenario_id: Final = f"phase12-a-default-{endpoint}-{stream}-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic unset-scope request {identity}"
    reply: Final = f"synthetic unset-scope response {identity}"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response(endpoint, scenario_id, reply, stream))

    def policy(_request: Request) -> Reply:
        return Reply(body=b'{"action":"NONE"}')

    try:
        with gateway.scenario() as scenario:
            api_base: Final = (
                upstream_handle.api_base() if endpoint == "messages" else f"{upstream_handle.api_base()}/v1"
            )
            model: Final = scenario.model(
                model={
                    "chat": "openai/gpt-4o-mini",
                    "messages": "anthropic/claude-3-7-sonnet-20250219",
                    "responses": "openai/gpt-4.1-mini",
                }[endpoint],
                api_base=api_base,
                api_key="synthetic-provider-key",
            )
            with wire_server(policy) as guardrail:
                config: Final = _configuration(tmp_path, identity, guardrail.url, None, include_scope=False)
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    baseline: Final = _call_client(
                        client_kind,
                        endpoint,
                        gateway,
                        model,
                        prompt,
                        stream,
                        f"{scenario_id}-baseline",
                    )
                    assert baseline.status == 200 and baseline.text == reply, baseline
                    assert len(_drain_upstream(gateway.upstream_url)) == 1
                    result: Final = _call_client(
                        client_kind,
                        endpoint,
                        candidate,
                        model,
                        prompt,
                        stream,
                        f"{scenario_id}-candidate",
                    )
                    assert (result.status, _response_body_without_ids(result.body)) == (
                        baseline.status,
                        _response_body_without_ids(baseline.body),
                    ), (result, baseline)
                    assert result.text == reply, result
                    upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(upstream) == 1 and prompt in json.dumps(upstream[0]["body"]), upstream
                    expected_directions: Final = BASE_DEFAULT_NORMAL_DIRECTIONS[(endpoint, stream)]
                    eventually(
                        lambda: guardrail.received.qsize(),
                        lambda count: count == len(expected_directions),
                        seconds=20,
                    )
                    payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                    assert tuple(sorted(_direction(payload) for payload in payloads)) == tuple(
                        sorted(expected_directions)
                    ), payloads
                    assert all(_policy_call_id_matches(payload, f"{scenario_id}-candidate") for payload in payloads), (
                        payloads
                    )
                    assert tuple(payload["texts"] for payload in payloads) == tuple(
                        [prompt] if _direction(payload) == "request" else [reply] for payload in payloads
                    ), payloads
                    rows: Final = _spend_rows(model, 2)
                    guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                    assert len(guarded_rows) == 1, rows
                    _assert_response_id(
                        endpoint,
                        str(guarded_rows[0]["request_id"]),
                        result.response_id,
                        scenario_id,
                    )
                    entries: Final = _guardrail_entries(guarded_rows[0])
                    assert tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    ) == tuple((identity, "logging_only", "success") for _ in expected_directions), entries
    finally:
        delete_scenario(upstream_handle)


@pytest.mark.parametrize("inventory_id", (pytest.param("B4", id="B4-monitor-usage-detail"),))
def test_logging_only_monitor_counts_only_the_observed_direction(
    gateway: Gateway, tmp_path: Path, inventory_id: str
) -> None:
    input_identity: Final = f"logging-scope-b4-input-{uuid.uuid4().hex}"
    output_identity: Final = f"logging-scope-b4-output-{uuid.uuid4().hex}"
    scenario_id: Final = f"phase12-{inventory_id.lower()}-{uuid.uuid4().hex}"
    prompt_by_identity: Final = {
        input_identity: f"synthetic input {input_identity}",
        output_identity: f"synthetic input {output_identity}",
    }
    provider_reply: Final = f"synthetic response {scenario_id}"
    upstream_handle: Final = register_scenario(
        scenario_id, _provider_response("chat", scenario_id, provider_reply, False)
    )

    def policy(request: Request) -> Reply:
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic monitor denial"}).encode())

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            with wire_server(policy) as input_guardrail, wire_server(policy) as output_guardrail:
                config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
                config["litellm_settings"]["cache"] = False
                config["guardrails"] = [
                    {
                        "guardrail_name": input_identity,
                        "litellm_params": {
                            "guardrail": "generic_guardrail_api",
                            "mode": "logging_only",
                            "logging_only_scope": "input",
                            "default_on": False,
                            "api_base": input_guardrail.url,
                            "api_key": "synthetic-guardrail-key",
                            "extra_headers": ["x-litellm-call-id"],
                        },
                    },
                    {
                        "guardrail_name": output_identity,
                        "litellm_params": {
                            "guardrail": "generic_guardrail_api",
                            "mode": "logging_only",
                            "logging_only_scope": "output",
                            "default_on": False,
                            "api_base": output_guardrail.url,
                            "api_key": "synthetic-guardrail-key",
                            "extra_headers": ["x-litellm-call-id"],
                        },
                    },
                ]
                config_path: Final = tmp_path / "b4.yaml"
                config_path.write_text(yaml.safe_dump(config))
                with owned_proxy(gateway, tmp_path, {}, config=config_path, workers=1) as candidate:
                    cases: Final = tuple(
                        (
                            identity,
                            gateway.request(
                                "POST",
                                "/v1/chat/completions",
                                {
                                    "model": model,
                                    "messages": [{"role": "user", "content": prompt_by_identity[identity]}],
                                },
                                headers={"x-litellm-call-id": f"{scenario_id}-{identity}-baseline"},
                            ),
                            candidate.request(
                                "POST",
                                "/v1/chat/completions",
                                {
                                    "model": model,
                                    "messages": [{"role": "user", "content": prompt_by_identity[identity]}],
                                    "guardrails": [identity],
                                },
                                headers={"x-litellm-call-id": f"{scenario_id}-{identity}"},
                            ),
                        )
                        for identity in (input_identity, output_identity)
                    )
                    observed_upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(observed_upstream) == 4, observed_upstream
                    for identity, baseline_response, guarded_response in cases:
                        assert baseline_response.status_code == guarded_response.status_code == 200, (
                            baseline_response.text,
                            guarded_response.text,
                        )
                        assert _response_body_without_ids(JSON_OBJECT.validate_python(baseline_response.json())) == (
                            _response_body_without_ids(JSON_OBJECT.validate_python(guarded_response.json()))
                        ), (
                            baseline_response.text,
                            guarded_response.text,
                        )
                        assert baseline_response.json()["choices"][0]["message"]["content"] == provider_reply
                        assert (
                            sum(
                                prompt_by_identity[identity] in json.dumps(observation["body"])
                                for observation in observed_upstream
                            )
                            == 2
                        ), observed_upstream
                    expected_call_ids: Final = tuple(
                        f"{scenario_id}-{identity}" for identity in (input_identity, output_identity)
                    )
                    eventually(
                        lambda: input_guardrail.received.qsize(),
                        lambda count: count >= len(cases),
                        seconds=20,
                    )
                    eventually(
                        lambda: output_guardrail.received.qsize(),
                        lambda count: count >= len(cases),
                        seconds=20,
                    )
                    policy_payloads: Final = {
                        input_identity: tuple(JSON_OBJECT.validate_json(call.body) for call in input_guardrail.drain()),
                        output_identity: tuple(
                            JSON_OBJECT.validate_json(call.body) for call in output_guardrail.drain()
                        ),
                    }
                    for identity in (input_identity, output_identity):
                        expected_direction: Final = (
                            "request" if _is_base_audit_leg() or identity == input_identity else "response"
                        )
                        assert len(policy_payloads[identity]) == len(expected_call_ids), policy_payloads[identity]
                        for case_identity in (input_identity, output_identity):
                            call_id: Final = f"{scenario_id}-{case_identity}"
                            calls_for_id: Final = tuple(
                                payload
                                for payload in policy_payloads[identity]
                                if _policy_call_id_matches(payload, call_id)
                            )
                            call_summary: Final = tuple(
                                (
                                    _direction(payload),
                                    payload.get("litellm_call_id"),
                                    _policy_call_id(payload),
                                    tuple(payload["texts"]),
                                )
                                for payload in calls_for_id
                            )
                            assert tuple(_direction(payload) for payload in calls_for_id) == (expected_direction,), (
                                call_id,
                                call_summary,
                            )
                            expected_text: Final = (
                                prompt_by_identity[case_identity] if expected_direction == "request" else provider_reply
                            )
                            assert tuple(tuple(payload["texts"]) for payload in calls_for_id) == ((expected_text,),), (
                                call_id,
                                call_summary,
                            )
                    rows: Final = _spend_rows_for_calls(
                        (model,),
                        tuple(
                            (model, str(response.json()["id"]), f"{scenario_id}-{identity}")
                            for identity, _baseline_response, response in cases
                        ),
                    )
                    guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                    assert len(guarded_rows) == 2, rows
                    expected_entries: Final = tuple(
                        sorted(
                            (
                                (input_identity, "logging_only", "guardrail_intervened"),
                                (output_identity, "logging_only", "guardrail_intervened"),
                            )
                        )
                    )
                    for identity, _baseline_response, response in cases:
                        call_id: Final = f"{scenario_id}-{identity}"
                        matching_rows: Final = _spend_rows_matching_call(rows, model, call_id)
                        assert len(matching_rows) == 1, (call_id, rows)
                        _assert_response_id(
                            "chat",
                            str(matching_rows[0]["request_id"]),
                            str(response.json()["id"]),
                            scenario_id,
                        )
                        entries: Final = _guardrail_entries(matching_rows[0])
                        assert (
                            tuple(
                                sorted(
                                    (
                                        entry["guardrail_name"],
                                        entry["guardrail_mode"],
                                        entry["guardrail_status"],
                                    )
                                    for entry in entries
                                )
                            )
                            == expected_entries
                        ), entries
                    listed: Final = candidate.get("/v2/guardrails/list")["guardrails"]
                    assert isinstance(listed, list), listed
                    guardrail_ids: Final = {
                        object_value(row)["guardrail_name"]: str(object_value(row)["guardrail_id"])
                        for row in listed
                        if object_value(row)["guardrail_name"] in (input_identity, output_identity)
                    }
                    assert set(guardrail_ids) == {input_identity, output_identity}, listed
                    today: Final = datetime.now(timezone.utc).date().isoformat()
                    for identity in (input_identity, output_identity):
                        detail: Final = eventually(
                            lambda identity=identity: candidate.request(
                                "GET",
                                f"/guardrails/usage/detail/{guardrail_ids[identity]}",
                                params={"start_date": today, "end_date": today},
                            ).json(),
                            lambda body: body["requestsEvaluated"] >= 1,
                            seconds=30,
                            return_last_on_timeout=True,
                        )
                        assert detail["requestsEvaluated"] == len(cases), detail
    finally:
        delete_scenario(upstream_handle)


@pytest.mark.parametrize(
    ("row_id", "endpoint", "scope", "expected_direction"),
    (
        pytest.param("C1", "chat", "output", "response", id="C1-chat-cache-output"),
        pytest.param("C2", "chat", "input", "request", id="C2-chat-cache-input"),
        pytest.param("C3", "messages", "output", "response", id="C3-messages-cache-output"),
        pytest.param("C4", "responses", "output", "response", id="C4-responses-cache-output"),
    ),
)
def test_cache_hit_directional_scope_uses_measured_base_default(
    gateway: Gateway,
    tmp_path: Path,
    row_id: str,
    endpoint: Endpoint,
    scope: str,
    expected_direction: Direction,
) -> None:
    identity: Final = f"logging-scope-{row_id.lower()}-{uuid.uuid4().hex}"
    scenario_id: Final = f"phase12-{row_id.lower()}-{uuid.uuid4().hex}"
    baseline_prompt: Final = f"uncached control {identity}"
    cached_prompt: Final = f"repeated cache prompt {identity}"
    reply: Final = f"cache response {identity}"
    base_default_on_hit: Final = BASE_DEFAULT_CACHE_HIT_DIRECTIONS[endpoint]
    expected_miss_directions: Final = _directions_for_audit_leg(
        BASE_DEFAULT_NORMAL_DIRECTIONS[(endpoint, False)], scope
    )
    expected_hit_directions: Final = _directions_for_audit_leg(base_default_on_hit, scope)
    if _is_base_audit_leg():
        assert expected_hit_directions == base_default_on_hit, (base_default_on_hit, expected_hit_directions)
    upstream_handle: Final = register_scenario(scenario_id, _provider_response(endpoint, scenario_id, reply, False))

    def policy(_request: Request) -> Reply:
        return Reply(body=b'{"action":"NONE"}')

    try:
        with gateway.scenario() as scenario:
            api_base: Final = (
                upstream_handle.api_base() if endpoint == "messages" else f"{upstream_handle.api_base()}/v1"
            )
            model: Final = scenario.model(
                model={
                    "chat": "openai/gpt-4o-mini",
                    "messages": "anthropic/claude-3-7-sonnet-20250219",
                    "responses": "openai/gpt-4.1-mini",
                }[endpoint],
                api_base=api_base,
                api_key="synthetic-provider-key",
            )
            with wire_server(policy) as guardrail:
                config: Final = _configuration(tmp_path, identity, guardrail.url, scope, cache=True)
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    baseline: Final = _call_cache_client(
                        endpoint, gateway, model, baseline_prompt, f"{scenario_id}-baseline"
                    )
                    assert baseline.status == 200, baseline.body
                    assert baseline.text == reply, baseline
                    assert len(_drain_upstream(gateway.upstream_url)) == 1
                    first: Final = _call_cache_client(endpoint, candidate, model, cached_prompt, f"{scenario_id}-first")
                    assert (first.status, _response_body_without_ids(first.body)) == (
                        baseline.status,
                        _response_body_without_ids(baseline.body),
                    ), (first, baseline)
                    assert first.text == reply, first
                    first_upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(first_upstream) == 1, first_upstream
                    assert cached_prompt in json.dumps(first_upstream[0]["body"]), first_upstream
                    eventually(
                        lambda: guardrail.received.qsize(),
                        lambda count: count >= len(expected_miss_directions),
                        seconds=20,
                    )
                    miss_payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                    assert all(_policy_call_id_matches(payload, f"{scenario_id}-first") for payload in miss_payloads), (
                        miss_payloads
                    )
                    second: Final = _call_cache_client(
                        endpoint, candidate, model, cached_prompt, f"{scenario_id}-second"
                    )
                    assert (second.status, _response_body_without_ids(second.body)) == (
                        first.status,
                        _response_body_without_ids(first.body),
                    ), (second, first)
                    assert _drain_upstream(gateway.upstream_url) == (), "The identical second request must hit Redis"
                    rows: Final = _spend_rows(model, 3)
                    eventually(
                        lambda: guardrail.received.qsize(),
                        lambda count: count >= len(expected_hit_directions),
                        seconds=20,
                    )
                    hit_payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                    for response_id, policy_payloads, expected_directions in (
                        (first.response_id, miss_payloads, expected_miss_directions),
                        (second.response_id, hit_payloads, expected_hit_directions),
                    ):
                        assert tuple(sorted(_direction(payload) for payload in policy_payloads)) == tuple(
                            sorted(expected_directions)
                        ), (response_id, policy_payloads)
                        assert all(
                            payload["texts"] == ([cached_prompt] if _direction(payload) == "request" else [reply])
                            for payload in policy_payloads
                        ), (response_id, policy_payloads)
                    assert sum(_cache_hit(row["cache_hit"]) for row in rows) == 1, rows
                    guarded_rows: Final = tuple(
                        row for row in rows if identity in object_value(row["metadata"]).get("applied_guardrails", [])
                    )
                    assert len(guarded_rows) == 2, rows
                    miss_rows: Final = tuple(row for row in guarded_rows if not _cache_hit(row["cache_hit"]))
                    hit_rows: Final = tuple(row for row in guarded_rows if _cache_hit(row["cache_hit"]))
                    assert len(miss_rows) == len(hit_rows) == 1, rows
                    _assert_response_id(endpoint, str(miss_rows[0]["request_id"]), first.response_id, scenario_id)
                    assert str(hit_rows[0]["request_id"]).startswith(f"{second.response_id}_cache_hit"), hit_rows
                    for row, expected_directions in (
                        (miss_rows[0], expected_miss_directions),
                        (hit_rows[0], expected_hit_directions),
                    ):
                        entries: Final = _guardrail_entries(row)
                        assert len(entries) == len(expected_directions), entries
                        assert tuple(
                            (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                            for entry in entries
                        ) == tuple((identity, "logging_only", "success") for _ in expected_directions), entries
    finally:
        delete_scenario(upstream_handle)


@pytest.mark.parametrize(
    ("row_id", "scope", "modes", "expected_directions", "expected_statuses", "expected_status"),
    (
        pytest.param(
            "D1",
            "output",
            ["pre_call", "logging_only"],
            ("request",),
            ("guardrail_intervened",),
            400,
            id="D1-pre-call-blocks-before-upstream",
        ),
        pytest.param(
            "D2",
            "output",
            ["pre_call", "logging_only"],
            ("request", "response"),
            ("success", "guardrail_intervened"),
            200,
            id="D2-pre-call-and-output-observation",
        ),
        pytest.param(
            "D3",
            "input",
            ["logging_only", "post_call"],
            ("response",),
            ("guardrail_intervened",),
            400,
            id="D3-post-call-block-remains-enforced",
        ),
    ),
)
def test_combined_modes_preserve_blocking_and_directional_observation(
    gateway: Gateway,
    tmp_path: Path,
    row_id: str,
    scope: str,
    modes: list[str],
    expected_directions: tuple[Direction, ...],
    expected_statuses: tuple[str, ...],
    expected_status: int,
) -> None:
    identity: Final = f"logging-scope-{row_id.lower()}-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic combined-mode prompt {identity}"
    reply: Final = f"synthetic combined-mode response {identity}"
    scenario_id: Final = f"phase12-{row_id.lower()}-{uuid.uuid4().hex}"
    call_id: Final = f"{scenario_id}-guarded"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def policy(request: Request) -> Reply:
        payload: Final = JSON_OBJECT.validate_json(request.body)
        direction: Final = _direction(payload)
        blocked: Final = row_id == "D1" or direction == "response"
        verdict: Final = (
            {"action": "BLOCKED", "blocked_reason": f"synthetic denial from {identity}"}
            if blocked
            else {"action": "NONE"}
        )
        return Reply(body=json.dumps(verdict).encode())

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            with wire_server(policy) as guardrail:
                config: Final = _configuration(tmp_path, identity, guardrail.url, scope, mode=modes)
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    baseline: Final = gateway.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": prompt}]},
                        headers={"x-litellm-call-id": f"{scenario_id}-baseline"},
                    )
                    assert baseline.status_code == 200, baseline.text
                    assert baseline.json()["choices"][0]["message"]["content"] == reply, baseline.text
                    assert len(_drain_upstream(gateway.upstream_url)) == 1
                    guarded: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": prompt}]},
                        headers={"x-litellm-call-id": call_id},
                    )
                    leg_expected_directions: Final = (
                        ("request", "request", "response")
                        if _is_base_audit_leg() and row_id == "D2"
                        else expected_directions
                    )
                    leg_expected_statuses: Final = (
                        ("success", "success", "guardrail_intervened")
                        if _is_base_audit_leg() and row_id == "D2"
                        else expected_statuses
                    )
                    assert guarded.status_code == expected_status, guarded.text
                    candidate_upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(candidate_upstream) == (0 if row_id == "D1" else 1), candidate_upstream
                    if candidate_upstream:
                        assert prompt in json.dumps(candidate_upstream[0]["body"]), candidate_upstream
                    if row_id == "D2":
                        assert _response_body_without_ids(JSON_OBJECT.validate_python(guarded.json())) == (
                            _response_body_without_ids(JSON_OBJECT.validate_python(baseline.json()))
                        ), (guarded.text, baseline.text)
                    else:
                        assert identity in guarded.text, guarded.text
                        assert f"synthetic denial from {identity}" in guarded.text, guarded.text
                    expected_policy_call_count: Final = len(leg_expected_directions)
                    eventually(
                        lambda: guardrail.received.qsize(),
                        lambda count: count >= expected_policy_call_count,
                        seconds=20,
                    )
                    calls: Final = guardrail.drain()
                    payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in calls)
                    assert all(_policy_call_id_matches(payload, call_id) for payload in payloads), payloads
                    observed_directions: Final = tuple(_direction(payload) for payload in payloads)
                    assert tuple(payload["texts"] for payload in payloads) == tuple(
                        [prompt] if direction == "request" else [reply] for direction in observed_directions
                    ), payloads
                    rows: Final = _spend_rows(model, 2)
                    guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                    assert len(guarded_rows) == 1, rows
                    entries: Final = _guardrail_entries(guarded_rows[0])
                    observed_modes: Final = _guardrail_mode_status_pairs(entries)
                    assert all(mode_values == tuple(modes) for mode_values, _ in observed_modes), entries
                    observed_statuses: Final = tuple(status for _, status in observed_modes)
                    assert (
                        tuple(sorted(observed_directions)),
                        tuple(sorted(observed_statuses)),
                    ) == (tuple(sorted(leg_expected_directions)), tuple(sorted(leg_expected_statuses))), (
                        payloads,
                        entries,
                        rows,
                    )
                    if row_id == "D2":
                        _assert_response_id(
                            "chat",
                            str(guarded_rows[0]["request_id"]),
                            str(guarded.json()["id"]),
                            scenario_id,
                        )
                    else:
                        assert _guardrail_entries(_spend_row_for_call_id(call_id)) == entries, entries
    finally:
        delete_scenario(upstream_handle)


@pytest.mark.parametrize(
    ("row_id", "scope", "selection"),
    (
        pytest.param("E1", "output", "request", id="E1-selected-by-request-body"),
        pytest.param("E2", "input", "virtual-key", id="E2-selected-by-key-metadata"),
        pytest.param("E3", "output", "unselected", id="E3-no-request-or-key-selection"),
    ),
)
def test_logging_only_scope_respects_guardrail_selection_level(
    gateway: Gateway, tmp_path: Path, row_id: str, scope: str, selection: str
) -> None:
    identity: Final = f"logging-scope-{row_id.lower()}-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic selected request {identity}"
    reply: Final = f"synthetic selected response {identity}"
    scenario_id: Final = f"phase12-{row_id.lower()}-{uuid.uuid4().hex}"
    call_id: Final = f"{scenario_id}-guarded"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def policy(_request: Request) -> Reply:
        return Reply(body=b'{"action":"NONE"}')

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            key: Final = (
                scenario.key(metadata={"guardrails": [identity]}) if selection == "virtual-key" else gateway.key
            )
            with wire_server(policy) as guardrail:
                config: Final = _configuration(tmp_path, identity, guardrail.url, scope, default_on=False)
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    request_body: Final[dict[str, JsonValue]] = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        **({"guardrails": [identity]} if selection == "request" else {}),
                    }
                    baseline: Final = gateway.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": prompt}]},
                        key=key,
                        headers={"x-litellm-call-id": f"{scenario_id}-baseline"},
                    )
                    assert baseline.status_code == 200, baseline.text
                    assert baseline.json()["choices"][0]["message"]["content"] == reply, baseline.text
                    assert len(_drain_upstream(gateway.upstream_url)) == 1
                    guarded: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        request_body,
                        key=key,
                        headers={"x-litellm-call-id": call_id},
                    )
                    assert guarded.status_code == 200, guarded.text
                    assert _response_body_without_ids(JSON_OBJECT.validate_python(guarded.json())) == (
                        _response_body_without_ids(JSON_OBJECT.validate_python(baseline.json()))
                    ), (guarded.text, baseline.text)
                    observed_upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(observed_upstream) == 1, observed_upstream
                    assert prompt in json.dumps(observed_upstream[0]["body"]), observed_upstream
                    expected_directions: Final = _directions_for_audit_leg(("request", "response"), scope)
                    directions_to_collect: Final = expected_directions
                    eventually(
                        lambda: guardrail.received.qsize(),
                        lambda count: count >= len(directions_to_collect),
                        seconds=20,
                    )
                    payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                    if expected_directions:
                        assert all(_policy_call_id_matches(payload, call_id) for payload in payloads), payloads
                    assert tuple(sorted(_direction(payload) for payload in payloads)) == tuple(
                        sorted(expected_directions)
                    ), payloads
                    if expected_directions:
                        rows: Final = _spend_rows(model, 2)
                        guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                        assert len(guarded_rows) == 1, rows
                        row: Final = guarded_rows[0]
                        _assert_response_id("chat", str(row["request_id"]), str(guarded.json()["id"]), scenario_id)
                        entries: Final = _guardrail_entries(row)
                        assert tuple(
                            (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                            for entry in entries
                        ) == tuple((identity, "logging_only", "success") for _ in expected_directions), entries
                    else:
                        row: Final = _spend_row_for_response_id(str(guarded.json()["id"]))
                        assert all(entry["guardrail_name"] != identity for entry in _guardrail_entries(row)), row
    finally:
        delete_scenario(upstream_handle)


def test_X1_missing_null_and_both_scope_have_identical_scans(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-x1-{uuid.uuid4().hex}"
    variants: Final = (
        ("both", "both", True),
        ("missing", None, False),
        ("null", None, True),
    )

    def policy(_request: Request) -> Reply:
        return Reply(body=b'{"action":"NONE"}')

    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        baseline: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": identity}]},
            headers={"x-litellm-call-id": f"{identity}-baseline"},
        )
        assert baseline.status_code == 200, baseline.text
        baseline_body: Final = JSON_OBJECT.validate_json(baseline.content)
        baseline_reply: Final = _response_text("chat", baseline_body)
        assert len(_drain_upstream(gateway.upstream_url)) == 1
        for suffix, scope, include_scope in variants:
            name: Final = f"{identity}-{suffix}"
            call_id: Final = f"{identity}-{suffix}"
            with wire_server(policy) as edge:
                config: Final = _configuration(
                    tmp_path,
                    name,
                    edge.url,
                    scope,
                    include_scope=include_scope,
                    default_on=False,
                )
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    response: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": model,
                            "messages": [{"role": "user", "content": identity}],
                            "guardrails": [name],
                        },
                        headers={"x-litellm-call-id": call_id},
                    )
                    assert response.status_code == baseline.status_code, response.text
                    body: Final = JSON_OBJECT.validate_json(response.content)
                    assert body["choices"] == baseline_body["choices"], response.text
                    assert _response_text("chat", body) == baseline_reply, response.text
                    upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(upstream) == 1 and identity in json.dumps(upstream[0]["body"]), upstream
                    expected_directions: Final = _directions_for_audit_leg(("request", "response"), scope)
                    eventually(
                        lambda: edge.received.qsize(),
                        lambda count, expected_directions=expected_directions: count == len(expected_directions),
                        seconds=20,
                    )
                    payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in edge.drain())
                    assert tuple(sorted(_direction(payload) for payload in payloads)) == tuple(
                        sorted(expected_directions)
                    ), payloads
                    assert all(_policy_call_id_matches(payload, call_id) for payload in payloads), payloads
                    assert tuple(sorted((_direction(payload), tuple(payload["texts"])) for payload in payloads)) == (
                        ("request", (identity,)),
                        ("response", (baseline_reply,)),
                    ), payloads
                    row: Final = _spend_row_for_response_id(str(body["id"]))
                    entries: Final = _guardrail_entries(row)
                    assert tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    ) == tuple((name, "logging_only", "success") for _ in expected_directions), entries


def test_X3_five_identical_requests_each_receive_one_response_scan(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-x3-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic repeated request {identity}"

    def policy(_request: Request) -> Reply:
        return Reply(body=b'{"action":"NONE"}')

    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        baseline: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}]},
            headers={"x-litellm-call-id": f"{identity}-baseline"},
        )
        assert baseline.status_code == 200, baseline.text
        assert len(_drain_upstream(gateway.upstream_url)) == 1
        with wire_server(policy) as edge:
            config: Final = _configuration(tmp_path, identity, edge.url, "output")
            with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                expected_directions: Final = _directions_for_audit_leg(("request", "response"), "output")
                results: Final = tuple(
                    candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": prompt}]},
                        headers={"x-litellm-call-id": f"{identity}-{index}"},
                    )
                    for index in range(5)
                )
                assert all(result.status_code == baseline.status_code for result in results), results
                assert all(result.json()["choices"] == baseline.json()["choices"] for result in results), results
                response_ids: Final = tuple(str(result.json()["id"]) for result in results)
                assert len(set(response_ids)) == 5, response_ids
                upstream: Final = _drain_upstream(gateway.upstream_url)
                assert len(upstream) == 5, upstream
                eventually(
                    lambda: edge.received.qsize(),
                    lambda count: count == 5 * len(expected_directions),
                    seconds=20,
                )
                calls: Final = edge.drain()
                payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in calls)
                for index, result in enumerate(results):
                    call_id: Final = f"{identity}-{index}"
                    matching_payloads: Final = tuple(
                        payload for payload in payloads if _policy_call_id_matches(payload, call_id)
                    )
                    assert tuple(sorted(_direction(payload) for payload in matching_payloads)) == tuple(
                        sorted(expected_directions)
                    ), (
                        call_id,
                        matching_payloads,
                    )
                    expected_reply: Final = _response_text("chat", JSON_OBJECT.validate_json(result.content))
                    assert all(
                        payload["texts"] == ([prompt] if _direction(payload) == "request" else [expected_reply])
                        for payload in matching_payloads
                    ), matching_payloads
                rows: Final = _spend_rows(model, 6)
                for index, result in enumerate(results):
                    matching_rows: Final = tuple(row for row in rows if row["request_id"] == result.json()["id"])
                    assert len(matching_rows) == 1, (result.json()["id"], matching_rows)
                    _assert_response_id(
                        "chat",
                        str(matching_rows[0]["request_id"]),
                        str(result.json()["id"]),
                        f"{identity}-{index}",
                    )
                    entries: Final = _guardrail_entries(matching_rows[0])
                    assert tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    ) == tuple((identity, "logging_only", "success") for _ in expected_directions), entries
                rows: Final = _spend_rows(model, 6)
                guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                assert len(guarded_rows) == 5, rows
                assert {str(row["request_id"]) for row in guarded_rows} == set(response_ids), guarded_rows
                assert all(
                    tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in _guardrail_entries(row)
                    )
                    == tuple((identity, "logging_only", "success") for _ in expected_directions)
                    for row in guarded_rows
                ), guarded_rows


def test_X2_scope_patch_toggles_during_concurrent_requests_keep_one_scan_per_response(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = f"logging-scope-x2-{uuid.uuid4().hex}"
    marker: Final = uuid.uuid4().hex
    scan_started: Final = threading.Event()
    release_scans: Final = threading.Event()

    def policy(_request: Request) -> Reply:
        scan_started.set()
        assert release_scans.wait(timeout=60), identity
        return Reply(body=b'{"action":"NONE"}')

    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        baseline: Final = _call_client(
            "openai_sync", "chat", gateway, model, f"synthetic X2 control {marker}", False, f"{marker}-control"
        )
        assert baseline.status == 200, baseline
        assert len(_drain_upstream(gateway.upstream_url)) == 1
        with wire_server(policy) as edge:
            config: Final = _empty_proxy_configuration(tmp_path, identity)
            with (
                _database_guardrail(identity, edge.url, "output", mode="logging_only"),
                owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate,
            ):
                guardrails: Final = candidate.get("/v2/guardrails/list")["guardrails"]
                guardrail_id: Final = next(
                    string_value(object_value(guardrail)["guardrail_id"])
                    for guardrail in guardrails
                    if object_value(guardrail)["guardrail_name"] == identity
                )
                calls: Final = tuple(
                    ChaosCall(
                        index=index,
                        endpoint="chat",
                        client_kind="openai_sync",
                        model=model,
                        stream=False,
                        prompt=f"synthetic X2 request {marker}-{index}",
                        call_id=f"{marker}-x2-{index}",
                    )
                    for index in range(20)
                )
                expected_scan_count: Final = len(calls) * (2 if _is_base_audit_leg() else 1)
                with ThreadPoolExecutor(max_workers=20) as pool:
                    futures: Final = tuple(
                        pool.submit(
                            _call_client,
                            call.client_kind,
                            call.endpoint,
                            candidate,
                            call.model,
                            call.prompt,
                            call.stream,
                            call.call_id,
                        )
                        for call in calls
                    )
                    try:
                        assert eventually(lambda: scan_started.is_set(), bool, seconds=30)
                        patch_responses: Final = tuple(
                            candidate.request(
                                "PATCH",
                                f"/guardrails/{guardrail_id}",
                                {"litellm_params": {"logging_only_scope": "input" if index % 2 == 0 else "output"}},
                            )
                            for index in range(10)
                        )
                        assert all(response.status_code == 200 for response in patch_responses), patch_responses
                    finally:
                        release_scans.set()
                    results: Final = tuple(future.result(timeout=90) for future in futures)
                assert all(result.status == baseline.status for result in results), results
                assert all(result.text == baseline.text for result in results), results
                response_ids: Final = tuple(result.response_id for result in results)
                assert len(set(response_ids)) == 20, response_ids
                eventually(
                    lambda: edge.received.qsize(),
                    lambda count: count == expected_scan_count,
                    seconds=30,
                )
                edge_calls: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in edge.drain())
                for call in calls:
                    payloads_for_call: Final = tuple(
                        payload for payload in edge_calls if _policy_call_id_matches(payload, call.call_id)
                    )
                    directions: Final = tuple(_direction(payload) for payload in payloads_for_call)
                    if _is_base_audit_leg():
                        assert tuple(sorted(directions)) == ("request", "response"), (call, payloads_for_call)
                    else:
                        assert len(directions) == 1 and directions[0] in ("request", "response"), (
                            call,
                            payloads_for_call,
                        )
                    assert all(
                        payload["texts"] == ([call.prompt] if _direction(payload) == "request" else [baseline.text])
                        for payload in payloads_for_call
                    ), payloads_for_call
                upstream: Final = _drain_upstream(gateway.upstream_url)
                assert len(upstream) == 20, upstream
                assert (
                    tuple(
                        sum(_json_contains_exact_string(observation["body"], call.prompt) for observation in upstream)
                        for call in calls
                    )
                    == (1,) * 20
                ), upstream
                rows: Final = _spend_rows(model, 21)
                for call, result in zip(calls, results):
                    row: Final = next(row for row in rows if row["request_id"] == result.response_id)
                    entries: Final = _guardrail_entries(row)
                    payloads_for_call: Final = tuple(
                        payload for payload in edge_calls if _policy_call_id_matches(payload, call.call_id)
                    )
                    assert tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    ) == tuple(
                        (identity, "logging_only", "success")
                        for _ in tuple(_direction(payload) for payload in payloads_for_call)
                    ), (call.call_id, entries)


def test_S1_logging_only_output_scope_fails_open_on_policy_500(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-s1-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic policy outage prompt {identity}"
    reply: Final = f"synthetic policy outage response {identity}"
    scenario_id: Final = f"phase12-s1-{uuid.uuid4().hex}"
    control_call_id: Final = f"{scenario_id}-control"
    guarded_call_id: Final = f"{scenario_id}-guarded"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(status=500, body=b'{"error":"synthetic policy outage"}')

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            control: Final = _call_client("openai_sync", "chat", gateway, model, prompt, False, control_call_id)
            assert control.status == 200 and control.text == reply, control
            assert len(_drain_upstream(gateway.upstream_url)) == 1
            with wire_server(policy) as guardrail:
                config: Final = _configuration(tmp_path, identity, guardrail.url, "output")
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    result: Final = _call_client(
                        "openai_sync", "chat", candidate, model, prompt, False, guarded_call_id
                    )
                    assert (result.status, _response_body_without_ids(result.body)) == (
                        control.status,
                        _response_body_without_ids(control.body),
                    ), (result, control)
                    expected_directions: Final = _directions_for_scope(("request", "response"), "output")
                    directions_to_collect: Final = _directions_for_audit_leg(("request", "response"), "output")
                    upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(upstream) == 1 and prompt in json.dumps(upstream[0]["body"]), upstream
                    eventually(
                        lambda: guardrail.received.qsize(),
                        lambda count: count >= len(directions_to_collect),
                        seconds=20,
                    )
                    calls: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                    assert all(_policy_call_id_matches(call, guarded_call_id) for call in calls), calls
                    assert tuple(call["texts"] for call in calls) == tuple(
                        [prompt] if _direction(call) == "request" else [reply] for call in calls
                    ), calls
                    guarded_row: Final = _spend_row_for_response_id(result.response_id)
                    _assert_response_id("chat", str(guarded_row["request_id"]), result.response_id, scenario_id)
                    entries: Final = _guardrail_entries(guarded_row)
                    observed_directions: Final = tuple(_direction(call) for call in calls)
                    observed_entries: Final = tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    )
                    expected_entries: Final = tuple(
                        (identity, "logging_only", "guardrail_failed_to_respond") for _ in expected_directions
                    )
                    assert (
                        tuple(sorted(observed_directions)),
                        tuple(sorted(observed_entries)),
                    ) == (tuple(sorted(expected_directions)), tuple(sorted(expected_entries))), (
                        calls,
                        entries,
                        guarded_row,
                    )
    finally:
        delete_scenario(upstream_handle)


def test_S2_logging_only_output_scope_scans_both_chat_choices(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-s2-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic multiple choice prompt {identity}"
    first_reply: Final = f"synthetic first choice {identity}"
    second_reply: Final = f"synthetic second choice {identity}"
    scenario_id: Final = f"phase12-s2-{uuid.uuid4().hex}"
    control_call_id: Final = f"{scenario_id}-control"
    guarded_call_id: Final = f"{scenario_id}-guarded"
    provider_response: Final = JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$UNIQUE_ID",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": first_reply}, "finish_reason": "stop"},
                {"index": 1, "message": {"role": "assistant", "content": second_reply}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19},
        },
    )
    upstream_handle: Final = register_scenario(scenario_id, provider_response)

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"BLOCKED","blocked_reason":"synthetic multiple choice monitor"}')

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            control: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": prompt}], "n": 2},
                headers={"x-litellm-call-id": control_call_id},
            )
            assert control.status_code == 200, control.text
            assert len(control.json()["choices"]) == 2, control.text
            assert len(_drain_upstream(gateway.upstream_url)) == 1
            with wire_server(policy) as guardrail:
                config: Final = _configuration(tmp_path, identity, guardrail.url, "output")
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    result: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": prompt}], "n": 2},
                        headers={"x-litellm-call-id": guarded_call_id},
                    )
                    assert result.status_code == control.status_code, result.text
                    assert _response_body_without_ids(JSON_OBJECT.validate_python(result.json())) == (
                        _response_body_without_ids(JSON_OBJECT.validate_python(control.json()))
                    ), (result.text, control.text)
                    upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(upstream) == 1 and prompt in json.dumps(upstream[0]["body"]), upstream
                    expected_directions: Final = ("response",)
                    eventually(
                        lambda: guardrail.received.qsize(),
                        lambda count: count == len(expected_directions),
                        seconds=20,
                    )
                    calls: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                    assert tuple(sorted(_direction(call) for call in calls)) == tuple(sorted(expected_directions)), (
                        calls
                    )
                    assert all(_policy_call_id_matches(call, guarded_call_id) for call in calls), calls
                    assert tuple(sorted((_direction(call), tuple(call["texts"])) for call in calls)) == tuple(
                        sorted(
                            (direction, tuple([prompt] if direction == "request" else [first_reply, second_reply]))
                            for direction in expected_directions
                        )
                    ), calls
                    rows: Final = _spend_rows(model, 2)
                    guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                    assert len(guarded_rows) == 1, rows
                    _assert_response_id(
                        "chat",
                        str(guarded_rows[0]["request_id"]),
                        str(result.json()["id"]),
                        scenario_id,
                    )
                    entries: Final = _guardrail_entries(guarded_rows[0])
                    assert tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    ) == tuple((identity, "logging_only", "guardrail_intervened") for _ in expected_directions), entries
    finally:
        delete_scenario(upstream_handle)


@pytest.mark.parametrize(
    ("endpoint", "scope"),
    (
        pytest.param("chat", "input", id="S3-chat-upstream-401-input"),
        pytest.param("chat", "output", id="S3-chat-upstream-401-output"),
        pytest.param("messages", "input", id="S3-messages-upstream-401-input"),
        pytest.param("messages", "output", id="S3-messages-upstream-401-output"),
        pytest.param("responses", "input", id="S3-responses-upstream-401-input"),
        pytest.param("responses", "output", id="S3-responses-upstream-401-output"),
    ),
)
def test_logging_only_scope_on_upstream_401_uses_measured_base_default(
    gateway: Gateway,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
    endpoint: Endpoint,
    scope: str,
) -> None:
    identity: Final = f"logging-scope-s3-{endpoint}-{scope}-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic upstream unauthorized marker {identity}"
    scenario_id: Final = f"phase12-s3-{endpoint}-{scope}-{uuid.uuid4().hex}"
    base_default_on_failure: Final = BASE_DEFAULT_UPSTREAM_FAILURE_DIRECTIONS[endpoint]
    expected_directions: Final = _directions_for_audit_leg(base_default_on_failure, scope)
    provider_response: Final = JsonResponse(
        content_type="application/json",
        body={
            "error": {
                "message": f"synthetic upstream unauthorized {identity}",
                "type": "invalid_request_error",
                "code": "401",
            }
        },
        status=401,
    )
    upstream_handle: Final = register_scenario(scenario_id, provider_response)

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"NONE"}')

    try:
        with gateway.scenario() as scenario:
            api_base: Final = (
                upstream_handle.api_base() if endpoint == "messages" else f"{upstream_handle.api_base()}/v1"
            )
            model: Final = scenario.model(
                model={
                    "chat": "openai/gpt-4o-mini",
                    "messages": "anthropic/claude-3-7-sonnet-20250219",
                    "responses": "openai/gpt-4.1-mini",
                }[endpoint],
                api_base=api_base,
                api_key="synthetic-provider-key",
            )
            path: Final = {
                "chat": "/v1/chat/completions",
                "messages": "/v1/messages",
                "responses": "/v1/responses",
            }[endpoint]
            request_body: Final = {
                "chat": {"model": model, "messages": [{"role": "user", "content": prompt}]},
                "messages": {
                    "model": model,
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}],
                },
                "responses": {"model": model, "input": prompt},
            }[endpoint]
            control_call_id: Final = f"{scenario_id}-control"
            candidate_call_id: Final = f"{scenario_id}-candidate"
            control: Final = gateway.client.request(
                "POST",
                path,
                json=request_body,
                headers={
                    "Authorization": f"Bearer {gateway.key}",
                    "x-litellm-call-id": control_call_id,
                },
                timeout=60,
            )
            control_upstream: Final = _drain_upstream(gateway.upstream_url)
            control_upstream_count: Final = len(control_upstream)
            record_property("s3_control_upstream_request_count", control_upstream_count)
            assert control_upstream_count >= 1, control_upstream
            assert control.status_code >= 400, control.text
            assert all(prompt in json.dumps(observation["body"]) for observation in control_upstream), control_upstream
            with wire_server(policy) as guardrail:
                config: Final = _configuration(tmp_path, identity, guardrail.url, scope)
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    result: Final = candidate.client.request(
                        "POST",
                        path,
                        json=request_body,
                        headers={
                            "Authorization": f"Bearer {candidate.key}",
                            "x-litellm-call-id": candidate_call_id,
                        },
                        timeout=60,
                    )
                    upstream: Final = _drain_upstream(gateway.upstream_url)
                    upstream_count: Final = len(upstream)
                    record_property("s3_candidate_upstream_request_count", upstream_count)
                    assert upstream_count >= 1, upstream
                    assert result.status_code >= 400, result.text
                    assert result.status_code == control.status_code, result.text
                    assert _response_body_without_ids(JSON_OBJECT.validate_python(result.json())) == (
                        _response_body_without_ids(JSON_OBJECT.validate_python(control.json()))
                    ), (result.text, control.text)
                    assert all(prompt in json.dumps(observation["body"]) for observation in upstream), upstream
                    calls: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                    assert all(_policy_call_id_matches(call, candidate_call_id) for call in calls), calls
                    assert all(call["texts"] == [prompt] for call in calls if _direction(call) == "request"), calls
                    spend_rows: Final = _spend_rows(model, 2)
                    assert all(not _guardrail_entries(row) for row in spend_rows), spend_rows
                    assert {str(row["request_id"]) for row in spend_rows} >= {
                        control_call_id,
                        candidate_call_id,
                    }, spend_rows
                    assert tuple(sorted(_direction(call) for call in calls)) == tuple(sorted(expected_directions)), (
                        calls,
                        spend_rows,
                    )
    finally:
        delete_scenario(upstream_handle)


def test_S4_logging_only_input_scope_scans_every_multipart_text_part(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-s4-{uuid.uuid4().hex}"
    first_part: Final = f"synthetic first text part {identity}"
    second_part: Final = f"synthetic second text part {identity}"
    reply: Final = f"synthetic multipart response {identity}"
    scenario_id: Final = f"phase12-s4-{uuid.uuid4().hex}"
    control_call_id: Final = f"{scenario_id}-control"
    guarded_call_id: Final = f"{scenario_id}-guarded"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"BLOCKED","blocked_reason":"synthetic multipart monitor"}')

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            request_body: Final = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": first_part},
                            {"type": "text", "text": second_part},
                        ],
                    }
                ],
            }
            control: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                request_body,
                headers={"x-litellm-call-id": control_call_id},
            )
            assert control.status_code == 200, control.text
            assert control.json()["choices"][0]["message"]["content"] == reply, control.text
            assert len(_drain_upstream(gateway.upstream_url)) == 1
            with wire_server(policy) as guardrail:
                config: Final = _configuration(tmp_path, identity, guardrail.url, "input")
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    result: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        request_body,
                        headers={"x-litellm-call-id": guarded_call_id},
                    )
                    assert result.status_code == control.status_code, result.text
                    assert _response_body_without_ids(JSON_OBJECT.validate_python(result.json())) == (
                        _response_body_without_ids(JSON_OBJECT.validate_python(control.json()))
                    ), (result.text, control.text)
                    observed_upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(observed_upstream) == 1, observed_upstream
                    assert first_part in json.dumps(observed_upstream[0]["body"]), observed_upstream
                    assert second_part in json.dumps(observed_upstream[0]["body"]), observed_upstream
                    expected_directions: Final = ("request",)
                    eventually(
                        lambda: guardrail.received.qsize(),
                        lambda count: count == len(expected_directions),
                        seconds=20,
                    )
                    calls: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                    assert tuple(sorted(_direction(call) for call in calls)) == tuple(sorted(expected_directions)), (
                        calls
                    )
                    assert all(_policy_call_id_matches(call, guarded_call_id) for call in calls), calls
                    assert tuple(sorted((_direction(call), tuple(call["texts"])) for call in calls)) == tuple(
                        sorted(
                            (direction, tuple([first_part, second_part] if direction == "request" else [reply]))
                            for direction in expected_directions
                        )
                    ), calls
                    rows: Final = _spend_rows(model, 2)
                    guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                    assert len(guarded_rows) == 1, rows
                    _assert_response_id(
                        "chat", str(guarded_rows[0]["request_id"]), str(result.json()["id"]), scenario_id
                    )
                    entries: Final = _guardrail_entries(guarded_rows[0])
                    assert tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    ) == tuple((identity, "logging_only", "guardrail_intervened") for _ in expected_directions), entries
    finally:
        delete_scenario(upstream_handle)
