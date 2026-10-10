from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs

import pytest
from _logging_only_scope_support import (
    JSON_OBJECT,
    Direction,
    _assert_response_id,
    _call_client,
    _configuration,
    _content_filter_configuration,
    _create_guardrail,
    _delete_database_guardrail,
    _direction,
    _directions_for_audit_leg,
    _drain_upstream,
    _empty_proxy_configuration,
    _guardrail_entries,
    _guardrail_mode_values,
    _insert_database_guardrail,
    _is_base_audit_leg,
    _management_guardrail_rows,
    _model_armor_configuration,
    _policy_call_id,
    _policy_call_id_matches,
    _post_guardrail_body,
    _presidio_configuration,
    _provider_response,
    _response_body_without_ids,
    _spend_row_for_call_id,
    _spend_rows,
    wire_server,
)
from _logging_only_scope_support import (
    _record_audit_properties as _record_audit_properties,
)
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows, write_rows
from integration._support.process import owned_proxy, owned_proxy_process
from integration._support.upstream import delete_scenario, register_scenario
from integration._support.wire import Reply, Request
from pydantic import JsonValue


def test_G7_database_invalid_scope_reads_preserve_pre_call_blocking(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-g7-{uuid.uuid4().hex}"
    blocked_word: Final = f"pineapple{uuid.uuid4().hex[:8]}"
    prompt: Final = f"synthetic request containing {blocked_word}"
    reply: Final = f"synthetic upstream response {identity}"
    scenario_id: Final = f"phase12-g7-{uuid.uuid4().hex}"
    guardrail_id: Final = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            write_rows(
                'INSERT INTO "LiteLLM_GuardrailsTable" '
                "(guardrail_id, guardrail_name, litellm_params, guardrail_info, updated_at) "
                "VALUES (%s, %s, %s::jsonb, %s::jsonb, NOW())",
                (
                    guardrail_id,
                    identity,
                    json.dumps(
                        {
                            "guardrail": "litellm_content_filter",
                            "mode": "pre_call",
                            "logging_only_scope": "Input",
                            "default_on": True,
                            "blocked_words": [{"keyword": blocked_word, "action": "BLOCK"}],
                        }
                    ),
                    "{}",
                ),
            )
            try:
                config: Final = _empty_proxy_configuration(tmp_path, identity)
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    listing_response: Final = candidate.request("GET", "/v2/guardrails/list")
                    assert listing_response.status_code == 200, listing_response.text
                    listing: Final = JSON_OBJECT.validate_json(listing_response.content)
                    rows: Final = tuple(object_value(row) for row in listing["guardrails"])
                    stored_guardrail: Final = next(row for row in rows if row.get("guardrail_id") == guardrail_id)
                    stored_params: Final = object_value(stored_guardrail["litellm_params"])
                    expected_scope: Final = "Input" if _is_base_audit_leg() else None
                    assert stored_params.get("logging_only_scope") == expected_scope, stored_guardrail
                    assert stored_params["mode"] == "pre_call", stored_guardrail
                    assert stored_params["guardrail"] == "litellm_content_filter", stored_guardrail

                    info_response: Final = candidate.request("GET", f"/guardrails/{guardrail_id}/info")
                    assert info_response.status_code == 200, info_response.text
                    info: Final = JSON_OBJECT.validate_json(info_response.content)
                    info_params: Final = object_value(info["litellm_params"])
                    assert info_params.get("logging_only_scope") == expected_scope, info

                    blocked_response: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": prompt}]},
                    )
                    assert blocked_response.status_code == 400, blocked_response.text
                    assert _drain_upstream(gateway.upstream_url) == ()
            finally:
                _delete_database_guardrail(identity)
    finally:
        delete_scenario(upstream_handle)


@pytest.mark.parametrize(
    ("row_id", "scope", "blocked_side"),
    (
        pytest.param("F1", "output", "response", id="F1-native-filter-response"),
        pytest.param("F2", "input", "request", id="F2-native-filter-request"),
    ),
)
def test_native_content_filter_scope_logs_without_blocking(
    gateway: Gateway, tmp_path: Path, row_id: str, scope: str, blocked_side: str
) -> None:
    identity: Final = f"logging-scope-{row_id.lower()}-{uuid.uuid4().hex}"
    blocked_word: Final = f"pineapple{uuid.uuid4().hex[:6]}"
    prompt: Final = (
        f"synthetic {blocked_word} request {identity}"
        if blocked_side == "request"
        else f"synthetic clean request {identity}"
    )
    reply: Final = (
        f"synthetic {blocked_word} response {identity}"
        if blocked_side == "response"
        else f"synthetic clean response {identity}"
    )
    scenario_id: Final = f"phase12-{row_id.lower()}-{uuid.uuid4().hex}"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            config: Final = _content_filter_configuration(tmp_path, identity, scope, blocked_word)
            with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                baseline: Final = _call_client(
                    "openai_sync", "chat", gateway, model, prompt, False, f"{scenario_id}-baseline"
                )
                assert baseline.status == 200 and baseline.text == reply, baseline
                assert len(_drain_upstream(gateway.upstream_url)) == 1
                guarded: Final = _call_client(
                    "openai_sync", "chat", candidate, model, prompt, False, f"{scenario_id}-guarded"
                )
                assert (guarded.status, _response_body_without_ids(guarded.body)) == (
                    baseline.status,
                    _response_body_without_ids(baseline.body),
                ), (guarded, baseline)
                assert guarded.text == reply, guarded
                observed_upstream: Final = _drain_upstream(gateway.upstream_url)
                assert len(observed_upstream) == 1, observed_upstream
                assert prompt in json.dumps(observed_upstream[0]["body"]), observed_upstream
                rows: Final = _spend_rows(model, 2)
                guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                assert len(guarded_rows) == 1, rows
                _assert_response_id("chat", str(guarded_rows[0]["request_id"]), guarded.response_id, scenario_id)
                entries: Final = _guardrail_entries(guarded_rows[0])
                expected_directions: Final = _directions_for_audit_leg(("request", "response"), scope)
                assert tuple(
                    (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"]) for entry in entries
                ) == tuple(
                    (
                        identity,
                        "logging_only",
                        "guardrail_intervened" if direction == blocked_side else "success",
                    )
                    for direction in expected_directions
                ), entries
                intervened_entries: Final = tuple(
                    entry for entry in entries if entry["guardrail_status"] == "guardrail_intervened"
                )
                assert len(intervened_entries) == 1, entries
                assert intervened_entries[0]["guardrail_response"] is not None, entries
                assert intervened_entries[0]["guardrail_response"] == "REDACTED_BY_LITELM", entries
    finally:
        delete_scenario(upstream_handle)


@pytest.mark.parametrize(
    ("row_id", "scope"),
    (
        pytest.param("F3", "output", id="F3-model-armor-response-only"),
        pytest.param("F4", "input", id="F4-model-armor-request-only"),
    ),
)
def test_model_armor_directional_scope_uses_real_service_account_oauth(
    gateway: Gateway,
    tmp_path: Path,
    row_id: str,
    scope: str,
) -> None:
    identity: Final = f"logging-scope-{row_id.lower()}-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic Model Armor prompt {identity}"
    reply: Final = f"synthetic Model Armor response {identity}"
    scenario_id: Final = f"phase12-{row_id.lower()}-{uuid.uuid4().hex}"
    expected_directions: Final = _directions_for_audit_leg(("request", "response"), scope)
    directions_to_collect: Final = _directions_for_audit_leg(("request", "response"), scope)
    expected_fields: Final = tuple(
        "userPromptData" if direction == "request" else "modelResponseData" for direction in expected_directions
    )
    fields_to_collect: Final = tuple(
        "userPromptData" if direction == "request" else "modelResponseData" for direction in directions_to_collect
    )
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def oauth(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/token", request
        form: Final = parse_qs(request.body.decode())
        assert form.get("grant_type") == ["urn:ietf:params:oauth:grant-type:jwt-bearer"], form
        assertion: Final = form.get("assertion")
        assert assertion is not None and len(assertion[0].split(".")) == 3, form
        return Reply(body=b'{"access_token":"synthetic-model-armor-token","expires_in":3600,"token_type":"Bearer"}')

    def model_armor(request: Request) -> Reply:
        assert request.method == "POST", request
        assert request.headers.get("authorization") == "Bearer synthetic-model-armor-token", request.headers
        payload: Final = JSON_OBJECT.validate_json(request.body)
        assert len(payload) == 1, payload
        field: Final = next(iter(payload))
        assert field in ("userPromptData", "modelResponseData"), payload
        expected_text: Final = prompt if field == "userPromptData" else reply
        assert payload[field] == {"text": expected_text}, payload
        return Reply(body=b'{"sanitizationResult":{"filterMatchState":"NO_MATCH_FOUND"}}')

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            baseline: Final = _call_client(
                "openai_sync", "chat", gateway, model, prompt, False, f"{scenario_id}-baseline"
            )
            assert baseline.status == 200 and baseline.text == reply, baseline
            assert len(_drain_upstream(gateway.upstream_url)) == 1
            with wire_server(oauth, policy_edge=False) as token_edge, wire_server(model_armor) as armor_edge:
                config: Final = _model_armor_configuration(
                    tmp_path, identity, armor_edge.url, token_edge.url + "/token", scope
                )
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    result: Final = _call_client(
                        "openai_sync", "chat", candidate, model, prompt, False, f"{scenario_id}-candidate"
                    )
                    assert (result.status, _response_body_without_ids(result.body)) == (
                        baseline.status,
                        _response_body_without_ids(baseline.body),
                    ), (result, baseline)
                    assert result.text == reply, result
                    upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(upstream) == 1 and prompt in json.dumps(upstream[0]["body"]), upstream
                    eventually(
                        lambda: armor_edge.received.qsize(),
                        lambda count: count >= len(fields_to_collect),
                        seconds=30,
                    )
                    eventually(
                        lambda: token_edge.received.qsize(),
                        lambda count: count >= 1,
                        seconds=30,
                    )
                    armor_calls: Final = armor_edge.drain()
                    assert len(armor_calls) == len(fields_to_collect), armor_calls
                    armor_payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in armor_calls)
                    observed_fields: Final = tuple(next(iter(payload)) for payload in armor_payloads)
                    assert tuple(sorted(observed_fields)) == tuple(sorted(fields_to_collect)), armor_calls
                    assert all(
                        call.target.endswith(
                            ":sanitizeUserPrompt" if field == "userPromptData" else ":sanitizeModelResponse"
                        )
                        for call, field in zip(armor_calls, observed_fields)
                    ), armor_calls
                    token_calls: Final = token_edge.drain()
                    assert token_calls and all(call.target == "/token" for call in token_calls), token_calls
                    rows: Final = _spend_rows(model, 2)
                    guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                    assert len(guarded_rows) == 1, rows
                    _assert_response_id("chat", str(guarded_rows[0]["request_id"]), result.response_id, scenario_id)
                    entries: Final = _guardrail_entries(guarded_rows[0])
                    observed_entries: Final = tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    )
                    expected_entries: Final = tuple((identity, "logging_only", "success") for _ in expected_directions)
                    assert (
                        tuple(sorted(observed_fields)),
                        tuple(sorted(observed_entries)),
                    ) == (tuple(sorted(expected_fields)), tuple(sorted(expected_entries))), (
                        armor_calls,
                        entries,
                        rows,
                    )
    finally:
        delete_scenario(upstream_handle)


@pytest.mark.parametrize(
    ("row_id", "scope"),
    (
        pytest.param("F5", "input", id="F5-presidio-input-scope-ignored"),
        pytest.param("F6", "both", id="F6-presidio-both-scope-ignored"),
    ),
)
def test_presidio_scope_matches_no_scope_behavior(gateway: Gateway, tmp_path: Path, row_id: str, scope: str) -> None:
    identity: Final = f"logging-scope-{row_id.lower()}-{uuid.uuid4().hex}"
    scenario_id: Final = f"phase12-{row_id.lower()}-{uuid.uuid4().hex}"
    person: Final = f"synthetic Person {identity}"
    prompt: Final = f"synthetic Presidio prompt {person}"
    reply: Final = f"synthetic Presidio response {person}"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def analyzer(response_seen: threading.Event) -> Callable[[Request], Reply]:
        def handle(request: Request) -> Reply:
            assert request.method == "POST" and request.target == "/analyze", request
            payload: Final = JSON_OBJECT.validate_json(request.body)
            text: Final = payload["text"]
            assert isinstance(text, str), payload
            if reply in text:
                response_seen.set()
            start: Final = text.index(person)
            return Reply(
                body=json.dumps(
                    [{"entity_type": "PERSON", "start": start, "end": start + len(person), "score": 0.99}]
                ).encode()
            )

        return handle

    def anonymizer(response_seen: threading.Event) -> Callable[[Request], Reply]:
        def handle(request: Request) -> Reply:
            assert request.method == "POST" and request.target == "/anonymize", request
            payload: Final = JSON_OBJECT.validate_json(request.body)
            text: Final = payload["text"]
            results: Final = payload["analyzer_results"]
            assert isinstance(text, str) and isinstance(results, list) and len(results) == 1, payload
            if reply in text:
                response_seen.set()
            return Reply(body=json.dumps({"text": text, "items": [{"entity_type": "PERSON"}]}).encode())

        return handle

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            no_scope_analyzer_response_seen: Final = threading.Event()
            no_scope_anonymizer_response_seen: Final = threading.Event()
            scoped_analyzer_response_seen: Final = threading.Event()
            scoped_anonymizer_response_seen: Final = threading.Event()
            with (
                wire_server(analyzer(no_scope_analyzer_response_seen)) as no_scope_analyzer,
                wire_server(anonymizer(no_scope_anonymizer_response_seen)) as no_scope_anonymizer,
                wire_server(analyzer(scoped_analyzer_response_seen)) as scoped_analyzer,
                wire_server(anonymizer(scoped_anonymizer_response_seen)) as scoped_anonymizer,
            ):
                no_scope_config: Final = _presidio_configuration(
                    tmp_path, identity, no_scope_analyzer.url, no_scope_anonymizer.url, None
                )
                scoped_config: Final = _presidio_configuration(
                    tmp_path, identity, scoped_analyzer.url, scoped_anonymizer.url, scope
                )
                with owned_proxy_process(gateway, tmp_path, {}, config=no_scope_config, workers=1) as no_scope_owned:
                    no_scope_proxy: Final = no_scope_owned.gateway
                    no_scope_guardrails: Final = no_scope_proxy.get("/v2/guardrails/list")["guardrails"]
                    assert identity in {object_value(item)["guardrail_name"] for item in no_scope_guardrails}, (
                        no_scope_guardrails
                    )
                    no_scope_result: Final = _call_client(
                        "openai_sync", "chat", no_scope_proxy, model, prompt, False, f"{scenario_id}-no-scope"
                    )
                    assert no_scope_result.status == 200 and no_scope_result.text == reply, no_scope_result
                    no_scope_upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(no_scope_upstream) == 1 and prompt in json.dumps(no_scope_upstream[0]["body"]), (
                        no_scope_upstream
                    )
                    eventually(
                        lambda: no_scope_analyzer_response_seen.is_set() and no_scope_anonymizer_response_seen.is_set(),
                        bool,
                        seconds=30,
                    )
                    no_scope_analyzer_calls: Final = no_scope_analyzer.drain()
                    no_scope_anonymizer_calls: Final = no_scope_anonymizer.drain()
                    no_scope_call_counts: Final = (
                        len(no_scope_analyzer_calls),
                        len(no_scope_anonymizer_calls),
                    )
                    assert no_scope_call_counts[0] == no_scope_call_counts[1] > 0, no_scope_call_counts
                with owned_proxy_process(gateway, tmp_path, {}, config=scoped_config, workers=1) as scoped_owned:
                    scoped_proxy: Final = scoped_owned.gateway
                    guardrails: Final = scoped_proxy.get("/v2/guardrails/list")["guardrails"]
                    assert identity in {object_value(item)["guardrail_name"] for item in guardrails}, guardrails
                    scoped_result: Final = _call_client(
                        "openai_sync", "chat", scoped_proxy, model, prompt, False, f"{scenario_id}-scoped"
                    )
                    assert (scoped_result.status, _response_body_without_ids(scoped_result.body)) == (
                        no_scope_result.status,
                        _response_body_without_ids(no_scope_result.body),
                    ), (scoped_result, no_scope_result)
                    scoped_upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(scoped_upstream) == 1 and prompt in json.dumps(scoped_upstream[0]["body"]), (
                        scoped_upstream
                    )
                    eventually(
                        lambda: scoped_analyzer_response_seen.is_set() and scoped_anonymizer_response_seen.is_set(),
                        bool,
                        seconds=30,
                    )
                    scoped_analyzer_calls: Final = scoped_analyzer.drain()
                    scoped_anonymizer_calls: Final = scoped_anonymizer.drain()
                    scoped_call_counts: Final = (
                        len(scoped_analyzer_calls),
                        len(scoped_anonymizer_calls),
                    )
                    assert scoped_call_counts == no_scope_call_counts, (
                        scoped_call_counts,
                        no_scope_call_counts,
                    )
                    assert tuple(sorted((call.method, call.target, call.body) for call in scoped_analyzer_calls)) == (
                        tuple(sorted((call.method, call.target, call.body) for call in no_scope_analyzer_calls))
                    ), (scoped_analyzer_calls, no_scope_analyzer_calls)
                    assert tuple(
                        sorted((call.method, call.target, call.body) for call in scoped_anonymizer_calls)
                    ) == tuple(sorted((call.method, call.target, call.body) for call in no_scope_anonymizer_calls)), (
                        scoped_anonymizer_calls,
                        no_scope_anonymizer_calls,
                    )
                    log_text: Final = scoped_owned.log.read_text()
                    if row_id == "F5" and not _is_base_audit_leg():
                        assert "whose logging_only hook scans on its own" in log_text, log_text
                    if row_id == "F6":
                        assert "whose logging_only hook scans on its own" not in log_text, log_text
    finally:
        delete_scenario(upstream_handle)


def test_F7_guardrail_ui_settings_classify_directional_scope_support(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _empty_proxy_configuration(tmp_path, f"logging-scope-f7-{uuid.uuid4().hex}")
    with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
        response: Final = candidate.get("/guardrails/ui/add_guardrail_settings")
    if _is_base_audit_leg():
        assert "providers_without_directional_logging_only_scope" not in response, response
        return
    unsupported: Final = response.get("providers_without_directional_logging_only_scope")
    assert unsupported is not None, response
    assert isinstance(unsupported, list), response
    assert set(unsupported) == {
        "lakera",
        "lakera_v2",
        "presidio",
        "tool_permission",
        "cisco_ai_defense",
        "xecguard",
        "repelloai",
        "noma",
        "mcp_jwt_signer",
        "microsoft_purview",
        "agent_365",
        "guardrails_ai",
        "mcp_security",
        "conduct",
        "javelin",
        "pillar",
        "lasso",
        "dynamoai",
        "pangea",
        "aporia",
        "aim",
        "ibm_guardrails",
        "semantic_guard",
        "cato_networks",
    }, response
    assert not {"generic_guardrail_api", "litellm_content_filter", "model_armor"}.intersection(unsupported), response


@pytest.mark.parametrize(
    ("row_id", "mode", "scope", "expected_status", "expected_directions", "expected_guardrail_mode"),
    (
        pytest.param("G1", "pre_call", "input", 400, ("request",), "pre_call", id="G1-yaml-blocking-valid-scope"),
        pytest.param("G2", "pre_call", "Input", 400, ("request",), "pre_call", id="G2-yaml-blocking-invalid-literal"),
        pytest.param(
            "G3",
            "logging_only",
            "sideways",
            200,
            ("request",),
            "logging_only",
            id="G3-yaml-logging-invalid-literal",
        ),
    ),
)
def test_yaml_scope_loading_keeps_guardrail_enforcement(
    gateway: Gateway,
    tmp_path: Path,
    row_id: str,
    mode: str,
    scope: str,
    expected_status: int,
    expected_directions: tuple[Direction, ...],
    expected_guardrail_mode: str,
) -> None:
    identity: Final = f"logging-scope-{row_id.lower()}-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic yaml blocking marker {identity}"
    reply: Final = f"synthetic yaml response {identity}"
    scenario_id: Final = f"phase12-{row_id.lower()}-{uuid.uuid4().hex}"
    call_id: Final = f"{scenario_id}-guarded"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"BLOCKED","blocked_reason":"synthetic yaml denial"}')

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            with wire_server(policy) as guardrail:
                config: Final = _configuration(tmp_path, identity, guardrail.url, scope, mode=mode)
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
                    assert guarded.status_code == expected_status, guarded.text
                    caller_body: Final = JSON_OBJECT.validate_python(guarded.json())
                    candidate_upstream: Final = _drain_upstream(gateway.upstream_url)
                    assert len(candidate_upstream) == (0 if expected_status == 400 else 1), candidate_upstream
                    if expected_status == 200:
                        assert _response_body_without_ids(caller_body) == _response_body_without_ids(
                            JSON_OBJECT.validate_python(baseline.json())
                        ), (guarded.text, baseline.text)
                        assert prompt in json.dumps(candidate_upstream[0]["body"]), candidate_upstream
                    else:
                        assert "synthetic yaml denial" in guarded.text, guarded.text
                    if expected_status == 200:
                        eventually(
                            lambda: guardrail.received.qsize(),
                            lambda count: count >= len(expected_directions),
                            seconds=30,
                        )
                    payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                    assert tuple(sorted(_direction(payload) for payload in payloads)) == tuple(
                        sorted(expected_directions)
                    ), payloads
                    assert all(_policy_call_id_matches(payload, call_id) for payload in payloads), payloads
                    assert all(
                        payload["texts"] == ([prompt] if _direction(payload) == "request" else [reply])
                        for payload in payloads
                    ), payloads
                    rows: Final = _spend_rows(model, 2)
                    guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                    assert len(guarded_rows) == 1, rows
                    entries: Final = _guardrail_entries(guarded_rows[0])
                    assert tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    ) == tuple(
                        (identity, expected_guardrail_mode, "guardrail_intervened") for _ in expected_directions
                    ), entries
                    if expected_status == 400:
                        blocked_row: Final = _spend_row_for_call_id(call_id)
                        assert _guardrail_entries(blocked_row) == entries, blocked_row
                    else:
                        _assert_response_id(
                            "chat",
                            str(guarded_rows[0]["request_id"]),
                            str(caller_body["id"]),
                            scenario_id,
                        )
    finally:
        delete_scenario(upstream_handle)


@pytest.mark.parametrize(
    ("row_id", "scope"),
    (
        pytest.param("G4", "output", id="G4-database-invalid-combination-before-boot"),
        pytest.param("G5", "sideways", id="G5-database-invalid-literal-before-boot"),
    ),
)
def test_database_guardrail_load_keeps_pre_call_blocking(
    gateway: Gateway, tmp_path: Path, row_id: str, scope: str
) -> None:
    identity: Final = f"logging-scope-{row_id.lower()}-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic DB blocked marker {identity}"
    reply: Final = f"synthetic DB response {identity}"
    scenario_id: Final = f"phase12-{row_id.lower()}-{uuid.uuid4().hex}"
    call_id: Final = f"{scenario_id}-guarded"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"BLOCKED","blocked_reason":"synthetic DB denial"}')

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            baseline: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": prompt}]},
                headers={"x-litellm-call-id": f"{scenario_id}-baseline"},
            )
            assert baseline.status_code == 200, baseline.text
            assert len(_drain_upstream(gateway.upstream_url)) == 1
            with wire_server(policy) as guardrail:
                _insert_database_guardrail(identity, guardrail.url, scope)
                try:
                    config: Final = _empty_proxy_configuration(tmp_path, identity)
                    with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                        loaded: Final = read_rows(
                            'SELECT guardrail_name, litellm_params FROM "LiteLLM_GuardrailsTable" '
                            "WHERE guardrail_name=%s",
                            (identity,),
                        )
                        assert len(loaded) == 1, loaded
                        guarded: Final = candidate.request(
                            "POST",
                            "/v1/chat/completions",
                            {"model": model, "messages": [{"role": "user", "content": prompt}]},
                            headers={"x-litellm-call-id": call_id},
                        )
                        assert guarded.status_code == 400 and "synthetic DB denial" in guarded.text, guarded.text
                        assert _drain_upstream(gateway.upstream_url) == ()
                        payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                        assert tuple(_direction(payload) for payload in payloads) == ("request",), payloads
                        assert _policy_call_id_matches(payloads[0], call_id), payloads
                        assert payloads[0]["texts"] == [prompt], payloads
                        spend_row: Final = _spend_row_for_call_id(call_id)
                        entries: Final = _guardrail_entries(spend_row)
                        assert tuple(
                            (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                            for entry in entries
                        ) == ((identity, "pre_call", "guardrail_intervened"),), entries
                finally:
                    _delete_database_guardrail(identity)
    finally:
        delete_scenario(upstream_handle)


def test_G6_database_guardrail_polling_normalizes_invalid_scope_without_reinitializing(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = f"logging-scope-g6-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic DB polling marker {identity}"
    reply: Final = f"synthetic DB polling response {identity}"
    scenario_id: Final = f"phase12-g6-{uuid.uuid4().hex}"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def denial(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"BLOCKED","blocked_reason":"synthetic polling denial"}')

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            baseline: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": prompt}]},
                headers={"x-litellm-call-id": f"{scenario_id}-baseline"},
            )
            assert baseline.status_code == 200, baseline.text
            assert len(_drain_upstream(gateway.upstream_url)) == 1
            with wire_server(denial) as original_policy, wire_server(denial) as updated_policy:
                _insert_database_guardrail(identity, original_policy.url, None)
                try:
                    config: Final = _empty_proxy_configuration(tmp_path, identity, reload_seconds=1)
                    with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                        first_call_id: Final = f"{scenario_id}-before-update"
                        first: Final = candidate.request(
                            "POST",
                            "/v1/chat/completions",
                            {"model": model, "messages": [{"role": "user", "content": prompt}]},
                            headers={"x-litellm-call-id": first_call_id},
                        )
                        assert first.status_code == 400 and "synthetic polling denial" in first.text, first.text
                        first_policy_calls: Final = original_policy.drain()
                        assert len(first_policy_calls) == 1, first_policy_calls
                        updated_params: Final = {
                            "guardrail": "generic_guardrail_api",
                            "mode": "pre_call",
                            "default_on": True,
                            "api_base": updated_policy.url,
                            "api_key": "synthetic-guardrail-key",
                            "extra_headers": ["x-litellm-call-id"],
                            "logging_only_scope": "sideways",
                        }
                        write_rows(
                            'UPDATE "LiteLLM_GuardrailsTable" SET litellm_params=%s::jsonb, updated_at=NOW() '
                            "WHERE guardrail_name=%s",
                            (json.dumps(updated_params), identity),
                        )

                        def probe_updated_policy() -> tuple[str, int]:
                            call_id: Final = f"{scenario_id}-poll-{uuid.uuid4().hex}"
                            response: Final = candidate.request(
                                "POST",
                                "/v1/chat/completions",
                                {"model": model, "messages": [{"role": "user", "content": prompt}]},
                                headers={"x-litellm-call-id": call_id},
                            )
                            assert response.status_code == 400 and "synthetic polling denial" in response.text, (
                                response.text
                            )
                            return call_id, updated_policy.received.qsize()

                        observed_call_id: Final = eventually(
                            probe_updated_policy,
                            lambda result: result[1] >= 1,
                            seconds=30,
                        )[0]
                        final_call_id: Final = f"{scenario_id}-after-sync"
                        final: Final = candidate.request(
                            "POST",
                            "/v1/chat/completions",
                            {"model": model, "messages": [{"role": "user", "content": prompt}]},
                            headers={"x-litellm-call-id": final_call_id},
                        )
                        assert final.status_code == 400 and "synthetic polling denial" in final.text, final.text
                        old_payloads: Final = tuple(
                            JSON_OBJECT.validate_json(call.body)
                            for call in (*first_policy_calls, *original_policy.drain())
                        )
                        new_payloads: Final = tuple(
                            JSON_OBJECT.validate_json(call.body) for call in updated_policy.drain()
                        )
                        assert all(payload["texts"] == [prompt] for payload in (*old_payloads, *new_payloads)), (
                            old_payloads,
                            new_payloads,
                        )
                        all_call_ids: Final = tuple(
                            str(_policy_call_id(payload)) for payload in (*old_payloads, *new_payloads)
                        )
                        assert len(all_call_ids) == len(set(all_call_ids)), all_call_ids
                        assert first_call_id in all_call_ids and observed_call_id in all_call_ids, all_call_ids
                        assert final_call_id in all_call_ids, all_call_ids
                        assert len(new_payloads) >= 2, new_payloads
                        for call_id in (first_call_id, observed_call_id, final_call_id):
                            spend_row: Final = _spend_row_for_call_id(call_id)
                            entries: Final = _guardrail_entries(spend_row)
                            assert tuple(
                                (
                                    entry["guardrail_name"],
                                    entry["guardrail_mode"],
                                    entry["guardrail_status"],
                                )
                                for entry in entries
                            ) == ((identity, "pre_call", "guardrail_intervened"),), (call_id, entries)
                        assert _drain_upstream(gateway.upstream_url) == ()
                finally:
                    _delete_database_guardrail(identity)
    finally:
        delete_scenario(upstream_handle)


@pytest.mark.parametrize(
    ("row_id", "provider", "mode", "scope", "expected_status", "expected_scope", "expected_message"),
    (
        pytest.param(
            "H1",
            "generic_guardrail_api",
            "pre_call",
            "input",
            400,
            None,
            "mode does not include logging_only",
            id="H1-post-rejects-scope-outside-logging-only",
        ),
        pytest.param(
            "H2",
            "generic_guardrail_api",
            "logging_only",
            "sideways",
            422,
            None,
            "logging_only_scope",
            id="H2-post-rejects-invalid-string",
        ),
        pytest.param(
            "H2",
            "generic_guardrail_api",
            "logging_only",
            5,
            422,
            None,
            "logging_only_scope",
            id="H2-post-rejects-invalid-number",
        ),
        pytest.param(
            "H2",
            "generic_guardrail_api",
            "logging_only",
            "",
            422,
            None,
            "logging_only_scope",
            id="H2-post-rejects-invalid-empty-string",
        ),
        pytest.param(
            "H2",
            "generic_guardrail_api",
            "logging_only",
            ["input"],
            422,
            None,
            "logging_only_scope",
            id="H2-post-rejects-invalid-list",
        ),
        pytest.param(
            "H2",
            "generic_guardrail_api",
            "logging_only",
            "x" * 5000,
            422,
            None,
            "logging_only_scope",
            id="H2-post-rejects-oversized-string",
        ),
        pytest.param(
            "H3",
            "generic_guardrail_api",
            "logging_only",
            None,
            200,
            None,
            "",
            id="H3-post-accepts-null-scope",
        ),
        pytest.param(
            "H4",
            "presidio",
            "logging_only",
            "input",
            400,
            None,
            "whose logging_only hook scans on its own",
            id="H4-post-rejects-presidio-input-scope",
        ),
        pytest.param(
            "H4",
            "presidio",
            "logging_only",
            "both",
            200,
            "both",
            "",
            id="H4-post-accepts-presidio-both-scope",
        ),
    ),
)
def test_management_post_validates_logging_only_scope(
    gateway: Gateway,
    tmp_path: Path,
    row_id: str,
    provider: str,
    mode: str | list[str],
    scope: JsonValue,
    expected_status: int,
    expected_scope: JsonValue,
    expected_message: str,
) -> None:
    identity: Final = f"logging-scope-{row_id.lower()}-{uuid.uuid4().hex}"
    api_base: Final = "http://127.0.0.1:9"
    config: Final = _empty_proxy_configuration(tmp_path, identity)
    try:
        with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
            before_rows: Final = _management_guardrail_rows(identity)
            before_list: Final = candidate.get("/v2/guardrails/list")["guardrails"]
            expected_leg_status: Final = 200 if _is_base_audit_leg() else expected_status
            expected_leg_scope: Final = scope if _is_base_audit_leg() else expected_scope
            response: Final = candidate.request(
                "POST",
                "/guardrails",
                _post_guardrail_body(identity, provider, mode, api_base, scope),
            )
            assert response.status_code == expected_leg_status, response.text
            if expected_message and not _is_base_audit_leg():
                assert expected_message in response.text, response.text
            if expected_leg_status == 200:
                body: Final = JSON_OBJECT.validate_json(response.content)
                params: Final = object_value(body["litellm_params"])
                assert params.get("logging_only_scope") == expected_leg_scope, body
                rows: Final = _management_guardrail_rows(identity)
                assert (
                    len(rows) == 1
                    and object_value(rows[0]["litellm_params"]).get("logging_only_scope") == expected_leg_scope
                ), rows
            else:
                if row_id == "H2" and not _is_base_audit_leg():
                    details: Final = object_value(JSON_OBJECT.validate_python(response.json())["detail"][0])
                    assert details["type"] == "literal_error", details
                    assert details["loc"] == [
                        "body",
                        "guardrail",
                        "litellm_params",
                        "logging_only_scope",
                    ], details
                assert _management_guardrail_rows(identity) == before_rows == ()
                after_list: Final = candidate.get("/v2/guardrails/list")["guardrails"]
                assert after_list == before_list, (before_list, after_list)
                assert isinstance(after_list, list), after_list
                assert identity not in {object_value(item)["guardrail_name"] for item in after_list}, after_list
    finally:
        _delete_database_guardrail(identity)


def test_H5_management_put_rejection_preserves_database_and_runtime(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-h5-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic management marker {identity}"
    reply: Final = f"synthetic management response {identity}"
    scenario_id: Final = f"phase12-h5-{uuid.uuid4().hex}"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"BLOCKED","blocked_reason":"synthetic management denial"}')

    try:
        with wire_server(policy) as guardrail:
            config: Final = _empty_proxy_configuration(tmp_path, identity)
            with (
                owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate,
                candidate.scenario() as scenario,
            ):
                model: Final = scenario.model(
                    model="openai/gpt-4o-mini",
                    api_base=f"{upstream_handle.api_base()}/v1",
                    api_key="synthetic-provider-key",
                )
                created: Final = _create_guardrail(
                    candidate,
                    identity,
                    {
                        "guardrail": "generic_guardrail_api",
                        "mode": "pre_call",
                        "default_on": True,
                        "api_base": guardrail.url,
                        "api_key": "synthetic-guardrail-key",
                    },
                )
                guardrail_id: Final = string_value(created["guardrail_id"])
                before_rows: Final = _management_guardrail_rows(identity)
                assert len(before_rows) == 1, before_rows
                before_info: Final = candidate.get(f"/guardrails/{guardrail_id}/info")
                before_list: Final = tuple(
                    object_value(item)
                    for item in candidate.get("/v2/guardrails/list")["guardrails"]
                    if object_value(item)["guardrail_id"] == guardrail_id
                )
                before_call_id: Final = f"{scenario_id}-before-put"
                before_response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": prompt}]},
                    headers={"x-litellm-call-id": before_call_id},
                )
                assert before_response.status_code == 400, before_response.text
                before_policy: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                assert len(before_policy) == 1 and _policy_call_id_matches(before_policy[0], before_call_id), (
                    before_policy
                )
                assert before_policy[0]["texts"] == [prompt], before_policy
                update: Final = candidate.request(
                    "PUT",
                    f"/guardrails/{guardrail_id}",
                    _post_guardrail_body(
                        identity,
                        "generic_guardrail_api",
                        "pre_call",
                        guardrail.url,
                        "output",
                    ),
                )
                if _is_base_audit_leg():
                    assert update.status_code == 200, update.text
                    updated_rows: Final = _management_guardrail_rows(identity)
                    assert len(updated_rows) == 1, updated_rows
                    assert object_value(updated_rows[0]["litellm_params"]).get("logging_only_scope") == "output", (
                        updated_rows
                    )
                    assert object_value(updated_rows[0]["litellm_params"])["mode"] == "pre_call", updated_rows
                else:
                    assert update.status_code == 422, update.text
                    assert "logging_only_scope" in update.text and "logging_only" in update.text, update.text
                    assert _management_guardrail_rows(identity) == before_rows
                    after_info: Final = candidate.get(f"/guardrails/{guardrail_id}/info")
                    assert {key: value for key, value in before_info.items() if key != "updated_at"} == {
                        key: value for key, value in after_info.items() if key != "updated_at"
                    }
                after_list: Final = tuple(
                    object_value(item)
                    for item in candidate.get("/v2/guardrails/list")["guardrails"]
                    if object_value(item)["guardrail_id"] == guardrail_id
                )
                if _is_base_audit_leg():
                    assert (
                        len(after_list) == 1
                        and object_value(after_list[0]["litellm_params"]).get("logging_only_scope") == "output"
                    ), after_list
                else:
                    assert tuple(
                        {key: value for key, value in item.items() if key != "updated_at"} for item in after_list
                    ) == tuple(
                        {key: value for key, value in item.items() if key != "updated_at"} for item in before_list
                    )
                after_call_id: Final = f"{scenario_id}-after-put"
                after_response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": prompt}]},
                    headers={"x-litellm-call-id": after_call_id},
                )
                assert after_response.status_code == before_response.status_code, after_response.text
                assert after_response.text == before_response.text, (after_response.text, before_response.text)
                after_policy: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                assert len(after_policy) == 1 and _policy_call_id_matches(after_policy[0], after_call_id), after_policy
                assert after_policy[0]["texts"] == [prompt], after_policy
                assert _drain_upstream(gateway.upstream_url) == ()
                for call_id in (before_call_id, after_call_id):
                    entries: Final = _guardrail_entries(_spend_row_for_call_id(call_id))
                    assert tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    ) == ((identity, "pre_call", "guardrail_intervened"),), (call_id, entries)
    finally:
        _delete_database_guardrail(identity)
        delete_scenario(upstream_handle)


def test_H6_management_patch_clears_scope_when_switching_to_blocking_only_mode(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = f"logging-scope-h6-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic mode patch marker {identity}"
    scenario_id: Final = f"phase12-h6-{uuid.uuid4().hex}"
    upstream_handle: Final = register_scenario(
        scenario_id, _provider_response("chat", scenario_id, f"response {identity}", False)
    )

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"BLOCKED","blocked_reason":"synthetic mode patch denial"}')

    try:
        with wire_server(policy) as guardrail:
            config: Final = _empty_proxy_configuration(tmp_path, identity)
            with (
                owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate,
                candidate.scenario() as scenario,
            ):
                model: Final = scenario.model(
                    model="openai/gpt-4o-mini",
                    api_base=f"{upstream_handle.api_base()}/v1",
                    api_key="synthetic-provider-key",
                )
                created: Final = _create_guardrail(
                    candidate,
                    identity,
                    {
                        "guardrail": "generic_guardrail_api",
                        "mode": ["pre_call", "logging_only"],
                        "default_on": True,
                        "api_base": guardrail.url,
                        "api_key": "synthetic-guardrail-key",
                        "logging_only_scope": "output",
                    },
                )
                guardrail_id: Final = string_value(created["guardrail_id"])
                before: Final = _management_guardrail_rows(identity)
                assert len(before) == 1, before
                patched: Final = candidate.request(
                    "PATCH",
                    f"/guardrails/{guardrail_id}",
                    {"litellm_params": {"mode": ["pre_call"]}},
                )
                assert patched.status_code == 200, patched.text
                persisted: Final = _management_guardrail_rows(identity)
                assert len(persisted) == 1, persisted
                params: Final = object_value(persisted[0]["litellm_params"])
                assert params.get("logging_only_scope") == ("output" if _is_base_audit_leg() else None), persisted
                assert params["mode"] == ["pre_call"], persisted
                call_id: Final = f"{scenario_id}-after-patch"
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": prompt}]},
                    headers={"x-litellm-call-id": call_id},
                )
                assert response.status_code == 400 and "synthetic mode patch denial" in response.text, response.text
                calls: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                assert tuple(_direction(call) for call in calls) == ("request",), calls
                assert _policy_call_id_matches(calls[0], call_id), calls
                assert calls[0]["texts"] == [prompt], calls
                assert _drain_upstream(gateway.upstream_url) == ()
                entries: Final = _guardrail_entries(_spend_row_for_call_id(call_id))
                assert tuple(
                    (
                        entry["guardrail_name"],
                        _guardrail_mode_values(entry["guardrail_mode"]),
                        entry["guardrail_status"],
                    )
                    for entry in entries
                ) == ((identity, ("pre_call",), "guardrail_intervened"),), entries
    finally:
        _delete_database_guardrail(identity)
        delete_scenario(upstream_handle)


def test_H7_management_patch_rejection_preserves_pre_call_guardrail(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-h7-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic invalid patch marker {identity}"
    scenario_id: Final = f"phase12-h7-{uuid.uuid4().hex}"
    upstream_handle: Final = register_scenario(
        scenario_id, _provider_response("chat", scenario_id, f"response {identity}", False)
    )

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"BLOCKED","blocked_reason":"synthetic invalid patch denial"}')

    try:
        with wire_server(policy) as guardrail:
            config: Final = _empty_proxy_configuration(tmp_path, identity)
            with (
                owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate,
                candidate.scenario() as scenario,
            ):
                model: Final = scenario.model(
                    model="openai/gpt-4o-mini",
                    api_base=f"{upstream_handle.api_base()}/v1",
                    api_key="synthetic-provider-key",
                )
                created: Final = _create_guardrail(
                    candidate,
                    identity,
                    {
                        "guardrail": "generic_guardrail_api",
                        "mode": "pre_call",
                        "default_on": True,
                        "api_base": guardrail.url,
                        "api_key": "synthetic-guardrail-key",
                    },
                )
                guardrail_id: Final = string_value(created["guardrail_id"])
                before: Final = _management_guardrail_rows(identity)
                before_call_id: Final = f"{scenario_id}-before-patch"
                response_before: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": prompt}]},
                    headers={"x-litellm-call-id": before_call_id},
                )
                assert response_before.status_code == 400, response_before.text
                policy_before: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                rejected: Final = candidate.request(
                    "PATCH",
                    f"/guardrails/{guardrail_id}",
                    {"litellm_params": {"logging_only_scope": "input"}},
                )
                if _is_base_audit_leg():
                    assert rejected.status_code == 200, rejected.text
                    updated: Final = _management_guardrail_rows(identity)
                    assert len(updated) == 1, updated
                    assert object_value(updated[0]["litellm_params"]).get("logging_only_scope") == "input", updated
                else:
                    assert rejected.status_code == 422, rejected.text
                    assert "logging_only_scope" in rejected.text and "logging_only" in rejected.text, rejected.text
                    assert _management_guardrail_rows(identity) == before
                after_call_id: Final = f"{scenario_id}-after-patch"
                response_after: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": prompt}]},
                    headers={"x-litellm-call-id": after_call_id},
                )
                assert response_after.status_code == response_before.status_code, response_after.text
                policy_after: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                assert tuple(_direction(call) for call in policy_before) == ("request",), policy_before
                assert tuple(_direction(call) for call in policy_after) == ("request",), policy_after
                assert _policy_call_id_matches(policy_before[0], before_call_id), policy_before
                assert _policy_call_id_matches(policy_after[0], after_call_id), policy_after
                assert policy_before[0]["texts"] == [prompt], policy_before
                assert policy_after[0]["texts"] == [prompt], policy_after
                assert _drain_upstream(gateway.upstream_url) == ()
                for call_id in (before_call_id, after_call_id):
                    entries: Final = _guardrail_entries(_spend_row_for_call_id(call_id))
                    assert tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    ) == ((identity, "pre_call", "guardrail_intervened"),), (call_id, entries)
    finally:
        _delete_database_guardrail(identity)
        delete_scenario(upstream_handle)


@pytest.mark.parametrize(
    ("row_id", "stored_scope"),
    (
        pytest.param("H8", "output", id="H8-patch-heals-invalid-mode-combination"),
        pytest.param("H9", "sideways", id="H9-patch-heals-invalid-stored-literal"),
    ),
)
def test_management_patch_default_on_heals_stored_scope(
    gateway: Gateway, tmp_path: Path, row_id: str, stored_scope: str
) -> None:
    identity: Final = f"logging-scope-{row_id.lower()}-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic healed scope marker {identity}"
    scenario_id: Final = f"phase12-{row_id.lower()}-{uuid.uuid4().hex}"
    upstream_handle: Final = register_scenario(
        scenario_id, _provider_response("chat", scenario_id, f"response {identity}", False)
    )

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"BLOCKED","blocked_reason":"synthetic healed scope denial"}')

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            baseline: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": prompt}]},
                headers={"x-litellm-call-id": f"{scenario_id}-baseline"},
            )
            assert baseline.status_code == 200, baseline.text
            assert len(_drain_upstream(gateway.upstream_url)) == 1
            with wire_server(policy) as guardrail:
                _insert_database_guardrail(
                    identity,
                    guardrail.url,
                    stored_scope,
                    default_on=False,
                )
                try:
                    config: Final = _empty_proxy_configuration(tmp_path, identity)
                    with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                        before: Final = _management_guardrail_rows(identity)
                        assert len(before) == 1, before
                        guardrail_id: Final = string_value(before[0]["guardrail_id"])
                        patched: Final = candidate.request(
                            "PATCH",
                            f"/guardrails/{guardrail_id}",
                            {"litellm_params": {"default_on": True}},
                        )
                        assert patched.status_code == 200, patched.text
                        body: Final = JSON_OBJECT.validate_json(patched.content)
                        params: Final = object_value(body["litellm_params"])
                        assert params["default_on"] is True, body
                        persisted: Final = _management_guardrail_rows(identity)
                        assert len(persisted) == 1, persisted
                        persisted_params: Final = object_value(persisted[0]["litellm_params"])
                        assert persisted_params.get("logging_only_scope") == (
                            stored_scope if _is_base_audit_leg() else None
                        ), persisted
                        assert persisted_params["default_on"] is True, persisted
                        call_id: Final = f"{scenario_id}-after-patch"
                        response: Final = candidate.request(
                            "POST",
                            "/v1/chat/completions",
                            {"model": model, "messages": [{"role": "user", "content": prompt}]},
                            headers={"x-litellm-call-id": call_id},
                        )
                        assert response.status_code == 400 and "synthetic healed scope denial" in response.text, (
                            response.text
                        )
                        calls: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                        assert tuple(_direction(call) for call in calls) == ("request",), calls
                        assert _policy_call_id_matches(calls[0], call_id), calls
                        assert calls[0]["texts"] == [prompt], calls
                        assert _drain_upstream(gateway.upstream_url) == ()
                        entries: Final = _guardrail_entries(_spend_row_for_call_id(call_id))
                        assert tuple(
                            (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                            for entry in entries
                        ) == ((identity, "pre_call", "guardrail_intervened"),), entries
                finally:
                    _delete_database_guardrail(identity)
    finally:
        delete_scenario(upstream_handle)


def test_H10_management_patch_null_scope_restores_both_directions(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-h10-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic reset-scope prompt {identity}"
    reply: Final = f"synthetic reset-scope response {identity}"
    scenario_id: Final = f"phase12-h10-{uuid.uuid4().hex}"
    control_call_id: Final = f"{scenario_id}-control"
    guarded_call_id: Final = f"{scenario_id}-guarded"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        payload: Final = JSON_OBJECT.validate_json(request.body)
        direction: Final = _direction(payload)
        verdict: Final = (
            {"action": "NONE"}
            if direction == "request"
            else {"action": "BLOCKED", "blocked_reason": "synthetic reset-scope monitor"}
        )
        return Reply(body=json.dumps(verdict).encode())

    try:
        with wire_server(policy) as guardrail:
            config: Final = _empty_proxy_configuration(tmp_path, identity)
            with (
                owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate,
                candidate.scenario() as scenario,
            ):
                model: Final = scenario.model(
                    model="openai/gpt-4o-mini",
                    api_base=f"{upstream_handle.api_base()}/v1",
                    api_key="synthetic-provider-key",
                )
                control: Final = _call_client("openai_sync", "chat", candidate, model, prompt, False, control_call_id)
                assert control.status == 200 and control.text == reply, control
                assert len(_drain_upstream(gateway.upstream_url)) == 1
                created: Final = _create_guardrail(
                    candidate,
                    identity,
                    {
                        "guardrail": "generic_guardrail_api",
                        "mode": "logging_only",
                        "default_on": True,
                        "api_base": guardrail.url,
                        "api_key": "synthetic-guardrail-key",
                        "logging_only_scope": "output",
                    },
                )
                guardrail_id: Final = string_value(created["guardrail_id"])
                patched: Final = candidate.request(
                    "PATCH",
                    f"/guardrails/{guardrail_id}",
                    {"litellm_params": {"logging_only_scope": None}},
                )
                assert patched.status_code == 200, patched.text
                persisted: Final = _management_guardrail_rows(identity)
                assert len(persisted) == 1, persisted
                params: Final = object_value(persisted[0]["litellm_params"])
                assert params.get("logging_only_scope") is None, persisted
                result: Final = _call_client("openai_sync", "chat", candidate, model, prompt, False, guarded_call_id)
                assert (result.status, _response_body_without_ids(result.body)) == (
                    control.status,
                    _response_body_without_ids(control.body),
                ), (result, control)
                upstream: Final = _drain_upstream(gateway.upstream_url)
                assert len(upstream) == 1, upstream
                assert prompt in json.dumps(upstream[0]["body"]), upstream
                expected_directions: Final = ("request", "response")
                eventually(
                    lambda: guardrail.received.qsize(),
                    lambda count: count == len(expected_directions),
                    seconds=20,
                )
                policy_calls: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                assert tuple(sorted(_direction(call) for call in policy_calls)) == tuple(sorted(expected_directions)), (
                    policy_calls
                )
                assert all(_policy_call_id_matches(call, guarded_call_id) for call in policy_calls), policy_calls
                assert all(
                    call["texts"] == ([prompt] if _direction(call) == "request" else [reply]) for call in policy_calls
                ), policy_calls
                rows: Final = _spend_rows(model, 2)
                guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                assert len(guarded_rows) == 1, rows
                _assert_response_id("chat", str(guarded_rows[0]["request_id"]), result.response_id, scenario_id)
                entries: Final = _guardrail_entries(guarded_rows[0])
                assert tuple(
                    (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"]) for entry in entries
                ) == (
                    (identity, "logging_only", "success"),
                    (identity, "logging_only", "guardrail_intervened"),
                ), entries
    finally:
        _delete_database_guardrail(identity)
        delete_scenario(upstream_handle)


def test_H11_management_patch_same_scope_is_idempotent_and_output_only(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-h11-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic idempotent prompt {identity}"
    reply: Final = f"synthetic idempotent response {identity}"
    scenario_id: Final = f"phase12-h11-{uuid.uuid4().hex}"
    control_call_id: Final = f"{scenario_id}-control"
    guarded_call_id: Final = f"{scenario_id}-guarded"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"BLOCKED","blocked_reason":"synthetic idempotent monitor"}')

    try:
        with wire_server(policy) as guardrail:
            config: Final = _empty_proxy_configuration(tmp_path, identity)
            with (
                owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate,
                candidate.scenario() as scenario,
            ):
                model: Final = scenario.model(
                    model="openai/gpt-4o-mini",
                    api_base=f"{upstream_handle.api_base()}/v1",
                    api_key="synthetic-provider-key",
                )
                control: Final = _call_client("openai_sync", "chat", candidate, model, prompt, False, control_call_id)
                assert control.status == 200 and control.text == reply, control
                assert len(_drain_upstream(gateway.upstream_url)) == 1
                created: Final = _create_guardrail(
                    candidate,
                    identity,
                    {
                        "guardrail": "generic_guardrail_api",
                        "mode": "logging_only",
                        "default_on": True,
                        "api_base": guardrail.url,
                        "api_key": "synthetic-guardrail-key",
                        "logging_only_scope": "output",
                    },
                )
                guardrail_id: Final = string_value(created["guardrail_id"])
                expected_directions: Final = _directions_for_audit_leg(("request", "response"), "output")
                before: Final = _management_guardrail_rows(identity)
                first: Final = candidate.request(
                    "PATCH",
                    f"/guardrails/{guardrail_id}",
                    {"litellm_params": {"logging_only_scope": "output"}},
                )
                second: Final = candidate.request(
                    "PATCH",
                    f"/guardrails/{guardrail_id}",
                    {"litellm_params": {"logging_only_scope": "output"}},
                )
                assert first.status_code == 200, first.text
                assert second.status_code == 200, second.text
                assert _management_guardrail_rows(identity) == before
                result: Final = _call_client("openai_sync", "chat", candidate, model, prompt, False, guarded_call_id)
                assert (result.status, _response_body_without_ids(result.body)) == (
                    control.status,
                    _response_body_without_ids(control.body),
                ), (result, control)
                upstream: Final = _drain_upstream(gateway.upstream_url)
                assert len(upstream) == 1 and prompt in json.dumps(upstream[0]["body"]), upstream
                eventually(
                    lambda: guardrail.received.qsize(),
                    lambda count: count == len(expected_directions),
                    seconds=20,
                )
                calls: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                assert tuple(sorted(_direction(call) for call in calls)) == tuple(sorted(expected_directions)), calls
                assert all(_policy_call_id_matches(call, guarded_call_id) for call in calls), calls
                assert all(
                    call["texts"] == ([prompt] if _direction(call) == "request" else [reply]) for call in calls
                ), calls
                rows: Final = _spend_rows(model, 2)
                guarded_rows: Final = tuple(row for row in rows if _guardrail_entries(row))
                assert len(guarded_rows) == 1, rows
                _assert_response_id("chat", str(guarded_rows[0]["request_id"]), result.response_id, scenario_id)
                entries: Final = _guardrail_entries(guarded_rows[0])
                assert tuple(
                    (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"]) for entry in entries
                ) == tuple((identity, "logging_only", "guardrail_intervened") for _ in expected_directions), entries
    finally:
        _delete_database_guardrail(identity)
        delete_scenario(upstream_handle)


def test_H12_management_reads_expose_typed_logging_only_scope(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-h12-{uuid.uuid4().hex}"
    try:
        with owned_proxy(
            gateway, tmp_path, {}, config=_empty_proxy_configuration(tmp_path, identity), workers=1
        ) as candidate:
            created: Final = _create_guardrail(
                candidate,
                identity,
                {
                    "guardrail": "generic_guardrail_api",
                    "mode": "logging_only",
                    "default_on": True,
                    "api_base": "http://127.0.0.1:9",
                    "api_key": "synthetic-guardrail-key",
                    "logging_only_scope": "input",
                },
            )
            guardrail_id: Final = string_value(created["guardrail_id"])
            info: Final = candidate.get(f"/guardrails/{guardrail_id}/info")
            listed: Final = candidate.get("/v2/guardrails/list")["guardrails"]
            assert isinstance(listed, list), listed
            matches: Final = tuple(
                object_value(item) for item in listed if object_value(item)["guardrail_id"] == guardrail_id
            )
            assert len(matches) == 1, listed
            assert object_value(info["litellm_params"])["logging_only_scope"] == "input", info
            assert object_value(matches[0]["litellm_params"])["logging_only_scope"] == "input", matches
    finally:
        _delete_database_guardrail(identity)


def test_H13_unauthenticated_management_and_chat_requests_do_not_scan(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-h13-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic unauthorized prompt {identity}"
    scenario_id: Final = f"phase12-h13-{uuid.uuid4().hex}"
    upstream_handle: Final = register_scenario(
        scenario_id, _provider_response("chat", scenario_id, f"response {identity}", False)
    )

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"NONE"}')

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            with wire_server(policy) as guardrail:
                config: Final = _configuration(tmp_path, identity, guardrail.url, "input")
                with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
                    unauthorized_post: Final = candidate.request(
                        "POST",
                        "/guardrails",
                        _post_guardrail_body(
                            f"{identity}-unauthorized",
                            "generic_guardrail_api",
                            "logging_only",
                            guardrail.url,
                            "input",
                        ),
                        key="synthetic-invalid-key",
                    )
                    assert unauthorized_post.status_code == 401, unauthorized_post.text
                    unauthorized_chat: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": prompt}]},
                        key="synthetic-invalid-key",
                        headers={"x-litellm-call-id": f"{scenario_id}-unauthorized"},
                    )
                    assert unauthorized_chat.status_code == 401, unauthorized_chat.text
                    assert _management_guardrail_rows(f"{identity}-unauthorized") == ()
                    assert guardrail.drain() == ()
                    assert _drain_upstream(gateway.upstream_url) == ()
    finally:
        delete_scenario(upstream_handle)


def test_K1_post_rejects_continue_flag_outside_logging_only(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-k1-{uuid.uuid4().hex}"
    config: Final = _empty_proxy_configuration(tmp_path, identity)
    try:
        with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
            before_list: Final = candidate.get("/v2/guardrails/list")["guardrails"]
            response: Final = candidate.request(
                "POST",
                "/guardrails",
                _post_guardrail_body(
                    identity,
                    "generic_guardrail_api",
                    "pre_call",
                    "http://127.0.0.1:9",
                    None,
                    include_scope=False,
                    continue_on_input_failure=True,
                ),
            )
            assert response.status_code == 400, response.text
            assert "logging_only_continue_on_input_failure" in response.text, response.text
            assert _management_guardrail_rows(identity) == ()
            after_list: Final = candidate.get("/v2/guardrails/list")["guardrails"]
            assert after_list == before_list, (before_list, after_list)
            assert identity not in {object_value(item)["guardrail_name"] for item in after_list}, after_list
    finally:
        _delete_database_guardrail(identity)


def test_K2_post_accepts_both_scope_and_stores_it_verbatim(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-k2-{uuid.uuid4().hex}"
    config: Final = _empty_proxy_configuration(tmp_path, identity)
    try:
        with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
            response: Final = candidate.request(
                "POST",
                "/guardrails",
                _post_guardrail_body(
                    identity,
                    "generic_guardrail_api",
                    "logging_only",
                    "http://127.0.0.1:9",
                    "both",
                ),
            )
            assert response.status_code == 200, response.text
            body: Final = JSON_OBJECT.validate_json(response.content)
            assert object_value(body["litellm_params"]).get("logging_only_scope") == "both", body
            rows: Final = _management_guardrail_rows(identity)
            assert len(rows) == 1, rows
            assert object_value(rows[0]["litellm_params"]).get("logging_only_scope") == "both", rows
    finally:
        _delete_database_guardrail(identity)


def test_K3_patch_flag_rejection_preserves_raw_stored_row(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-k3-{uuid.uuid4().hex}"
    config: Final = _empty_proxy_configuration(tmp_path, identity)
    try:
        with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
            created: Final = _create_guardrail(
                candidate,
                identity,
                {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_base": "http://127.0.0.1:9",
                    "api_key": "synthetic-guardrail-key",
                },
            )
            guardrail_id: Final = string_value(created["guardrail_id"])
            before_raw: Final = read_rows(
                'SELECT litellm_params::text AS raw_params FROM "LiteLLM_GuardrailsTable" WHERE guardrail_name=%s',
                (identity,),
            )
            assert len(before_raw) == 1, before_raw
            rejected: Final = candidate.request(
                "PATCH",
                f"/guardrails/{guardrail_id}",
                {"litellm_params": {"logging_only_continue_on_input_failure": True}},
            )
            assert rejected.status_code == 422, rejected.text
            assert "logging_only_continue_on_input_failure" in rejected.text, rejected.text
            after_raw: Final = read_rows(
                'SELECT litellm_params::text AS raw_params FROM "LiteLLM_GuardrailsTable" WHERE guardrail_name=%s',
                (identity,),
            )
            assert len(after_raw) == 1, after_raw
            # Rollback re-encrypts api_key, so ciphertext bytes differ even though the row is unchanged.
            before_params: Final = JSON_OBJECT.validate_json(before_raw[0]["raw_params"])
            after_params: Final = JSON_OBJECT.validate_json(after_raw[0]["raw_params"])
            assert set(before_params) == set(after_params), (before_params, after_params)
            assert {
                key: before_params[key] for key in before_params if key != "api_key"
            } == {key: after_params[key] for key in after_params if key != "api_key"}, (before_params, after_params)
            assert str(after_params["api_key"]).startswith("litellm_enc::"), after_params
    finally:
        _delete_database_guardrail(identity)


def test_K4_patch_dropping_logging_only_mode_clears_stored_flag(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-k4-{uuid.uuid4().hex}"
    config: Final = _empty_proxy_configuration(tmp_path, identity)
    try:
        with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
            created: Final = _create_guardrail(
                candidate,
                identity,
                {
                    "guardrail": "generic_guardrail_api",
                    "mode": ["pre_call", "logging_only"],
                    "default_on": True,
                    "api_base": "http://127.0.0.1:9",
                    "api_key": "synthetic-guardrail-key",
                    "logging_only_continue_on_input_failure": True,
                },
            )
            guardrail_id: Final = string_value(created["guardrail_id"])
            before: Final = _management_guardrail_rows(identity)
            assert len(before) == 1, before
            assert object_value(before[0]["litellm_params"]).get("logging_only_continue_on_input_failure") is True, (
                before
            )
            patched: Final = candidate.request(
                "PATCH",
                f"/guardrails/{guardrail_id}",
                {"litellm_params": {"mode": ["pre_call"]}},
            )
            assert patched.status_code == 200, patched.text
            persisted: Final = _management_guardrail_rows(identity)
            assert len(persisted) == 1, persisted
            params: Final = object_value(persisted[0]["litellm_params"])
            assert params.get("logging_only_continue_on_input_failure") is None, persisted
            assert params["mode"] == ["pre_call"], persisted
    finally:
        _delete_database_guardrail(identity)


def test_K5_yaml_flag_on_blocking_guardrail_boots_and_keeps_blocking(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-k5-{uuid.uuid4().hex}"
    prompt: Final = f"synthetic yaml flag marker {identity}"
    reply: Final = f"synthetic yaml flag response {identity}"
    scenario_id: Final = f"phase12-k5-{uuid.uuid4().hex}"
    upstream_handle: Final = register_scenario(scenario_id, _provider_response("chat", scenario_id, reply, False))

    def policy(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        return Reply(body=b'{"action":"BLOCKED","blocked_reason":"synthetic yaml flag denial"}')

    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=f"{upstream_handle.api_base()}/v1",
                api_key="synthetic-provider-key",
            )
            with wire_server(policy) as guardrail:
                config: Final = _configuration(
                    tmp_path,
                    identity,
                    guardrail.url,
                    None,
                    include_scope=False,
                    continue_on_input_failure=True,
                    mode="pre_call",
                )
                with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=1) as owned:
                    candidate: Final = owned.gateway
                    guarded: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": prompt}]},
                        headers={"x-litellm-call-id": f"{scenario_id}-guarded"},
                    )
                    assert guarded.status_code == 400 and "synthetic yaml flag denial" in guarded.text, guarded.text
                    assert _drain_upstream(gateway.upstream_url) == ()
                    payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in guardrail.drain())
                    assert tuple(_direction(payload) for payload in payloads) == ("request",), payloads
                    log_text: Final = owned.log.read_text()
                    assert "Ignoring logging_only_continue_on_input_failure" in log_text, log_text
    finally:
        _delete_database_guardrail(identity)
        delete_scenario(upstream_handle)
