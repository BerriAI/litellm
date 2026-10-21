import asyncio
import json
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import Gateway, eventually
from integration._support.wire import Request, wire_server
from integration.providers._mantle_gpt_prompt_cache_support import (
    ANTHROPIC_PATH,
    AZURE,
    BURST,
    CLAUDE,
    ENDPOINTS,
    EPHEMERAL,
    EXPLICIT,
    GPT,
    GPT_FLAGGED_ROW,
    GPT_BARE,
    GPT_REGION,
    GPT_UNFLAGGED_ROW,
    HOSTILE_OPTIONS,
    IMPLICIT,
    ITEMS,
    MALFORMED_POINTS,
    MAX_TOKENS,
    MIXED_POINTS,
    NO_CACHE,
    ODD_FLAGS,
    SYSTEM,
    SYSTEM_POINT,
    THIRD_PARTY,
    Endpoint,
    Outcome,
    assert_answered,
    assert_burst_landed,
    assert_marked_sdk_cell,
    assert_priced_row,
    assert_wire,
    async_sdk_call,
    body_of,
    breakpoint_count,
    burst,
    deployment,
    expected_wire,
    failing_peer,
    failure_row,
    fresh_marker,
    mantle_deployment,
    mantle_peer,
    marked_cell,
    observe,
    only_received,
    openai_shaped_peer,
    prompt_text,
    request_body,
    row_key,
    sdk_call,
    send,
    send_raw,
    settled,
    spend_rows,
    success_row,
    system_item,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(600)


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_configured_system_point_reaches_mantle_as_a_breakpoint_over_httpx(
    gateway: Gateway, endpoint: Endpoint, stream: bool
) -> None:
    marked_cell(gateway, endpoint, stream=stream)


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_the_sync_sdks_see_the_breakpoint_and_the_mapped_cache_usage(
    gateway: Gateway, endpoint: Endpoint, stream: bool
) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        outcome: Final = sdk_call(gateway, endpoint, name, prompt_text(marker), stream=stream)
        assert_marked_sdk_cell(wire, endpoint, name, marker, outcome, stream=stream)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_the_async_sdks_see_the_breakpoint_and_the_mapped_cache_usage(gateway: Gateway, endpoint: Endpoint) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        outcome: Final = asyncio.run(async_sdk_call(gateway, endpoint, name, prompt_text(marker)))
        assert_marked_sdk_cell(wire, endpoint, name, marker, outcome, stream=False)


@pytest.mark.parametrize("endpoint", ["responses", "chat"])
def test_b1_b2_a_pinned_explicit_mode_reaches_the_wire_with_the_breakpoint(
    gateway: Gateway, endpoint: Endpoint
) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire, prompt_cache_options=EXPLICIT)
        outcome, received = observe(gateway, wire, endpoint, request_body(endpoint, name, prompt_text(marker)))
        assert_answered(outcome, marker)
        assert_wire(
            received,
            expected_wire(GPT, prompt_text(marker), endpoint=endpoint, marked=True, options=EXPLICIT),
            streaming=False,
        )
        assert_priced_row(success_row(name, marker), GPT)


@pytest.mark.parametrize("endpoint", ["responses", "chat"])
def test_b3_b4_a_region_prefixed_deployment_reads_the_region_free_row(gateway: Gateway, endpoint: Endpoint) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire, GPT_REGION)
        outcome, received = observe(gateway, wire, endpoint, request_body(endpoint, name, prompt_text(marker)))
        assert_answered(outcome, marker)
        assert_wire(received, expected_wire(GPT, prompt_text(marker), endpoint=endpoint, marked=True), streaming=False)
        assert_priced_row(success_row(name, marker), GPT)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_b15_to_b17_a_bare_deployment_name_with_its_provider_reads_the_provider_keyed_row(
    gateway: Gateway, endpoint: Endpoint
) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire, GPT_BARE, custom_llm_provider="bedrock_mantle")
        outcome, received = observe(gateway, wire, endpoint, request_body(endpoint, name, prompt_text(marker)))
        assert_answered(outcome, marker)
        assert_wire(received, expected_wire(GPT, prompt_text(marker), endpoint=endpoint, marked=True), streaming=False)
        assert_priced_row(success_row(name, marker), GPT)


def assert_anthropic_marked_wire(received: Request, marker: str) -> None:
    body: Final = body_of(received)
    assert urlsplit(received.target).path == ANTHROPIC_PATH, received.target
    assert body["system"] == [{"type": "text", "text": SYSTEM, "cache_control": EPHEMERAL}], received.body
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": prompt_text(marker)}]}], (
        received.body
    )
    assert "prompt_cache_options" not in body, received.body
    assert "prompt_cache_breakpoint" not in received.body.decode(), received.body


@pytest.mark.parametrize("endpoint", ["messages", "chat"])
def test_b5_b6_claude_on_mantle_keeps_the_anthropic_dialect(gateway: Gateway, endpoint: Endpoint) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire, CLAUDE)
        outcome, received = observe(gateway, wire, endpoint, request_body(endpoint, name, prompt_text(marker)))
        assert_answered(outcome, marker)
        assert_anthropic_marked_wire(received, marker)
        success_row(name, marker, outcome.response_id)


def test_b7_a_deployment_flag_true_opts_an_unflagged_mantle_row_in(gateway: Gateway) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(
            gateway, scenario, wire, GPT_UNFLAGGED_ROW, model_info={"supports_prompt_cache_breakpoint": True}
        )
        outcome, received = observe(gateway, wire, "responses", request_body("responses", name, prompt_text(marker)))
        assert_answered(outcome, marker)
        assert_wire(
            received,
            expected_wire(GPT_UNFLAGGED_ROW, prompt_text(marker), endpoint="responses", marked=True),
            streaming=False,
        )
        success_row(name, marker)


def test_b8_a_deployment_flag_false_opts_a_flagged_mantle_row_out(gateway: Gateway) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(
            gateway, scenario, wire, GPT_FLAGGED_ROW, model_info={"supports_prompt_cache_breakpoint": False}
        )
        outcome, received = observe(gateway, wire, "responses", request_body("responses", name, prompt_text(marker)))
        assert_answered(outcome, marker)
        body: Final = body_of(received)
        assert breakpoint_count(body) == 0, received.body
        assert "prompt_cache_options" not in body, received.body
        assert "cache_control" not in received.body.decode(), received.body
        success_row(name, marker)


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
def test_b13_azure_openai_stays_ineligible_for_the_breakpoint_dialect(gateway: Gateway, endpoint: Endpoint) -> None:
    marker: Final = fresh_marker()
    with wire_server(openai_shaped_peer()) as wire, gateway.scenario() as scenario:
        name: Final = scenario.model(
            model=AZURE,
            api_base=wire.url,
            api_key="synthetic-azure-key",
            api_version="2025-04-01-preview",
            cache_control_injection_points=SYSTEM_POINT,
        )
        settled(gateway, name, wire)
        outcome, received = observe(gateway, wire, endpoint, request_body(endpoint, name, prompt_text(marker)))
        assert_answered(outcome, marker)
        assert urlsplit(received.target).path.startswith("/openai/"), received.target
        assert "prompt_cache_breakpoint" not in received.body.decode(), received.body
        assert "prompt_cache_options" not in body_of(received), received.body
        assert SYSTEM in received.body.decode(), received.body
        success_row(name, marker)


def test_b14_an_openai_entry_on_a_third_party_host_keeps_the_anthropic_dialect(gateway: Gateway) -> None:
    marker: Final = fresh_marker()
    with wire_server(openai_shaped_peer()) as wire, gateway.scenario() as scenario:
        name: Final = scenario.model(
            model=THIRD_PARTY,
            api_base=wire.url,
            api_key="synthetic-third-party-key",
            cache_control_injection_points=SYSTEM_POINT,
        )
        settled(gateway, name, wire)
        outcome, received = observe(gateway, wire, "chat", request_body("chat", name, prompt_text(marker)))
        assert_answered(outcome, marker)
        body: Final = body_of(received)
        assert urlsplit(received.target).path == "/chat/completions", received.target
        assert body["messages"] == [
            {"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL},
            {"role": "user", "content": prompt_text(marker)},
        ], received.body
        assert "prompt_cache_options" not in body, received.body
        success_row(name, marker)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_c_a_response_cache_hit_serves_the_marked_request_again(gateway: Gateway, endpoint: Endpoint) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        body: Final = request_body(endpoint, name, prompt_text(marker))
        first, received = observe(gateway, wire, endpoint, body)
        assert_answered(first, marker)
        assert_wire(received, expected_wire(GPT, prompt_text(marker), endpoint=endpoint, marked=True), streaming=False)
        sends: Final[list[Outcome]] = []

        def resend() -> Outcome:
            served: Final = send(gateway, endpoint, body)
            sends.append(served)
            return served

        hit: Final = eventually(
            resend, lambda served: served.status == 200 and served.response_id == first.response_id, seconds=30
        )
        assert_answered(hit, marker)
        misses: Final = wire.drain()
        assert len(misses) == len(sends) - 1, (len(misses), len(sends))
        for miss in misses:
            assert_wire(miss, expected_wire(GPT, prompt_text(marker), endpoint=endpoint, marked=True), streaming=False)
        rows: Final = spend_rows(name, frozenset({marker}), expected=len(sends) + 1, seconds=70)
        assert len(rows) == len(sends) + 1, rows
        (hit_row,) = tuple(row for row in rows if row["cache_hit"] == "True")
        assert row_key(str(hit_row["request_id"])) == first.response_id, (hit_row, first.response_id)


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("shape", sorted(HOSTILE_OPTIONS))
def test_d1_to_d4_hostile_prompt_cache_options_reach_the_wire_verbatim_and_the_providers_400_reaches_the_caller(
    gateway: Gateway, endpoint: Endpoint, shape: str
) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        hostile: Final = HOSTILE_OPTIONS[shape]
        outcome, received = observe(
            gateway, wire, endpoint, request_body(endpoint, name, prompt_text(marker), prompt_cache_options=hostile)
        )
        assert outcome.status == 400, (outcome.status, outcome.raw)
        assert "Invalid prompt_cache_options" in outcome.raw, outcome.raw
        assert_wire(
            received,
            expected_wire(GPT, prompt_text(marker), endpoint=endpoint, marked=True, options=hostile),
            streaming=False,
        )
        failure_row(outcome.call_id)


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
def test_d5_a_duplicated_prompt_cache_options_key_resolves_to_the_last_value(
    gateway: Gateway, endpoint: Endpoint
) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        encoded: Final = json.dumps(request_body(endpoint, name, prompt_text(marker), prompt_cache_options=IMPLICIT))
        duplicated: Final = encoded[:-1] + ', "prompt_cache_options": {"mode": "explicit"}}'
        wire.drain()
        outcome: Final = send_raw(gateway, endpoint, duplicated)
        assert_answered(outcome, marker)
        assert_wire(
            only_received(wire),
            expected_wire(GPT, prompt_text(marker), endpoint=endpoint, marked=True, options=EXPLICIT),
            streaming=False,
        )
        success_row(name, marker)


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
def test_d6_a_null_prompt_cache_options_is_treated_as_unset(gateway: Gateway, endpoint: Endpoint) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        outcome, received = observe(
            gateway, wire, endpoint, request_body(endpoint, name, prompt_text(marker), prompt_cache_options=None)
        )
        assert_answered(outcome, marker)
        assert_wire(received, expected_wire(GPT, prompt_text(marker), endpoint=endpoint, marked=True), streaming=False)
        success_row(name, marker)


def test_d7_an_unauthenticated_hostile_request_never_reaches_the_wire(gateway: Gateway) -> None:
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        wire.drain()
        outcome: Final = send(
            gateway,
            "responses",
            request_body("responses", name, prompt_text(fresh_marker()), prompt_cache_options=5),
            key="sk-not-a-key",
        )
        assert outcome.status == 401, (outcome.status, outcome.raw)
        assert wire.drain() == (), "the upstream saw an unauthenticated request"


@pytest.mark.parametrize("shape", sorted(MALFORMED_POINTS))
def test_d8_to_d10_malformed_injection_points_never_crash_the_request(gateway: Gateway, shape: str) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = deployment(scenario, wire, GPT, points=MALFORMED_POINTS[shape])
        settled(gateway, name, wire)
        outcome, received = observe(gateway, wire, "responses", request_body("responses", name, prompt_text(marker)))
        assert_answered(outcome, marker)
        body: Final = body_of(received)
        assert breakpoint_count(body) == 0, received.body
        assert "prompt_cache_options" not in body, received.body
        success_row(name, marker)


@pytest.mark.parametrize(
    ("status", "endpoint"),
    [(400, "chat"), (400, "responses"), (401, "responses")],
    ids=["400-chat", "400-responses", "401-responses"],
)
def test_d11_d12_a_provider_error_on_the_marked_request_reaches_the_caller_after_one_attempt(
    gateway: Gateway, status: int, endpoint: Endpoint
) -> None:
    marker: Final = fresh_marker()
    message: Final = f"scripted provider failure {marker}"
    with wire_server(failing_peer(status, message, "scripted_failure")) as wire, gateway.scenario() as scenario:
        name: Final = deployment(scenario, wire, GPT)
        settled(gateway, name, wire, accepted=frozenset({status}))
        outcome: Final = send(gateway, endpoint, request_body(endpoint, name, prompt_text(marker)))
        assert outcome.status == status, (outcome.status, outcome.raw)
        assert message in outcome.raw, outcome.raw
        attempts: Final = wire.drain()
        assert len(attempts) == 1, [attempt.target for attempt in attempts]
        assert breakpoint_count(body_of(attempts[0])) == 1, attempts[0].body
        failure_row(outcome.call_id)


@pytest.mark.parametrize(("label", "model", "flag"), ODD_FLAGS, ids=[label for label, _, _ in ODD_FLAGS])
def test_d13_an_odd_typed_deployment_flag_never_opts_a_mantle_row_in(
    gateway: Gateway, label: str, model: str, flag: JsonValue
) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(
            gateway, scenario, wire, model, model_info={"supports_prompt_cache_breakpoint": flag}
        )
        outcome, received = observe(gateway, wire, "responses", request_body("responses", name, prompt_text(marker)))
        assert_answered(outcome, marker)
        body: Final = body_of(received)
        assert breakpoint_count(body) == 0, (label, received.body)
        assert "prompt_cache_options" not in body, (label, received.body)
        success_row(name, marker)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_d14_a_point_beside_junk_entries_still_marks_the_anthropic_dialect(
    gateway: Gateway, endpoint: Endpoint
) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire, CLAUDE, points=MIXED_POINTS)
        outcome, received = observe(gateway, wire, endpoint, request_body(endpoint, name, prompt_text(marker)))
        assert_answered(outcome, marker)
        assert_anthropic_marked_wire(received, marker)
        success_row(name, marker, outcome.response_id)


def test_e1_client_breakpoint_prompt_cache_key_and_explicit_mode_pass_through_with_no_second_breakpoint(
    gateway: Gateway,
) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        body: Final[dict[str, JsonValue]] = {
            "model": name,
            "input": [system_item(marked=True, endpoint="responses"), {"role": "user", "content": prompt_text(marker)}],
            "max_output_tokens": MAX_TOKENS,
            "prompt_cache_key": f"key-{marker}",
            "prompt_cache_options": EXPLICIT,
        }
        outcome, received = observe(gateway, wire, "responses", body)
        assert_answered(outcome, marker)
        assert_wire(
            received,
            expected_wire(
                GPT,
                prompt_text(marker),
                endpoint="responses",
                marked=True,
                options=EXPLICIT,
                prompt_cache_key=f"key-{marker}",
            ),
            streaming=False,
        )
        success_row(name, marker)


def test_e2_four_client_breakpoints_leave_no_slot_for_the_configured_point(gateway: Gateway) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        turns: Final = tuple(f"Earlier turn {index} marker-{marker}." for index in range(3))
        body: Final[dict[str, JsonValue]] = {
            "model": name,
            "input": [
                {"role": "system", "content": SYSTEM},
                *(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": turn, "prompt_cache_breakpoint": EXPLICIT}],
                    }
                    for turn in turns
                ),
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text(marker), "prompt_cache_breakpoint": EXPLICIT}
                    ],
                },
            ],
            "max_output_tokens": MAX_TOKENS,
        }
        outcome, received = observe(gateway, wire, "responses", body)
        assert_answered(outcome, marker)
        wire_body: Final = body_of(received)
        items: Final = ITEMS.validate_python(wire_body["input"])
        assert breakpoint_count(wire_body) == 4, received.body
        assert items[0]["role"] == "system" and "prompt_cache_breakpoint" not in json.dumps(items[0]), items[0]
        assert "prompt_cache_options" not in wire_body, received.body
        success_row(name, marker)


def test_e3_a_client_implicit_mode_on_a_pinned_explicit_deployment_wins_on_the_wire(gateway: Gateway) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire, prompt_cache_options=EXPLICIT)
        outcome, received = observe(
            gateway,
            wire,
            "responses",
            request_body("responses", name, prompt_text(marker), prompt_cache_options=IMPLICIT),
        )
        assert_answered(outcome, marker)
        assert_wire(
            received,
            expected_wire(GPT, prompt_text(marker), endpoint="responses", marked=True, options=IMPLICIT),
            streaming=False,
        )
        success_row(name, marker)


def test_e4_an_empty_prompt_cache_options_object_is_kept_as_the_clients_value(gateway: Gateway) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        outcome, received = observe(
            gateway, wire, "responses", request_body("responses", name, prompt_text(marker), prompt_cache_options={})
        )
        assert_answered(outcome, marker)
        assert_wire(
            received,
            expected_wire(GPT, prompt_text(marker), endpoint="responses", marked=True, options={}),
            streaming=False,
        )
        success_row(name, marker)


def test_e5_the_same_request_twice_writes_two_rows_and_two_marked_upstream_requests(gateway: Gateway) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        body: Final = request_body("chat", name, prompt_text(marker), cache=NO_CACHE)
        first, first_received = observe(gateway, wire, "chat", body)
        second, second_received = observe(gateway, wire, "chat", body)
        assert_answered(first, marker)
        assert_answered(second, marker)
        assert first.response_id != second.response_id, (first.response_id, second.response_id)
        for received in (first_received, second_received):
            assert_wire(
                received, expected_wire(GPT, prompt_text(marker), endpoint="chat", marked=True), streaming=False
            )
        rows: Final = spend_rows(name, frozenset({marker}), expected=2)
        assert {row_key(str(row["request_id"])) for row in rows} == {first.response_id, second.response_id}, rows


def test_f1_a_mixed_burst_marks_every_request_and_lands_every_id_once(gateway: Gateway) -> None:
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        wire.drain()
        assert_burst_landed(wire, name, burst(gateway, name, BURST), marked=True)


def test_f4_a_slow_mantle_stream_during_a_burst_completes_with_every_id_landing_once(gateway: Gateway) -> None:
    with wire_server(mantle_peer(pause=0.3)) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        wire.drain()
        assert_burst_landed(wire, name, burst(gateway, name, 12), marked=True)
