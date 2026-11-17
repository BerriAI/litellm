"""Contract: usage, cost and spend rows of ``/v1/realtime`` sessions, and budgets and rate limits around them.

Billing (``RealtimeAPITokenUsageProcessor`` and ``handle_realtime_stream_cost_calculation``,
litellm/cost_calculator.py): a session writes ONE ``_arealtime`` spend row when it ends, whose tokens are the sum
of every ``response.done`` usage relayed during the session. Prices are looked up in order: the deployment's
custom rates, its ``base_model``, the model the provider reported in ``session.created``, then the request
model. Audio tokens bill at ``input_cost_per_audio_token`` / ``output_cost_per_audio_token`` and fall back to
the text rate when the model has none (``_modality_token_cost``); cached audio bills at
``cache_read_input_audio_token_cost`` and falls back to ``cache_read_input_token_cost``. A provider that closes
before sending any non-error frame is logged as a failure (``_backend_refused_session``), otherwise the
session is logged as a success with the turns completed so far.

Admission (``_route_realtime_websocket_session``, proxy_server.py): budgets and rate limits are checked once,
at the upgrade (auth and ``common_processing_pre_call_logic``). A pre-call refusal sends one ``error`` event
of type ``guardrail_error`` and closes 1011 ``Pre-call error``. Turns inside an open session are never
re-checked, which is the main behavioural difference from the Responses websocket.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration.providers.dec1_realtime_support import (
    INTERNAL_ERROR,
    REALTIME_MODEL,
    AwaitClose,
    Close,
    CloseWith,
    PeerScript,
    Reply,
    Send,
    done_event,
    key_spend,
    openai_deployment,
    peer_deployment,
    priced_deployment,
    realtime,
    realtime_peer,
    scripted,
    spend_rows,
    turn_steps,
    upgrades,
)

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]

AUDIO_RATES: Final = {"input_cost_per_audio_token": 0.01, "output_cost_per_audio_token": 0.02}


def _spend(row: dict[str, object]) -> float:
    return float(str(row["spend"]))


def test_one_session_with_three_turns_writes_one_row_with_the_summed_tokens(gateway: Gateway) -> None:
    """A priced deployment (0.001 in, 0.002 out) answers three ``response.create`` with 10 text in and 5 text out
    each. After the client closes, exactly one ``_arealtime`` row exists for the key with ``prompt_tokens`` 30,
    ``completion_tokens`` 15, ``total_tokens`` 45 and ``spend`` 30 * 0.001 + 15 * 0.002 = 0.06."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),) * 3)
        model: Final = priced_deployment(scenario, handle)
        key: Final = scenario.key(models=[model])
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(3))
        rows: Final = spend_rows(gateway, key, 1)
        assert len(rows) == 1, rows
        row: Final = rows[0]
        assert row["call_type"] == "_arealtime", row
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (30, 15, 45), row
        assert _spend(row) == pytest.approx(0.06), row


def test_audio_tokens_bill_at_the_deployment_audio_rates(gateway: Gateway) -> None:
    """The deployment sets text rates 0.001 / 0.002 and audio rates 0.01 / 0.02. One turn reports 10 text and 20
    audio input tokens and 5 text and 8 audio output tokens. The row has ``prompt_tokens`` 30,
    ``completion_tokens`` 13 and ``spend`` 10 * 0.001 + 20 * 0.01 + 5 * 0.002 + 8 * 0.02 = 0.38."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(text_in=10, audio_in=20, text_out=5, audio_out=8),))
        model: Final = priced_deployment(scenario, handle, **AUDIO_RATES)
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        row: Final = spend_rows(gateway, key, 1)[0]
        assert (row["prompt_tokens"], row["completion_tokens"]) == (30, 13), row
        assert _spend(row) == pytest.approx(0.38), row


def test_audio_tokens_without_an_audio_rate_bill_at_the_text_rate(gateway: Gateway) -> None:
    """The same turn on a deployment with only text rates (0.001 / 0.002) costs
    10 * 0.001 + 20 * 0.001 + 5 * 0.002 + 8 * 0.002 = 0.056: audio input falls back to the prompt base rate
    (``_modality_token_cost``) and audio output to the completion base rate."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(text_in=10, audio_in=20, text_out=5, audio_out=8),))
        model: Final = priced_deployment(scenario, handle)
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        assert _spend(spend_rows(gateway, key, 1)[0]) == pytest.approx(0.056)


def test_cached_audio_tokens_bill_at_the_deployment_cached_audio_rate(gateway: Gateway) -> None:
    """Rates: text 0.001, audio 0.01, cached text 0.0005, cached audio 0.0001, output 0.002. One turn reports 10
    text (4 cached) and 20 audio (10 cached) input tokens and 5 text output tokens. Spend is
    6 * 0.001 + 4 * 0.0005 + 10 * 0.01 + 10 * 0.0001 + 5 * 0.002 = 0.119."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(
            scenario, (done_event(text_in=10, audio_in=20, text_out=5, cached_text=4, cached_audio=10),)
        )
        model: Final = priced_deployment(
            scenario,
            handle,
            input_cost_per_audio_token=0.01,
            cache_read_input_token_cost=0.0005,
            cache_read_input_audio_token_cost=0.0001,
        )
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        assert _spend(spend_rows(gateway, key, 1)[0]) == pytest.approx(0.119)


def test_cached_audio_without_its_own_rate_bills_at_the_cached_text_rate(gateway: Gateway) -> None:
    """Same turn without ``cache_read_input_audio_token_cost``: the 10 cached audio tokens bill at
    ``cache_read_input_token_cost`` 0.0005 (``_calculate_cache_read_cost`` default). Spend is
    6 * 0.001 + 4 * 0.0005 + 10 * 0.01 + 10 * 0.0005 + 5 * 0.002 = 0.123."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(
            scenario, (done_event(text_in=10, audio_in=20, text_out=5, cached_text=4, cached_audio=10),)
        )
        model: Final = priced_deployment(
            scenario, handle, input_cost_per_audio_token=0.01, cache_read_input_token_cost=0.0005
        )
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        assert _spend(spend_rows(gateway, key, 1)[0]) == pytest.approx(0.123)


def test_an_unpriced_deployment_is_priced_by_the_model_the_provider_reports_in_session_created(
    gateway: Gateway,
) -> None:
    """The deployment is ``openai/gpt-4o-realtime-preview`` with no custom rates (absent from the cost map) and
    the provider's ``session.created`` reports ``session.model`` ``gpt-realtime-2``. One turn of 10 text in and 5
    text out is priced at the cost map rates of ``gpt-realtime-2`` (4e-06 in, 2.4e-05 out):
    10 * 4e-06 + 5 * 2.4e-05 = 0.00016 (``received_model`` in ``handle_realtime_stream_cost_calculation``)."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),), session_model="gpt-realtime-2")
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        assert _spend(spend_rows(gateway, key, 1)[0]) == pytest.approx(0.00016)


def test_deployment_custom_rates_win_over_the_model_reported_in_session_created(gateway: Gateway) -> None:
    """The same provider reports ``gpt-realtime-2`` but the deployment sets 0.001 / 0.002. The custom pricing
    key is tried first, so the 10 in / 5 out turn costs 0.02, not 0.00016."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),), session_model="gpt-realtime-2")
        model: Final = priced_deployment(scenario, handle)
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        assert _spend(spend_rows(gateway, key, 1)[0]) == pytest.approx(0.02)


def test_the_row_names_the_provider_model_the_public_group_and_the_provider(gateway: Gateway) -> None:
    """After one priced turn the row has ``model`` ``gpt-4o-realtime-preview`` (the provider model, not the
    alias), ``model_group`` equal to the public deployment name, ``custom_llm_provider`` ``openai`` and
    ``status`` ``success``. UNVERIFIED: ``model`` could carry the ``openai/`` prefix on this route."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = priced_deployment(scenario, handle)
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        row: Final = spend_rows(gateway, key, 1)[0]
        assert row["model"] == REALTIME_MODEL, row
        assert row["model_group"] == model, row
        assert row["custom_llm_provider"] == "openai", row
        assert row["status"] == "success", row


def test_a_session_with_no_turns_writes_a_zero_spend_row(gateway: Gateway) -> None:
    """A client opens a session, receives ``session.created`` and closes without any ``response.create``. The
    session still ends through ``log_messages``, so one ``_arealtime`` row is written with ``spend`` 0 and zero
    tokens. UNVERIFIED: a zero-cost success row might be skipped by the spend writer."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = priced_deployment(scenario, handle)
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key)
        row: Final = spend_rows(gateway, key, 1)[0]
        assert (row["prompt_tokens"], row["completion_tokens"], _spend(row)) == (0, 0, 0.0), row


def test_the_session_cost_is_added_to_the_key_spend(gateway: Gateway) -> None:
    """After a 0.38 session (audio rates as above) the key's ``LiteLLM_VerificationToken.spend`` converges to
    0.38, the same value as the row; realtime spend is not tracked apart from HTTP spend."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(text_in=10, audio_in=20, text_out=5, audio_out=8),))
        model: Final = priced_deployment(scenario, handle, **AUDIO_RATES)
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        spend_rows(gateway, key, 1)
        assert eventually(lambda: key_spend(gateway, key), lambda spent: spent > 0, seconds=60) == pytest.approx(0.38)


def test_the_session_cost_is_added_to_the_team_spend(gateway: Gateway) -> None:
    """A key of a fresh team runs one 0.06 session (three 10/5 turns at 0.001/0.002). The team's
    ``LiteLLM_TeamTable.spend`` converges to 0.06."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),) * 3)
        model: Final = priced_deployment(scenario, handle)
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(3))

        def team_spend() -> float:
            rows = gateway.rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
            return float(str(rows[0]["spend"]))

        assert eventually(team_spend, lambda spent: spent > 0, seconds=70) == pytest.approx(0.06)


def test_a_provider_that_errors_and_closes_before_any_session_frame_writes_a_failure_row(gateway: Gateway) -> None:
    """The owned peer sends only an ``error`` event and closes 1011 right after accept, before
    ``session.created``. ``_backend_refused_session`` is true, failure handlers run, and the key's one row has
    ``status`` ``failure`` and ``spend`` 0. The client gets the provider error and a close 1011."""
    script: Final = PeerScript(
        opening=(
            Reply({"type": "error", "error": {"type": "invalid_request_error", "message": "bad model"}}),
            Close(1011, "bye"),
        )
    )
    with realtime_peer(script) as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        key: Final = scenario.key()
        outcome: Final = realtime(gateway, f"model={model}", key=key, steps=(AwaitClose(),))
        assert outcome.close_code == INTERNAL_ERROR, outcome
        row: Final = spend_rows(gateway, key, 1)[0]
        assert (row["status"], _spend(row)) == ("failure", 0.0), row


def test_a_provider_that_drops_after_a_completed_turn_still_bills_that_turn(gateway: Gateway) -> None:
    """The owned peer answers the first ``response.create`` with a 10/5 ``response.done`` and the second by
    closing 1011. The session had non-error frames, so it is logged as a success: one row with ``status``
    ``success``, ``prompt_tokens`` 10 and ``spend`` 0.02 (rates 0.001/0.002)."""
    turns: Final = iter((Reply(done_event()), Close(1011, "provider crash")))

    def on_frame(frame: dict[str, object]) -> tuple[Reply | Close, ...]:
        return (next(turns),) if frame.get("type") == "response.create" else ()

    with realtime_peer(PeerScript(on_frame=on_frame)) as peer, gateway.scenario() as scenario:
        model: Final = peer_deployment(scenario, peer)
        key: Final = scenario.key()
        realtime(
            gateway,
            f"model={model}",
            key=key,
            steps=(*turn_steps(1), Send({"type": "response.create"}), AwaitClose()),
        )
        row: Final = spend_rows(gateway, key, 1)[0]
        assert (row["status"], row["prompt_tokens"]) == ("success", 10), row
        assert _spend(row) == pytest.approx(0.02), row


def test_a_client_that_disconnects_after_two_turns_is_billed_for_both(gateway: Gateway) -> None:
    """The client completes two 10/5 turns and closes with 1001 (going away) without waiting for anything else.
    One row is written with ``prompt_tokens`` 20 and ``spend`` 0.04."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(), done_event(), done_event()))
        model: Final = priced_deployment(scenario, handle)
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key, steps=(*turn_steps(2), CloseWith(1001)))
        row: Final = spend_rows(gateway, key, 1)[0]
        assert row["prompt_tokens"] == 20, row
        assert _spend(row) == pytest.approx(0.04), row


def test_two_sessions_on_one_key_write_two_rows_with_distinct_request_ids(gateway: Gateway) -> None:
    """Two sequential one-turn sessions on the same key write two ``_arealtime`` rows with different
    ``request_id`` values, each with ``prompt_tokens`` 10."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(), done_event()))
        model: Final = priced_deployment(scenario, handle)
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        rows: Final = spend_rows(gateway, key, 2)
        assert [row["prompt_tokens"] for row in rows] == [10, 10], rows
        assert rows[0]["request_id"] != rows[1]["request_id"], rows


def test_the_row_metadata_carries_the_key_alias_and_team(gateway: Gateway) -> None:
    """A key with ``key_alias`` ``dec1-rt-alias`` in a fresh team runs one session. The row's ``metadata`` has
    ``user_api_key_alias`` ``dec1-rt-alias`` and ``user_api_key_team_id`` equal to the team id, the same
    attribution an HTTP call gets."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = priced_deployment(scenario, handle)
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team, key_alias=f"dec1-rt-alias-{team[:8]}")
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        metadata: Final = object_value(spend_rows(gateway, key, 1)[0]["metadata"])
        assert metadata.get("user_api_key_alias") == f"dec1-rt-alias-{team[:8]}", metadata
        assert metadata.get("user_api_key_team_id") == team, metadata


# ---------------------------------------------------------------------------------------------------------
# admission: budgets and rate limits
# ---------------------------------------------------------------------------------------------------------


def test_each_session_counts_as_one_request_toward_the_key_rpm(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 2`` opens two sessions (both get ``session.created``). The third is refused by the
    pre-call limiter: one ``error`` event of type ``guardrail_error`` naming ``Rate limit exceeded`` and close
    1011 ``Pre-call error``; the upstream saw exactly two handshakes."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(rpm_limit=2)
        first: Final = realtime(gateway, f"model={model}", key=key)
        second: Final = realtime(gateway, f"model={model}", key=key)
        third: Final = realtime(gateway, f"model={model}", key=key, steps=(AwaitClose(),))
        assert first.types[:1] == second.types[:1] == ("session.created",), (first, second)
        assert third.error_types() == ("guardrail_error",), third
        assert "Rate limit exceeded" in third.error_messages()[0], third
        assert (third.close_code, third.close_reason) == (INTERNAL_ERROR, "Pre-call error"), third
        assert len(upgrades(gateway, handle.scenario_id)) == 2


def test_turns_inside_one_session_do_not_count_toward_the_key_rpm(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` opens one session and runs five turns; all five ``response.done`` arrive.
    Only the upgrade passes through the limiter, so the turns are not requests. A Rust port that meters frames
    like the Responses websocket would refuse turn two."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),) * 5)
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(rpm_limit=1)
        outcome: Final = realtime(gateway, f"model={model}", key=key, steps=turn_steps(5))
        assert outcome.types.count("response.done") == 5, outcome


def test_turns_inside_one_session_are_not_refused_by_the_key_tpm(gateway: Gateway) -> None:
    """A key with ``tpm_limit: 20`` runs one session of three 10/5 turns (45 tokens, over the limit after the
    second turn). Every turn completes: tokens are only counted when the session ends.
    UNVERIFIED: whether the 45 tokens are then counted toward the key TPM window (see the next test)."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),) * 3)
        model: Final = priced_deployment(scenario, handle)
        key: Final = scenario.key(tpm_limit=20)
        outcome: Final = realtime(gateway, f"model={model}", key=key, steps=turn_steps(3))
        assert outcome.types.count("response.done") == 3, outcome


def test_a_session_over_the_key_tpm_refuses_the_next_session_in_the_same_minute(gateway: Gateway) -> None:
    """After the 45 token session above on a ``tpm_limit: 20`` key, the next upgrade in the same minute is
    refused with a ``guardrail_error`` event naming ``Limit type: tokens`` and close 1011. UNVERIFIED: read
    from the v3 limiter success hook, which counts ``total_tokens`` of the logged response for any call type;
    a minute boundary between the two sessions makes this flaky by design."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),) * 4)
        model: Final = priced_deployment(scenario, handle)
        key: Final = scenario.key(tpm_limit=20)
        realtime(gateway, f"model={model}", key=key, steps=turn_steps(3))
        spend_rows(gateway, key, 1)
        refused: Final = eventually(
            lambda: realtime(gateway, f"model={model}", key=key, steps=(AwaitClose(),)),
            lambda outcome: outcome.close_code == INTERNAL_ERROR,
            seconds=20,
        )
        assert "tokens" in refused.error_messages()[0], refused


def test_a_session_admitted_under_budget_may_end_over_it_and_the_next_upgrade_is_refused(gateway: Gateway) -> None:
    """A key with ``max_budget: 0.1`` runs one 0.38 session: it is admitted and every turn completes because the
    budget is only checked at upgrade. Once the key spend reaches 0.38 the next upgrade is refused at auth
    (HTTP 403 handshake or close 1008) and the upstream sees no further handshake."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(text_in=10, audio_in=20, text_out=5, audio_out=8),) * 2)
        model: Final = priced_deployment(scenario, handle, **AUDIO_RATES)
        key: Final = scenario.key(max_budget=0.1)
        first: Final = realtime(gateway, f"model={model}", key=key, steps=turn_steps(1))
        assert first.types.count("response.done") == 1, first
        eventually(lambda: key_spend(gateway, key), lambda spent: spent >= 0.38 - 1e-9, seconds=70)
        second: Final = eventually(
            lambda: realtime(gateway, f"model={model}", key=key),
            lambda outcome: "session.created" not in outcome.types,
            seconds=30,
        )
        assert second.handshake_status == 403 or second.close_code == 1008, second
        assert len(upgrades(gateway, handle.scenario_id)) == 1


def test_an_over_budget_team_key_is_refused_at_the_upgrade(gateway: Gateway) -> None:
    """A team with ``max_budget: 0`` and a key in it upgrades on a realtime deployment. The team budget check
    in auth refuses before accept (HTTP 403, or close 1008) and the upstream sees no handshake."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        team: Final = scenario.team(max_budget=0)
        key: Final = scenario.key(team_id=team)
        outcome: Final = realtime(gateway, f"model={model}", key=key)
        assert "session.created" not in outcome.types, outcome
        assert outcome.handshake_status == 403 or outcome.close_code == 1008, outcome
        assert upgrades(gateway, handle.scenario_id) == ()


def test_a_key_over_its_model_max_budget_is_refused_on_that_model_only(gateway: Gateway) -> None:
    """A key with ``model_max_budget: {<realtime model>: {"budget_limit": 0.0}}`` is refused on that model
    (no ``session.created``) but still opens a session on a second realtime deployment.
    UNVERIFIED: the model budget is checked in auth with the realtime body ``{"model": <model>}``."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        limited: Final = openai_deployment(scenario, handle)
        other: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(model_max_budget={limited: {"budget_limit": 0.0, "time_period": "1d"}})
        assert "session.created" not in realtime(gateway, f"model={limited}", key=key).types
        assert realtime(gateway, f"model={other}", key=key).types[:1] == ("session.created",)


def test_an_open_session_holds_a_parallel_slot_until_it_ends(gateway: Gateway) -> None:
    """A key with ``max_parallel_requests: 1`` keeps session A open (it waits on a second socket in a thread).
    Session B, opened meanwhile, is refused with an ``error`` event and close 1011 ``Pre-call error``. After A
    closes, session C is admitted (``_release_realtime_max_parallel_slot`` or the success hook freed the slot).
    UNVERIFIED: the parallel slot is held for the whole session rather than released after admission."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),) * 3)
        model: Final = openai_deployment(scenario, handle)
        key: Final = scenario.key(max_parallel_requests=1)
        opened: Final = threading.Event()
        release: Final = threading.Event()

        async def hold() -> None:
            import websockets

            from tests.integration.providers.dec1_realtime_support import proxy_ws_base

            async with websockets.connect(
                f"{proxy_ws_base(gateway)}/v1/realtime?model={model}",
                additional_headers={"Authorization": f"Bearer {key}"},
            ) as socket:
                assert json.loads(await socket.recv())["type"] == "session.created"
                opened.set()
                await asyncio.to_thread(release.wait, 60)

        holder: Final = threading.Thread(target=lambda: asyncio.run(hold()), daemon=True)
        holder.start()
        assert opened.wait(30), "session A never opened"
        refused: Final = realtime(gateway, f"model={model}", key=key, steps=(AwaitClose(),))
        release.set()
        holder.join(30)
        assert refused.close_code == INTERNAL_ERROR, refused
        admitted: Final = eventually(
            lambda: realtime(gateway, f"model={model}", key=key),
            lambda outcome: outcome.types[:1] == ("session.created",),
            seconds=20,
        )
        assert admitted.types[:1] == ("session.created",), admitted


def test_a_refused_session_leaves_the_rpm_window_for_the_next_key_untouched(gateway: Gateway) -> None:
    """Two keys share nothing: key A with ``rpm_limit: 1`` is refused on its second session, and key B (no
    limit) then opens a session on the same deployment normally. A limiter keyed by deployment instead of key
    would refuse B."""
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, (done_event(),))
        model: Final = openai_deployment(scenario, handle)
        limited: Final = scenario.key(rpm_limit=1)
        free: Final = scenario.key()
        realtime(gateway, f"model={model}", key=limited)
        assert realtime(gateway, f"model={model}", key=limited, steps=(AwaitClose(),)).close_code == INTERNAL_ERROR
        assert realtime(gateway, f"model={model}", key=free).types[:1] == ("session.created",)
