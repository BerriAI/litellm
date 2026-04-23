"""
End-to-end tests for the adaptive router. Wires the real strategy + queue + hook
with a mocked Prisma client. No live proxy or DB required.

What we cover:
  1. Full lifecycle: pick -> record turn(s) -> flush -> DB upsert with correct deltas
  2. Owner cache pins attribution: same key + matching model -> updates flow
  3. Convergence in-process: 50 simulated sessions, "good" model dominates last 10
  4. Cold-start state load from DB overrides priors
  5. Failure signal increments beta in the next flush
  6. Unknown request types in DB rows are silently skipped
  7. Flush isolates writes per (router, session, model) tuple
"""

import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.router_strategy.adaptive_router.adaptive_router import AdaptiveRouter
from litellm.router_strategy.adaptive_router.signals import Turn
from litellm.types.router import (
    AdaptiveRouterConfig,
    AdaptiveRouterPreferences,
    AdaptiveRouterWeights,
    RequestType,
)


def _make_router(
    available=("gpt-4o-mini", "gpt-4o"),
    prefs=None,
    costs=None,
):
    if prefs is None:
        prefs = {
            "gpt-4o-mini": AdaptiveRouterPreferences(quality_tier=2, strengths=[]),
            "gpt-4o": AdaptiveRouterPreferences(
                quality_tier=3, strengths=[RequestType.CODE_GENERATION]
            ),
        }
    if costs is None:
        costs = {"gpt-4o-mini": 0.15, "gpt-4o": 5.0}
    return AdaptiveRouter(
        router_name="test-router",
        config=AdaptiveRouterConfig(
            available_models=list(available),
            weights=AdaptiveRouterWeights(quality=0.7, cost=0.3),
        ),
        model_to_prefs=prefs,
        model_to_cost=costs,
    )


def _make_mock_prisma():
    p = MagicMock()
    p.db.litellm_adaptiverouterstate.find_unique = AsyncMock(return_value=None)
    p.db.litellm_adaptiverouterstate.find_many = AsyncMock(return_value=[])
    p.db.litellm_adaptiverouterstate.upsert = AsyncMock()
    p.db.litellm_adaptiveroutersession.upsert = AsyncMock()
    return p


@pytest.mark.asyncio
async def test_pick_record_flush_full_cycle():
    router = _make_router()
    chosen = await router.pick_model(RequestType.CODE_GENERATION)
    assert chosen in router.config.available_models

    # Prime 2 prior turns (distinct content so no other signals fire) so the
    # MIN_TURNS_FOR_CLEAN_CREDIT satisfaction gate is satisfied on turn 3.
    priming = [
        Turn(
            user_content="alpha bravo charlie", assistant_content="delta echo foxtrot"
        ),
        Turn(
            user_content="golf hotel india juliet",
            assistant_content="kilo lima mike november",
        ),
    ]
    for t in priming:
        await router.record_turn(
            session_id="s1",
            model_name=chosen,
            request_type=RequestType.CODE_GENERATION,
            turn=t,
        )
    await router.record_turn(
        session_id="s1",
        model_name=chosen,
        request_type=RequestType.CODE_GENERATION,
        turn=Turn(user_content="thanks, that worked!", assistant_content="ok"),
    )

    prisma = _make_mock_prisma()
    n_state = await router.queue.flush_state_to_db(prisma)
    n_session = await router.queue.flush_session_to_db(prisma)

    assert n_state == 1
    assert n_session == 1
    state_call = prisma.db.litellm_adaptiverouterstate.upsert.call_args
    # satisfaction signal -> +1 alpha, no existing row -> create.alpha == 1.0
    assert state_call.kwargs["data"]["create"]["alpha"] >= 1.0
    assert state_call.kwargs["data"]["create"]["beta"] == 0.0
    assert state_call.kwargs["data"]["create"]["total_samples"] == 1

    session_call = prisma.db.litellm_adaptiveroutersession.upsert.call_args
    assert session_call.kwargs["data"]["create"]["satisfaction_count"] == 1
    assert session_call.kwargs["data"]["create"]["session_id"] == "s1"
    assert session_call.kwargs["data"]["create"]["model_name"] == chosen


@pytest.mark.asyncio
async def test_owner_cache_pins_attribution_to_first_picked_model():
    """First call claims ownership; matching model returns True, mismatch False."""
    router = _make_router()
    chosen = await router.pick_model(RequestType.GENERAL)
    assert router.claim_or_check_owner("sess-own", chosen) is True

    # Same model on later turns keeps attributing.
    for _ in range(5):
        assert router.claim_or_check_owner("sess-own", chosen) is True

    # A different model on a later turn is rejected.
    other = "gpt-4o" if chosen == "gpt-4o-mini" else "gpt-4o-mini"
    assert router.claim_or_check_owner("sess-own", other) is False
    assert router._skipped_updates_total == 1


@pytest.mark.asyncio
async def test_pick_model_returns_valid_models_without_error():
    router = _make_router()
    # Picks may legitimately differ across calls (Thompson sampling is stochastic).
    # Just confirm every pick is valid and nothing raises.
    for _ in range(10):
        m = await router.pick_model(RequestType.GENERAL)
        assert m in router.config.available_models


@pytest.mark.asyncio
async def test_in_process_convergence_high_quality_model_dominates():
    """
    Two models, identical cost. "good" satisfies every turn, "bad" fails every turn.
    After 50 sessions of 4 turns each, "good" should win >=70% of the last 10 picks.
    Seed `random` for determinism since pick_best uses the module-level RNG.
    """
    random.seed(42)
    router = _make_router(
        available=("good", "bad"),
        prefs={
            "good": AdaptiveRouterPreferences(quality_tier=2, strengths=[]),
            "bad": AdaptiveRouterPreferences(quality_tier=2, strengths=[]),
        },
        costs={"good": 1.0, "bad": 1.0},
    )

    picks = []
    for sess in range(50):
        sid = f"conv-{sess}"
        chosen = await router.pick_model(RequestType.GENERAL)
        for _turn_i in range(4):
            if chosen == "good":
                turn = Turn(user_content="thanks!", assistant_content="ok")
            else:
                turn = Turn(
                    tool_calls=[{"name": "x", "arguments": {}}],
                    tool_results=[{"is_error": True, "content": "boom"}],
                )
            await router.record_turn(sid, chosen, RequestType.GENERAL, turn)
        picks.append(chosen)

    last_10 = picks[-10:]
    good_share = last_10.count("good") / 10
    assert good_share >= 0.7, f"good_share={good_share} (last picks={picks})"


@pytest.mark.asyncio
async def test_failure_signal_increments_beta_after_flush():
    router = _make_router(
        available=("only",),
        prefs={"only": AdaptiveRouterPreferences(quality_tier=2, strengths=[])},
        costs={"only": 1.0},
    )
    chosen = await router.pick_model(RequestType.GENERAL)
    assert chosen == "only"

    await router.record_turn(
        session_id="f1",
        model_name=chosen,
        request_type=RequestType.GENERAL,
        turn=Turn(
            tool_calls=[{"name": "x", "arguments": {}}],
            tool_results=[{"is_error": True, "content": ""}],
        ),
    )

    prisma = _make_mock_prisma()
    n_state = await router.queue.flush_state_to_db(prisma)
    assert n_state == 1
    state_call = prisma.db.litellm_adaptiverouterstate.upsert.call_args
    assert state_call.kwargs["data"]["create"]["beta"] >= 1.0
    assert state_call.kwargs["data"]["create"]["alpha"] == 0.0


@pytest.mark.asyncio
async def test_load_state_from_db_overrides_cold_start():
    router = _make_router()
    fake_row = MagicMock()
    fake_row.request_type = RequestType.GENERAL.value
    fake_row.model_name = "gpt-4o"
    fake_row.alpha = 90.0
    fake_row.beta = 10.0

    prisma = _make_mock_prisma()
    prisma.db.litellm_adaptiverouterstate.find_many = AsyncMock(return_value=[fake_row])

    await router.load_state_from_db(prisma)

    cell = router._cells[(RequestType.GENERAL, "gpt-4o")]
    assert cell.alpha == 90.0
    assert cell.beta == 10.0


@pytest.mark.asyncio
async def test_load_state_from_db_handles_unknown_request_type():
    router = _make_router()
    bad_row = MagicMock()
    bad_row.request_type = "unknown_v1_type"
    bad_row.model_name = "gpt-4o"
    bad_row.alpha = 50.0
    bad_row.beta = 50.0

    prisma = _make_mock_prisma()
    prisma.db.litellm_adaptiverouterstate.find_many = AsyncMock(return_value=[bad_row])

    # Should not raise; bad row is silently skipped and cold-start cells remain.
    await router.load_state_from_db(prisma)
    cell = router._cells[(RequestType.GENERAL, "gpt-4o")]
    # Cold-start: tier 3 base = 0.7, mass = 10 -> alpha = 7, beta = 3
    assert cell.alpha == pytest.approx(7.0)
    assert cell.beta == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_flush_isolates_writes_per_router_session_model():
    router = _make_router()
    # Prime 2 prior turns per session to clear the MIN_TURNS_FOR_CLEAN_CREDIT gate.
    for sid, model in (("s1", "gpt-4o"), ("s2", "gpt-4o-mini")):
        for _ in range(2):
            await router.record_turn(
                sid,
                model,
                RequestType.GENERAL,
                Turn(user_content="hi", assistant_content="hello"),
            )
    await router.record_turn(
        "s1", "gpt-4o", RequestType.GENERAL, Turn(user_content="thanks!")
    )
    await router.record_turn(
        "s2", "gpt-4o-mini", RequestType.GENERAL, Turn(user_content="thanks!")
    )

    prisma = _make_mock_prisma()
    n = await router.queue.flush_session_to_db(prisma)
    assert n == 2
    assert prisma.db.litellm_adaptiveroutersession.upsert.call_count == 2

    n_state = await router.queue.flush_state_to_db(prisma)
    assert n_state == 2
    assert prisma.db.litellm_adaptiverouterstate.upsert.call_count == 2


@pytest.mark.asyncio
async def test_repeated_flush_drains_queue_and_subsequent_flush_is_noop():
    """Verifies the queue is fully drained on flush -- a second flush writes nothing."""
    router = _make_router()
    chosen = await router.pick_model(RequestType.GENERAL)
    # Prime 2 prior turns so satisfaction can fire on the third turn.
    for _ in range(2):
        await router.record_turn(
            "drain-1",
            chosen,
            RequestType.GENERAL,
            Turn(user_content="hi", assistant_content="hello"),
        )
    await router.record_turn(
        "drain-1", chosen, RequestType.GENERAL, Turn(user_content="thanks!")
    )

    prisma = _make_mock_prisma()
    assert await router.queue.flush_state_to_db(prisma) == 1
    assert await router.queue.flush_session_to_db(prisma) == 1

    # Second drain should be a no-op (queue is empty).
    assert await router.queue.flush_state_to_db(prisma) == 0
    assert await router.queue.flush_session_to_db(prisma) == 0
    assert prisma.db.litellm_adaptiverouterstate.upsert.call_count == 1
    assert prisma.db.litellm_adaptiveroutersession.upsert.call_count == 1
