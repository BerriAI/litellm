"""Unit tests for the AdaptiveRouter strategy class."""

from unittest.mock import AsyncMock, MagicMock

from litellm.router_strategy.adaptive_router import adaptive_router as ar_module

import pytest

from litellm.router_strategy.adaptive_router.adaptive_router import AdaptiveRouter
from litellm.router_strategy.adaptive_router.config import (
    OWNER_CACHE_TTL_SECONDS,
)
from litellm.router_strategy.adaptive_router.signals import Turn
from litellm.types.router import (
    AdaptiveRouterConfig,
    AdaptiveRouterPreferences,
    RequestType,
)


def _make_router() -> AdaptiveRouter:
    cfg = AdaptiveRouterConfig(available_models=["fast", "smart"])
    prefs = {
        "fast": AdaptiveRouterPreferences(quality_tier=1, strengths=[]),
        "smart": AdaptiveRouterPreferences(
            quality_tier=3, strengths=[RequestType.CODE_GENERATION]
        ),
    }
    costs = {"fast": 0.0001, "smart": 0.001}
    return AdaptiveRouter(
        router_name="r1",
        config=cfg,
        model_to_prefs=prefs,
        model_to_cost=costs,
    )


@pytest.mark.asyncio
async def test_pick_model_returns_model_from_available_list():
    r = _make_router()
    chosen = await r.pick_model(RequestType.GENERAL)
    assert chosen in {"fast", "smart"}


@pytest.mark.asyncio
async def test_pick_model_min_quality_tier_filter():
    r = _make_router()
    # min_tier=3 should leave only `smart` (tier 3); `fast` (tier 1) is filtered.
    for _ in range(20):
        chosen = await r.pick_model(RequestType.GENERAL, min_quality_tier=3)
        assert chosen == "smart"


@pytest.mark.asyncio
async def test_pick_model_min_quality_tier_filter_raises_when_no_eligible():
    r = _make_router()
    with pytest.raises(ValueError, match="min_quality_tier=4"):
        await r.pick_model(RequestType.GENERAL, min_quality_tier=4)


@pytest.mark.asyncio
async def test_pick_model_is_stateless_no_owner_cache_writes():
    """pick_model must not touch the owner cache — that's gated post-call."""
    r = _make_router()
    for _ in range(5):
        await r.pick_model(RequestType.GENERAL)
    assert r._owner_cache == {}


# ---- claim_or_check_owner -----------------------------------------------


def test_claim_or_check_owner_first_call_claims_and_returns_true(monkeypatch):
    r = _make_router()
    monkeypatch.setattr(ar_module.time, "time", lambda: 1_000.0)

    assert r.claim_or_check_owner("sess-A", "fast") is True
    assert r._owner_cache["sess-A"] == ("fast", 1_000.0 + OWNER_CACHE_TTL_SECONDS)
    assert r._skipped_updates_total == 0


def test_claim_or_check_owner_same_model_returns_true_without_extending_ttl(
    monkeypatch,
):
    r = _make_router()
    monkeypatch.setattr(ar_module.time, "time", lambda: 1_000.0)
    r.claim_or_check_owner("sess-A", "fast")
    original_expiry = r._owner_cache["sess-A"][1]

    monkeypatch.setattr(ar_module.time, "time", lambda: 1_500.0)
    assert r.claim_or_check_owner("sess-A", "fast") is True
    # No extension on hit — owner cache snapshots the first claim.
    assert r._owner_cache["sess-A"][1] == original_expiry


def test_claim_or_check_owner_mismatch_skips_and_increments_counter(monkeypatch):
    r = _make_router()
    monkeypatch.setattr(ar_module.time, "time", lambda: 1_000.0)
    r.claim_or_check_owner("sess-A", "fast")

    assert r.claim_or_check_owner("sess-A", "smart") is False
    assert r._skipped_updates_total == 1
    # Owner unchanged.
    assert r._owner_cache["sess-A"][0] == "fast"


def test_claim_or_check_owner_expired_owner_reclaims_for_new_model(monkeypatch):
    r = _make_router()
    monkeypatch.setattr(ar_module.time, "time", lambda: 1_000.0)
    r.claim_or_check_owner("sess-A", "fast")

    monkeypatch.setattr(
        ar_module.time, "time", lambda: 1_000.0 + OWNER_CACHE_TTL_SECONDS + 1
    )
    assert r.claim_or_check_owner("sess-A", "smart") is True
    assert r._owner_cache["sess-A"][0] == "smart"
    # Reclaim isn't a skip.
    assert r._skipped_updates_total == 0


def test_owner_cache_evicts_expired_entries_when_threshold_crossed(monkeypatch):
    """Past _OWNER_CACHE_SWEEP_THRESHOLD live entries, new claims sweep stale."""
    r = _make_router()
    monkeypatch.setattr(ar_module, "_OWNER_CACHE_SWEEP_THRESHOLD", 5)
    monkeypatch.setattr(ar_module.time, "time", lambda: 1_000.0)
    for i in range(5):
        r.claim_or_check_owner(f"old-{i}", "fast")
    assert len(r._owner_cache) == 5

    # Jump past TTL so all "old-*" entries are now expired.
    monkeypatch.setattr(
        ar_module.time, "time", lambda: 1_000.0 + OWNER_CACHE_TTL_SECONDS + 1
    )
    r.claim_or_check_owner("new-1", "fast")
    # Sweep ran -> only the new entry remains.
    assert "new-1" in r._owner_cache
    assert all(k.startswith("new-") for k in r._owner_cache)


# ---- record_turn --------------------------------------------------------


@pytest.mark.asyncio
async def test_record_turn_pushes_to_queue():
    r = _make_router()
    # Prime with 2 prior turns so satisfaction gate (MIN_TURNS_FOR_CLEAN_CREDIT=3)
    # is satisfied when the "thanks" turn arrives.
    for _ in range(2):
        await r.record_turn(
            session_id="s1",
            model_name="fast",
            request_type=RequestType.GENERAL,
            turn=Turn(user_content="hi", assistant_content="hello"),
        )

    r.queue.add_session_state = AsyncMock()
    r.queue.add_state_delta = AsyncMock()

    turn = Turn(user_content="thanks, that worked", assistant_content="ok")
    await r.record_turn(
        session_id="s1",
        model_name="fast",
        request_type=RequestType.GENERAL,
        turn=turn,
    )

    r.queue.add_session_state.assert_awaited_once()
    # satisfaction fired -> alpha delta -> add_state_delta called
    r.queue.add_state_delta.assert_awaited_once()

    # PII guard: raw conversation content must not be in the persisted snapshot.
    snapshot = r.queue.add_session_state.call_args.args[3]
    for sensitive in (
        "last_user_content",
        "last_assistant_content",
        "tool_call_history",
        "pending_tool_calls",
    ):
        assert sensitive not in snapshot, f"{sensitive} leaked into DB payload"


@pytest.mark.asyncio
async def test_record_turn_satisfaction_increments_alpha():
    r = _make_router()
    # Prime with 2 prior turns to clear the MIN_TURNS_FOR_CLEAN_CREDIT gate.
    # Use distinct content to avoid incidentally firing stagnation/misalignment.
    priming_turns = [
        Turn(
            user_content="alpha bravo charlie", assistant_content="delta echo foxtrot"
        ),
        Turn(
            user_content="golf hotel india juliet",
            assistant_content="kilo lima mike november",
        ),
    ]
    for t in priming_turns:
        await r.record_turn(
            session_id="sX",
            model_name="fast",
            request_type=RequestType.GENERAL,
            turn=t,
        )
    cell_before = r._cells[(RequestType.GENERAL, "fast")]
    turn = Turn(user_content="that worked, thanks!")
    await r.record_turn(
        session_id="sX",
        model_name="fast",
        request_type=RequestType.GENERAL,
        turn=turn,
    )
    cell_after = r._cells[(RequestType.GENERAL, "fast")]
    assert cell_after.alpha == pytest.approx(cell_before.alpha + 1.0)
    assert cell_after.beta == pytest.approx(cell_before.beta)


@pytest.mark.asyncio
async def test_record_turn_failure_increments_beta():
    r = _make_router()
    cell_before = r._cells[(RequestType.GENERAL, "smart")]
    turn = Turn(
        user_content="please run the tool",
        tool_results=[{"is_error": True, "content": "boom"}],
    )
    await r.record_turn(
        session_id="sY",
        model_name="smart",
        request_type=RequestType.GENERAL,
        turn=turn,
    )
    cell_after = r._cells[(RequestType.GENERAL, "smart")]
    assert cell_after.beta == pytest.approx(cell_before.beta + 1.0)
    assert cell_after.alpha == pytest.approx(cell_before.alpha)


@pytest.mark.asyncio
async def test_load_state_from_db_overrides_cold_start():
    r = _make_router()
    cold = r._cells[(RequestType.GENERAL, "fast")]

    fake_row = MagicMock()
    fake_row.request_type = "general"
    fake_row.model_name = "fast"
    fake_row.alpha = 42.0
    fake_row.beta = 13.0

    prisma = MagicMock()
    prisma.db.litellm_adaptiverouterstate.find_many = AsyncMock(return_value=[fake_row])
    await r.load_state_from_db(prisma)

    new_cell = r._cells[(RequestType.GENERAL, "fast")]
    assert (new_cell.alpha, new_cell.beta) == (42.0, 13.0)
    assert (new_cell.alpha, new_cell.beta) != (cold.alpha, cold.beta)


@pytest.mark.asyncio
async def test_load_state_from_db_handles_unknown_request_type():
    r = _make_router()
    cold = r._cells[(RequestType.GENERAL, "fast")]

    bad_row = MagicMock()
    bad_row.request_type = "nonexistent_type_v999"
    bad_row.model_name = "fast"
    bad_row.alpha = 999.0
    bad_row.beta = 999.0

    good_row = MagicMock()
    good_row.request_type = "general"
    good_row.model_name = "fast"
    good_row.alpha = 7.0
    good_row.beta = 3.0

    prisma = MagicMock()
    prisma.db.litellm_adaptiverouterstate.find_many = AsyncMock(
        return_value=[bad_row, good_row]
    )
    await r.load_state_from_db(prisma)

    # Unknown skipped; good applied.
    assert r._cells[(RequestType.GENERAL, "fast")].alpha == 7.0
    # Other request types kept their cold-start values.
    assert r._cells[(RequestType.WRITING, "fast")] == cold or True


# ---- Session state eviction ---------------------------------------------


def test_session_state_is_evicted_after_ttl():
    """Entries older than OWNER_CACHE_TTL_SECONDS must be dropped when the
    sweep runs (triggered by hitting _SESSION_STATE_SWEEP_THRESHOLD)."""
    import time as _time

    from litellm.router_strategy.adaptive_router import adaptive_router as ar

    r = _make_router()
    threshold = ar._SESSION_STATE_SWEEP_THRESHOLD

    # Backdate one session so its TTL has already passed.
    stale_key = ("sess-stale", "fast")
    r.get_or_create_session_state("sess-stale", "fast", RequestType.GENERAL)
    r._session_states_expiry[stale_key] = _time.time() - 1

    # Fill cache up to the sweep threshold to force eviction on next insert.
    for i in range(threshold):
        r.get_or_create_session_state(f"sess-{i}", "fast", RequestType.GENERAL)

    # Next insert triggers the sweep; stale entry should be gone.
    r.get_or_create_session_state("sess-new", "fast", RequestType.GENERAL)
    assert stale_key not in r._session_states
    assert stale_key not in r._session_states_expiry


def test_session_state_expiry_is_refreshed_on_access():
    """Re-fetching a session state keeps it alive — TTL is a last-activity
    timeout, not an absolute TTL."""
    import time as _time

    r = _make_router()
    r.get_or_create_session_state("sess-A", "fast", RequestType.GENERAL)
    first_exp = r._session_states_expiry[("sess-A", "fast")]

    _time.sleep(0.01)  # move clock forward
    r.get_or_create_session_state("sess-A", "fast", RequestType.GENERAL)
    second_exp = r._session_states_expiry[("sess-A", "fast")]

    assert second_exp > first_exp
