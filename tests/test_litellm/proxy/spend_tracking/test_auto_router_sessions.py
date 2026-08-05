"""Reducing one spend-log payload to a rollup turn.

Bucketing a turn is done by the upsert against the session's own cache record, so it is
covered in tests/proxy_behavior/spend against a real Postgres. What is pure, and covered
here, is deciding whether a request is an auto-routed turn at all and what it contributes.
"""

import datetime as dt
from dataclasses import replace
from types import SimpleNamespace

import pytest

from litellm.proxy.spend_tracking.auto_router_sessions import (
    CACHE_TTL_1H_SECONDS,
    CACHE_TTL_5M_SECONDS,
    AutoRouterSessionQueue,
    TurnFacts,
    build_turn_facts,
    ttl_seconds,
)

T0 = dt.datetime(2026, 8, 3, 12, 0, tzinfo=dt.timezone.utc)


def _payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "api_key": "hashed-key",
        "session_id": "sess-1",
        "model_group": "claude-auto",
        "model": "claude-haiku-4-5",
        "custom_llm_provider": "anthropic",
        "startTime": T0.isoformat(),
        "spend": 0.25,
        "prompt_tokens": 900,
        "completion_tokens": 100,
    }
    return {**base, **overrides}


def _metadata(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "routing_decision": {"router_type": "complexity"},
        "usage_object": {"cache_read_input_tokens": 500, "cache_creation_input_tokens": 0},
    }
    return {**base, **overrides}


def _build(payload=None, metadata=None, **kwargs):
    return build_turn_facts(
        payload=payload if payload is not None else _payload(),
        metadata=metadata if metadata is not None else _metadata(),
        autorouter_savings=kwargs.get("autorouter_savings", 0.75),
        cache_read_tokens=kwargs.get("cache_read_tokens", 500),
        cache_creation_tokens=kwargs.get("cache_creation_tokens", 0),
    )


class TestNotAutoRouted:
    def test_a_request_with_no_routing_decision_is_not_a_turn(self):
        assert _build(metadata={"usage_object": {}}) is None

    def test_a_routing_decision_without_a_kind_is_not_a_turn(self):
        assert _build(metadata=_metadata(routing_decision={})) is None

    @pytest.mark.parametrize("field", ["api_key", "session_id", "model_group", "model"])
    def test_a_turn_missing_any_identity_field_is_dropped(self, field: str):
        assert _build(payload=_payload(**{field: ""})) is None
        assert _build(payload=_payload(**{field: None})) is None

    def test_an_unparseable_start_time_is_dropped(self):
        assert _build(payload=_payload(startTime="not-a-timestamp")) is None


class TestTurnFacts:
    def test_the_router_kind_is_read_from_the_decision_the_router_recorded(self):
        built = _build()
        assert built is not None and built.router_kind == "complexity"

    def test_the_baseline_arm_is_this_turn_plus_what_the_router_saved(self):
        built = _build(autorouter_savings=0.75)
        assert built is not None
        assert (built.spend, built.baseline_spend) == (pytest.approx(0.25), pytest.approx(1.0))

    def test_a_route_that_lost_money_carries_a_baseline_below_what_was_paid(self):
        built = _build(autorouter_savings=-0.10)
        assert built is not None and built.baseline_spend == pytest.approx(0.15)

    def test_tokens_are_the_whole_turn(self):
        built = _build()
        assert built is not None and built.total_tokens == 1000

    def test_a_naive_start_time_is_read_as_utc(self):
        naive = _build(payload=_payload(startTime=T0.replace(tzinfo=None)))
        aware = _build(payload=_payload(startTime=T0))
        assert naive is not None and aware is not None
        assert naive.started_at == aware.started_at == T0.timestamp()

    def test_a_cache_read_is_a_hit(self):
        built = _build(cache_read_tokens=1)
        assert built is not None and built.cache_hit is True

    def test_no_cache_read_is_a_miss(self):
        built = _build(cache_read_tokens=0)
        assert built is not None and built.cache_hit is False


class TestCacheEvidence:
    def test_the_live_prefix_is_what_was_read_plus_what_was_written(self):
        built = _build(cache_read_tokens=500, cache_creation_tokens=200)
        assert built is not None and built.cached_prefix_tokens == 700

    def test_a_turn_that_touched_no_cache_has_no_prefix(self):
        """Coverage keys off this: a model with caching off is absent from the cache view
        rather than counted as a miss."""
        built = _build(cache_read_tokens=0, cache_creation_tokens=0)
        assert built is not None and built.cached_prefix_tokens == 0

    def test_a_cache_read_does_not_guess_which_ttl_created_the_entry(self):
        assert ttl_seconds({"cache_read_input_tokens": 10}) is None
        assert ttl_seconds(None) is None

    def test_five_minute_cache_writes_are_scored_against_the_five_minute_tier(self):
        usage = {
            "cache_creation_input_tokens": 10,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 10},
        }
        assert ttl_seconds(usage) == CACHE_TTL_5M_SECONDS

    def test_one_hour_cache_writes_are_scored_against_the_one_hour_tier(self):
        usage = {
            "cache_creation_input_tokens": 10,
            "cache_creation_token_details": {"ephemeral_1h_input_tokens": 10},
        }
        assert ttl_seconds(usage) == CACHE_TTL_1H_SECONDS

    def test_a_mixed_ttl_write_is_not_collapsed_to_one_ttl(self):
        usage = {
            "cache_creation_input_tokens": 20,
            "cache_creation_token_details": {
                "ephemeral_5m_input_tokens": 10,
                "ephemeral_1h_input_tokens": 10,
            },
        }
        assert ttl_seconds(usage) is None


class TestFlushOrdering:
    @pytest.mark.asyncio
    async def test_turns_apply_in_session_key_order_and_in_time_order_within_a_session(self):
        """Key order means every pod locks rollup rows in the same sequence (no cross-pod
        deadlock); time order within a session is what classification depends on."""
        recorded: list[tuple[object, ...]] = []

        class _DB:
            def batch_(self):
                return self

            async def __aenter__(self):
                return SimpleNamespace(execute_raw=lambda sql, *params: recorded.append(params))

            async def __aexit__(self, *exc: object) -> bool:
                return False

        base = TurnFacts(
            api_key="k",
            session_id="a",
            model_group="g",
            router_kind="complexity",
            baseline_model=None,
            model="m",
            started_at=0.0,
            total_tokens=0,
            spend=0.0,
            baseline_spend=0.0,
            cache_hit=False,
            cache_creation_tokens=0,
            cached_prefix_tokens=0,
            ttl_seconds=None,
        )
        queue = AutoRouterSessionQueue()
        for turn in (
            replace(base, session_id="b", started_at=3.0),
            replace(base, started_at=4.0),
            replace(base, started_at=2.0),
            replace(base, session_id="b", started_at=1.0),
        ):
            queue.update_queue.put_nowait(turn)

        await queue.flush(SimpleNamespace(db=_DB()))

        assert [(params[1], params[6]) for params in recorded] == [("a", 2.0), ("a", 4.0), ("b", 1.0), ("b", 3.0)]
