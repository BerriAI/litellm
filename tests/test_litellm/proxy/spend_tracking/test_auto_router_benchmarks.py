import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest

import litellm
from litellm.proxy.spend_tracking.auto_router_benchmarks import (
    BENCHMARKS_MAX_WINDOW_DAYS,
    _RoutedModelSpend,
    _SessionRow,
    _baseline_rates,
    _clamp_window,
    _derive_baseline_model,
    _pricing_candidates,
    auto_router_groups,
    compute_benchmarks,
    summarize_group,
)


def _rates(model: str) -> tuple[float, float]:
    info = litellm.get_model_info(model=model)
    return float(info["input_cost_per_token"] or 0.0), float(info["output_cost_per_token"] or 0.0)


def _session(session_id: str, turns: int, secs: float, p_tok: int, c_tok: int, spend: float) -> _SessionRow:
    return _SessionRow(
        session_id=session_id,
        turns=turns,
        session_length_seconds=secs,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=p_tok + c_tok,
        actual_spend=spend,
    )


def test_baseline_prices_provider_prefixed_model():
    # SpendLogs stores models provider-prefixed while the cost map keys them bare;
    # both must resolve to the same non-zero rates or every savings number is wrong.
    prefixed = _baseline_rates("anthropic/claude-opus-5")
    bare = _baseline_rates("claude-opus-5")
    assert prefixed.input_cost_per_token > 0
    assert prefixed.output_cost_per_token > 0
    assert prefixed.input_cost_per_token == bare.input_cost_per_token
    assert prefixed.output_cost_per_token == bare.output_cost_per_token


def test_baseline_falls_open_to_zero_for_unknown_model():
    rates = _baseline_rates("provider/model-that-does-not-exist")
    assert rates.input_cost_per_token == 0.0
    assert rates.output_cost_per_token == 0.0


def test_pricing_candidates_include_prefixed_and_stripped():
    assert _pricing_candidates("anthropic/claude-opus-5") == ("anthropic/claude-opus-5", "claude-opus-5")
    assert _pricing_candidates("claude-opus-5") == ("claude-opus-5",)


def test_counterfactual_uses_baseline_rates_on_actual_tokens():
    in_cost, out_cost = _rates("claude-opus-5")
    sessions = (_session("s1", turns=3, secs=100.0, p_tok=1000, c_tok=400, spend=0.002),)
    routed = (_RoutedModelSpend("anthropic/claude-haiku-4-5", 1000, 400),)
    bench = summarize_group("auto", "complexity", sessions, routed, configured_baseline="claude-opus-5")
    assert bench is not None
    expected_baseline_spend = 1000 * in_cost + 400 * out_cost
    assert bench.baseline_spend == pytest.approx(expected_baseline_spend)
    assert bench.savings == pytest.approx(expected_baseline_spend - 0.002)
    assert bench.savings_pct == pytest.approx(100.0 * (expected_baseline_spend - 0.002) / expected_baseline_spend)


def test_session_metrics_are_averaged_across_sessions():
    sessions = (
        _session("s1", turns=10, secs=600.0, p_tok=800, c_tok=200, spend=0.001),
        _session("s2", turns=4, secs=200.0, p_tok=400, c_tok=100, spend=0.001),
    )
    routed = (_RoutedModelSpend("anthropic/claude-haiku-4-5", 1200, 300),)
    bench = summarize_group("auto", "complexity", sessions, routed, configured_baseline="claude-opus-5")
    assert bench is not None
    assert bench.sessions == 2
    assert bench.turns == 14
    assert bench.avg_turns_per_session == pytest.approx(7.0)
    assert bench.avg_session_length_seconds == pytest.approx(400.0)
    assert bench.total_tokens == 1500
    assert bench.avg_tokens_per_session == pytest.approx(750.0)


def test_configured_baseline_overrides_derived():
    sessions = (_session("s1", turns=1, secs=10.0, p_tok=100, c_tok=100, spend=0.0001),)
    routed = (_RoutedModelSpend("anthropic/claude-haiku-4-5", 100, 100),)
    bench = summarize_group("auto", "complexity", sessions, routed, configured_baseline="claude-opus-5")
    assert bench is not None
    assert bench.baseline_model == "claude-opus-5"


def test_derived_baseline_is_priciest_routed_model_not_highest_volume():
    # Haiku emits far more tokens but opus is the pricier model; the baseline must
    # be chosen on price, not token volume, or a cheap high-traffic model would be
    # mislabelled the flagship and collapse the savings figure.
    routed = (
        _RoutedModelSpend("anthropic/claude-haiku-4-5", 1_000_000, 1_000_000),
        _RoutedModelSpend("anthropic/claude-opus-5", 5, 5),
    )
    assert _derive_baseline_model(routed) == "anthropic/claude-opus-5"


def test_empty_session_group_is_omitted():
    routed = (_RoutedModelSpend("anthropic/claude-opus-5", 10, 10),)
    assert summarize_group("auto", "complexity", (), routed, configured_baseline=None) is None


def test_group_omitted_when_baseline_cannot_be_priced():
    # No configured baseline and no priceable routed model means there is no
    # honest counterfactual, so the group is dropped rather than reported at zero.
    sessions = (_session("s1", turns=1, secs=10.0, p_tok=100, c_tok=100, spend=0.0001),)
    routed = (_RoutedModelSpend("provider/unpriced-model", 100, 100),)
    assert summarize_group("auto", "complexity", sessions, routed, configured_baseline=None) is None


def test_savings_pct_zero_when_baseline_unpriced_but_configured():
    # A configured baseline that the cost map cannot price yields a zero baseline
    # spend; savings_pct must not divide by zero.
    sessions = (_session("s1", turns=1, secs=10.0, p_tok=100, c_tok=100, spend=0.05),)
    routed = (_RoutedModelSpend("anthropic/claude-haiku-4-5", 100, 100),)
    bench = summarize_group("auto", "complexity", sessions, routed, configured_baseline="provider/unpriced")
    assert bench is not None
    assert bench.baseline_spend == 0.0
    assert bench.savings_pct == 0.0


def test_window_clamped_to_max_days():
    _, _, served_start, served_end = _clamp_window("2026-01-01", "2026-07-29")
    assert served_end == "2026-07-29"
    # start is pulled forward to end - BENCHMARKS_MAX_WINDOW_DAYS
    assert served_start == "2026-06-29"
    assert (
        __import__("datetime").date.fromisoformat(served_end) - __import__("datetime").date.fromisoformat(served_start)
    ).days == BENCHMARKS_MAX_WINDOW_DAYS


def test_window_within_cap_is_preserved():
    start, _, served_start, served_end = _clamp_window("2026-07-20", "2026-07-29")
    assert served_start == "2026-07-20"
    assert served_end == "2026-07-29"


class _FakeDB:
    def __init__(self, session_rows: list[dict[str, object]], model_rows: list[dict[str, object]]) -> None:
        self._session_rows = session_rows
        self._model_rows = model_rows
        self.queries: list[tuple[object, ...]] = []

    async def query_raw(self, sql: str, *params: object) -> list[dict[str, object]]:
        self.queries.append(params)
        return self._session_rows if "GROUP BY session_id" in sql else self._model_rows


class _FakePrisma:
    def __init__(self, db: _FakeDB) -> None:
        self.db = db


@pytest.mark.asyncio
async def test_compute_benchmarks_materializes_and_prices_rows_end_to_end():
    # Guards the two integration bugs the live rig caught: a tuple(await ...) that
    # produced an async generator instead of results, and untyped date params that
    # made Postgres compare timestamp >= text. This drives the real query ->
    # validate -> fold path with an injected fake DB returning dicts, the shape
    # prisma query_raw yields.
    session_rows = [
        {
            "session_id": "s1",
            "turns": 3,
            "session_length_seconds": 120.0,
            "prompt_tokens": 1000,
            "completion_tokens": 400,
            "total_tokens": 1400,
            "actual_spend": 0.002,
        },
        {
            "session_id": "s2",
            "turns": 1,
            "session_length_seconds": None,
            "prompt_tokens": 500,
            "completion_tokens": 100,
            "total_tokens": 600,
            "actual_spend": 0.001,
        },
    ]
    model_rows = [{"model": "anthropic/claude-haiku-4-5", "prompt_tokens": 1500, "completion_tokens": 500}]
    db = _FakeDB(session_rows, model_rows)

    response = await compute_benchmarks(
        _FakePrisma(db),
        groups=(("auto", "complexity", "claude-opus-5"),),
        start_date="2026-06-29",
        end_date="2026-07-29",
    )

    assert len(response.groups) == 1
    bench = response.groups[0]
    assert bench.sessions == 2
    assert bench.turns == 4
    in_cost, out_cost = _rates("claude-opus-5")
    expected_baseline = (1000 + 500) * in_cost + (400 + 100) * out_cost
    assert bench.baseline_spend == pytest.approx(expected_baseline)
    assert bench.savings == pytest.approx(expected_baseline - 0.003)
    # a null session_length must fold as zero, not crash
    assert bench.avg_session_length_seconds == pytest.approx(60.0)
    # date params reach the DB as ISO strings so the ::timestamptz cast applies
    assert all(isinstance(p, str) for query in db.queries for p in query[1:])


def test_auto_router_groups_enumerates_only_auto_routers_with_baseline_override():
    class _FakeRouter:
        model_list = [
            {
                "model_name": "auto",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "benchmark_baseline_model": "claude-opus-5",
                },
            },
            {"model_name": "smart", "litellm_params": {"model": "auto_router/my-semantic-router"}},
            {"model_name": "claude-sonnet", "litellm_params": {"model": "anthropic/claude-sonnet-5"}},
        ]

    groups = auto_router_groups(_FakeRouter())
    assert ("auto", "complexity", "claude-opus-5") in groups
    assert ("smart", "semantic", None) in groups
    # a plain provider deployment is not an auto-router and must not appear
    assert all(model_group != "claude-sonnet" for model_group, _, _ in groups)
