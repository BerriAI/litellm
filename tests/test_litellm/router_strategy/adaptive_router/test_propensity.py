"""Tests for the selection probability the adaptive router records.

A propensity is a claim: "this model gets chosen this often, in this state." The
tests that matter here check the claim against what the sampler actually does,
because a propensity that does not predict selection frequency is worse than no
propensity at all -- it makes a log look analyzable when it is not.
"""

import random

import pytest

from litellm.router_strategy.adaptive_router.bandit import (
    BanditCell,
    estimate_propensity,
    pick_best,
)

CELLS = {
    "cheap": BanditCell(alpha=5.0, beta=5.0),
    "mid": BanditCell(alpha=6.0, beta=4.0),
    "premium": BanditCell(alpha=7.0, beta=3.0),
}
COSTS = {"cheap": 0.0005, "mid": 0.003, "premium": 0.02}


def _empirical(n: int, seed: int = 0) -> dict:
    """Selection frequency over n independent picks."""
    rng = random.Random(seed)
    counts = {model: 0 for model in CELLS}
    for _ in range(n):
        counts[pick_best(CELLS, COSTS, rng=rng)] += 1
    return {model: count / n for model, count in counts.items()}


def _estimated(n_samples: int, draws: int, seed: int = 1) -> dict:
    """Average logged propensity per model over many decisions."""
    rng = random.Random(seed)
    totals = {model: 0.0 for model in CELLS}
    counts = {model: 0 for model in CELLS}
    for _ in range(draws):
        model = pick_best(CELLS, COSTS, rng=rng)
        propensity = estimate_propensity(
            CELLS, COSTS, model, rng=rng, propensity_samples=n_samples
        )
        totals[model] += propensity
        counts[model] += 1
    return {
        model: (totals[model] / counts[model] if counts[model] else 0.0) for model in CELLS
    }


class TestPropensityIsHonest:
    """The logged number has to describe the sampler that produced it."""

    def test_matches_empirical_selection_frequency(self):
        # The whole claim, checked directly. Off-policy evaluation divides by
        # this number, so a systematic error here biases every downstream
        # estimate without any symptom.
        empirical = _empirical(20_000)
        estimated = _estimated(n_samples=256, draws=4_000)
        for model in CELLS:
            assert estimated[model] == pytest.approx(empirical[model], abs=0.05), (
                f"{model}: logged propensity {estimated[model]:.3f} does not match "
                f"realized selection frequency {empirical[model]:.3f}"
            )

    def test_propensities_across_models_sum_to_one(self):
        # Each model's propensity is P(it wins); over all models that is a
        # distribution.
        rng = random.Random(7)
        seen = {}
        for _ in range(3_000):
            model = pick_best(CELLS, COSTS, rng=rng)
            propensity = estimate_propensity(
                CELLS, COSTS, model, rng=rng, propensity_samples=256
            )
            seen.setdefault(model, []).append(propensity)
        total = sum(sum(v) / len(v) for v in seen.values())
        assert total == pytest.approx(1.0, abs=0.1)

    def test_never_returns_zero(self):
        # The model demonstrably was chosen, so a zero would be
        # self-contradictory -- and an infinite importance weight downstream.
        skewed = {"a": BanditCell(1.0, 40.0), "b": BanditCell(40.0, 1.0)}
        costs = {"a": 0.001, "b": 0.001}
        rng = random.Random(3)
        for _ in range(500):
            model = pick_best(skewed, costs, rng=rng)
            propensity = estimate_propensity(
                skewed, costs, model, rng=rng, propensity_samples=8
            )
            assert propensity >= 1.0 / 9


class TestNoRegression:
    """Existing behaviour is untouched unless a deployment opts in."""

    def test_default_costs_nothing_and_returns_nothing(self):
        # Selection is untouched and no extra draws happen at the default.
        assert estimate_propensity(CELLS, COSTS, "mid") is None

    def test_pick_best_signature_is_unchanged(self):
        # Existing callers and their mocks must keep working.
        assert pick_best(CELLS, COSTS, rng=random.Random(0)) in CELLS

    def test_seeded_runs_are_reproducible(self):
        first = estimate_propensity(
            CELLS, COSTS, "mid", rng=random.Random(5), propensity_samples=32
        )
        second = estimate_propensity(
            CELLS, COSTS, "mid", rng=random.Random(5), propensity_samples=32
        )
        assert first == second

    def test_single_model_is_certain_without_sampling(self):
        assert (
            estimate_propensity(
                {"only": BanditCell(1.0, 1.0)}, {"only": 0.01}, "only", propensity_samples=64
            )
            == 1.0
        )

    def test_rejects_empty_and_negative_inputs(self):
        with pytest.raises(ValueError):
            estimate_propensity({}, {}, "a", propensity_samples=8)
        with pytest.raises(ValueError):
            estimate_propensity(CELLS, COSTS, "mid", propensity_samples=-1)


class TestRoutingDecision:
    """What lands in the spend log."""

    def _router(self, propensity_samples: int):
        from litellm.router_strategy.adaptive_router.adaptive_router import AdaptiveRouter
        from litellm.types.router import (
            AdaptiveRouterConfig,
            AdaptiveRouterPreferences,
            RequestType,
        )

        tiers = {"cheap": 1, "mid": 2, "premium": 3}
        return AdaptiveRouter(
            router_name="adaptive-test",
            config=AdaptiveRouterConfig(
                available_models=list(CELLS),
                propensity_samples=propensity_samples,
            ),
            model_to_prefs={
                model: AdaptiveRouterPreferences(
                    quality_tier=tier, strengths=[RequestType.GENERAL]
                )
                for model, tier in tiers.items()
            },
            model_to_cost=dict(COSTS),
        )

    @pytest.mark.asyncio
    async def test_records_candidate_set_even_when_not_sampling(self):
        from litellm.types.router import RequestType

        router = self._router(propensity_samples=0)
        model, propensity, candidates = await router.pick_model_with_propensity(
            RequestType.GENERAL
        )
        assert model in candidates
        # The candidate set is free and always logged: a decision is only
        # interpretable against the options that existed.
        assert set(candidates) == set(CELLS)
        assert propensity is None

    @pytest.mark.asyncio
    async def test_records_propensity_when_enabled(self):
        from litellm.types.router import RequestType

        router = self._router(propensity_samples=64)
        model, propensity, candidates = await router.pick_model_with_propensity(
            RequestType.GENERAL
        )
        assert model in candidates
        assert propensity is not None
        assert 0.0 < propensity <= 1.0
