import random

import pytest

from litellm.router_strategy.adaptive_router.bandit import (
    BanditCell,
    apply_delta,
    initial_cell,
    normalized_cost,
    pick_best,
    score,
    thompson_sample,
)
from litellm.router_strategy.adaptive_router.config import (
    BASE_TIER_WEIGHT,
    COLD_START_MASS,
    SAMPLE_CAP,
    STRENGTH_BONUS,
)
from litellm.types.router import AdaptiveRouterPreferences, RequestType


def test_initial_cell_tier_only():
    prefs = AdaptiveRouterPreferences(quality_tier=2, strengths=[])
    cell = initial_cell(prefs, RequestType.GENERAL)
    expected_mean = BASE_TIER_WEIGHT[2]
    assert abs(cell.mean - expected_mean) < 0.001
    assert abs(cell.alpha + cell.beta - COLD_START_MASS) < 0.001


def test_initial_cell_with_matching_strength():
    prefs = AdaptiveRouterPreferences(
        quality_tier=2, strengths=[RequestType.CODE_GENERATION]
    )
    cell = initial_cell(prefs, RequestType.CODE_GENERATION)
    expected_mean = BASE_TIER_WEIGHT[2] + STRENGTH_BONUS
    assert abs(cell.mean - expected_mean) < 0.001


def test_initial_cell_strength_does_not_apply_to_other_types():
    prefs = AdaptiveRouterPreferences(
        quality_tier=2, strengths=[RequestType.CODE_GENERATION]
    )
    cell = initial_cell(prefs, RequestType.WRITING)
    assert abs(cell.mean - BASE_TIER_WEIGHT[2]) < 0.001


def test_initial_cell_caps_mean_at_0_95():
    prefs = AdaptiveRouterPreferences(
        quality_tier=3, strengths=[RequestType.CODE_GENERATION]
    )
    cell = initial_cell(prefs, RequestType.CODE_GENERATION)
    assert cell.mean <= 0.95


def test_apply_delta_increments_alpha_and_beta():
    cell = BanditCell(alpha=5.0, beta=5.0)
    new_cell = apply_delta(cell, 1.0, 0.0)
    assert new_cell.alpha == 6.0
    assert new_cell.beta == 5.0


def test_apply_delta_respects_sample_cap():
    cell = BanditCell(alpha=SAMPLE_CAP - 1.0, beta=1.0)
    same_cell = apply_delta(cell, 5.0, 5.0)
    assert same_cell.alpha == cell.alpha
    assert same_cell.beta == cell.beta


def test_thompson_sample_in_range():
    cell = BanditCell(alpha=10.0, beta=5.0)
    rng = random.Random(42)
    for _ in range(100):
        s = thompson_sample(cell, rng=rng)
        assert 0.0 <= s <= 1.0


def test_normalized_cost_cheapest_wins():
    assert normalized_cost(0.001, [0.001, 0.005, 0.01]) == 1.0
    assert normalized_cost(0.01, [0.001, 0.005, 0.01]) == 0.0


def test_normalized_cost_no_spread():
    assert normalized_cost(0.005, [0.005, 0.005]) == 0.5


def test_normalized_cost_empty_list():
    assert normalized_cost(0.005, []) == 0.5


def test_score_combines_quality_and_cost():
    s = score(
        quality_sample=1.0,
        model_cost=0.001,
        all_costs=[0.001, 0.01],
        quality_weight=0.7,
        cost_weight=0.3,
    )
    assert abs(s - 1.0) < 0.001


def test_pick_best_empty_dict_raises():
    with pytest.raises(ValueError):
        pick_best({}, {})


def test_thompson_converges_to_better_model():
    """
    LOAD-BEARING TEST. If this regresses, the whole router is broken.

    Setup: 2 models, identical priors, identical cost. Model A's true mean = 0.8,
    Model B's true mean = 0.3. After 200 simulated turns, A must be picked >= 80% of
    last 50 turns.
    """
    rng = random.Random(42)
    cells = {
        "A": BanditCell(alpha=5.0, beta=5.0),
        "B": BanditCell(alpha=5.0, beta=5.0),
    }
    costs = {"A": 0.001, "B": 0.001}
    true_means = {"A": 0.8, "B": 0.3}

    picks = []
    for _ in range(200):
        chosen = pick_best(cells, costs, rng=rng)
        picks.append(chosen)
        outcome = 1.0 if rng.random() < true_means[chosen] else 0.0
        cells[chosen] = apply_delta(cells[chosen], outcome, 1.0 - outcome)

    last_50 = picks[-50:]
    a_share = last_50.count("A") / 50
    assert (
        a_share >= 0.80
    ), f"Expected A to dominate ({a_share=}); priors aren't biasing the sample correctly"
