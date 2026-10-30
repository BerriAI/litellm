import random

import pytest

from litellm.router_strategy.complexity_router.config import ComplexityTier
from litellm.router_strategy.hybrid_router import HybridRouter, HybridRouterConfig
from litellm.router_strategy.hybrid_router.bandit import (
    EpsilonGreedy,
    ThompsonSampling,
    UCB,
    make_bandit,
)


TIER_CANDIDATES = {
    "SIMPLE": ("model-small", "model-medium"),
    "MEDIUM": ("model-medium", "model-large"),
    "COMPLEX": ("model-large",),
    "REASONING": ("model-large",),
}


@pytest.fixture
def config() -> HybridRouterConfig:
    return HybridRouterConfig(tier_candidates=TIER_CANDIDATES, bandit="thompson")


@pytest.fixture
def router(config: HybridRouterConfig) -> HybridRouter:
    return HybridRouter(config)


class TestMakeBandit:
    def test_thompson(self):
        bandit = make_bandit("thompson", ("a", "b"))
        assert isinstance(bandit, ThompsonSampling)

    def test_ucb(self):
        bandit = make_bandit("ucb", ("a", "b"))
        assert isinstance(bandit, UCB)

    def test_epsilon_greedy(self):
        bandit = make_bandit("epsilon_greedy", ("a", "b"))
        assert isinstance(bandit, EpsilonGreedy)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown bandit"):
            make_bandit("nonexistent", ("a", "b"))

    def test_single_arm_raises(self):
        with pytest.raises(ValueError, match="at least 2 arms"):
            make_bandit("thompson", ("only-one",))


class TestUCB:
    def test_unexplored_arm_picked_first(self):
        bandit = UCB(("a", "b"), delta=0.1)
        first = bandit.select_arm()
        assert first in ("a", "b")

    def test_explored_arms_use_ucb_score(self):
        bandit = UCB(("a", "b"), delta=0.1)
        bandit.update("a", 1.0)
        bandit.update("b", 0.0)
        bandit.update("a", 1.0)
        bandit.update("b", 0.0)
        assert bandit.select_arm() == "a"

    def test_select_arm_from_respects_eligible(self):
        bandit = UCB(("a", "b", "c"), delta=0.1)
        for _ in range(10):
            bandit.update("a", 1.0)
            bandit.update("b", 0.0)
            bandit.update("c", 0.5)
        picked = bandit.select_arm_from(("b", "c"))
        assert picked in ("b", "c")

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta must be in"):
            UCB(("a", "b"), delta=0.0)

    def test_state_tracks_counts(self):
        bandit = UCB(("a", "b"), delta=0.1)
        bandit.update("a", 0.8)
        bandit.update("a", 0.6)
        bandit.update("b", 0.4)
        state = bandit.state()
        assert state["counts"]["a"] == 2
        assert state["counts"]["b"] == 1
        assert abs(state["means"]["a"] - 0.7) < 1e-9


class TestEpsilonGreedy:
    def test_unexplored_arm_picked_first(self):
        bandit = EpsilonGreedy(("a", "b"), epsilon=0.1)
        first = bandit.select_arm()
        assert first in ("a", "b")

    def test_greedy_picks_best_after_exploration(self):
        bandit = EpsilonGreedy(("a", "b"), epsilon=0.0)
        bandit.update("a", 1.0)
        bandit.update("b", 0.0)
        bandit.update("a", 1.0)
        bandit.update("b", 0.0)
        assert bandit.select_arm() == "a"

    def test_full_exploration_picks_randomly(self):
        random.seed(42)
        bandit = EpsilonGreedy(("a", "b"), epsilon=1.0)
        bandit.update("a", 1.0)
        bandit.update("b", 0.0)
        picks = {bandit.select_arm() for _ in range(50)}
        assert picks == {"a", "b"}

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon must be in"):
            EpsilonGreedy(("a", "b"), epsilon=1.5)

    def test_state_tracks_counts(self):
        bandit = EpsilonGreedy(("a", "b"), epsilon=0.1)
        bandit.update("a", 0.5)
        bandit.update("b", 0.3)
        state = bandit.state()
        assert state["counts"]["a"] == 1
        assert state["counts"]["b"] == 1
        assert state["algorithm"] == "EpsilonGreedy"


class TestThompsonSampling:
    def test_selects_from_arms(self):
        bandit = ThompsonSampling(("a", "b"))
        assert bandit.select_arm() in ("a", "b")

    def test_update_shifts_posterior(self):
        bandit = ThompsonSampling(("a", "b"))
        for _ in range(20):
            bandit.update("a", 1.0)
            bandit.update("b", 0.0)
        state = bandit.state()
        assert state["means"]["a"] > state["means"]["b"]

    def test_custom_priors(self):
        bandit = ThompsonSampling(("a", "b"), arm_priors={"a": (10.0, 1.0), "b": (1.0, 10.0)})
        state = bandit.state()
        assert state["alpha"]["a"] == 10.0
        assert state["beta"]["b"] == 10.0

    def test_select_arm_from_respects_eligible(self):
        bandit = ThompsonSampling(("a", "b", "c"))
        for _ in range(20):
            bandit.update("a", 1.0)
            bandit.update("b", 0.0)
            bandit.update("c", 0.5)
        picked = bandit.select_arm_from(("b", "c"))
        assert picked in ("b", "c")

    def test_state_reports_algorithm(self):
        bandit = ThompsonSampling(("a", "b"))
        assert bandit.state()["algorithm"] == "ThompsonSampling"


class TestClassification:
    def test_simple_greeting(self, router: HybridRouter):
        tier, score, signals = router.classify("hello")
        assert tier == ComplexityTier.SIMPLE

    def test_code_request(self, router: HybridRouter):
        tier, score, signals = router.classify(
            "Write a python function to implement a distributed database query with async error handling"
        )
        assert tier in (ComplexityTier.MEDIUM, ComplexityTier.COMPLEX, ComplexityTier.REASONING)
        assert score > 0

    def test_reasoning_keywords_trigger_reasoning_tier(self, router: HybridRouter):
        tier, score, signals = router.classify(
            "Think through step by step and reason through the chain of thought for this problem"
        )
        assert tier == ComplexityTier.REASONING

    def test_short_prompt_gets_simple_tier(self, router: HybridRouter):
        tier, score, signals = router.classify("hi")
        assert tier == ComplexityTier.SIMPLE

    def test_long_prompt_boosts_score(self, router: HybridRouter):
        long_prompt = "Explain " + "the architecture of distributed systems " * 50
        tier, score, _ = router.classify(long_prompt)
        assert score > 0

    def test_multi_step_pattern_detected(self, router: HybridRouter):
        tier, score, _ = router.classify("First do X, then do Y. Step 1: read the file. Step 2: parse it.")
        assert score > 0

    def test_simple_keywords_reduce_score(self, router: HybridRouter):
        tier, score, _ = router.classify("What is the definition of hello?")
        assert tier == ComplexityTier.SIMPLE


class TestGetCandidates:
    def test_returns_tier_candidates(self, router: HybridRouter):
        candidates = router.get_candidates(ComplexityTier.SIMPLE)
        assert candidates == ("model-small", "model-medium")

    def test_returns_complex_candidates(self, router: HybridRouter):
        candidates = router.get_candidates(ComplexityTier.COMPLEX)
        assert candidates == ("model-large",)

    def test_fallback_when_tier_missing(self):
        config = HybridRouterConfig(
            tier_candidates={"SIMPLE": ("model-small",)},
            bandit="thompson",
        )
        router = HybridRouter(config)
        candidates = router.get_candidates(ComplexityTier.COMPLEX)
        assert len(candidates) > 0


class TestPickModel:
    def test_single_candidate_returns_it(self, router: HybridRouter):
        model = router.pick_model(ComplexityTier.COMPLEX)
        assert model == "model-large"

    def test_multi_candidate_returns_valid_model(self, router: HybridRouter):
        model = router.pick_model(ComplexityTier.SIMPLE)
        assert model in ("model-small", "model-medium")


class TestRoute:
    def test_returns_tier_and_model(self, router: HybridRouter):
        tier, model = router.route("hello")
        assert isinstance(tier, ComplexityTier)
        assert model in ("model-small", "model-medium", "model-large")

    def test_simple_routes_to_simple_candidates(self, router: HybridRouter):
        tier, model = router.route("hi")
        assert tier == ComplexityTier.SIMPLE
        assert model in ("model-small", "model-medium")


class TestRewardRecording:
    def test_record_success_returns_reward_in_range(self, router: HybridRouter):
        reward = router.record_success(ComplexityTier.SIMPLE, "model-small", latency_seconds=0.5, completion_tokens=10)
        assert 0.0 < reward <= 1.0

    def test_record_success_updates_bandit_state(self, router: HybridRouter):
        state_before = router.state()
        count_before = state_before["tier_bandits"]["SIMPLE"]["counts"]["model-small"]

        router.record_success(ComplexityTier.SIMPLE, "model-small", latency_seconds=0.5, completion_tokens=10)

        state_after = router.state()
        count_after = state_after["tier_bandits"]["SIMPLE"]["counts"]["model-small"]
        assert count_after == count_before + 1

    def test_record_success_no_bandit_for_single_candidate_tier(self, router: HybridRouter):
        reward = router.record_success(ComplexityTier.COMPLEX, "model-large", latency_seconds=0.5, completion_tokens=10)
        assert 0.0 < reward <= 1.0
        assert "COMPLEX" not in router.state()["tier_bandits"]

    def test_record_failure_gives_zero_reward(self, router: HybridRouter):
        router.record_failure(ComplexityTier.SIMPLE, "model-small")
        state = router.state()
        assert state["tier_bandits"]["SIMPLE"]["counts"]["model-small"] == 1

    def test_record_quality_blends_efficiency_and_quality(self, router: HybridRouter):
        reward_high_quality = router.record_quality(
            ComplexityTier.SIMPLE,
            "model-small",
            latency_seconds=0.5,
            completion_tokens=10,
            quality_score=1.0,
            quality_weight=0.5,
        )
        router2 = HybridRouter(HybridRouterConfig(tier_candidates=TIER_CANDIDATES, bandit="thompson"))
        reward_low_quality = router2.record_quality(
            ComplexityTier.SIMPLE,
            "model-small",
            latency_seconds=0.5,
            completion_tokens=10,
            quality_score=0.0,
            quality_weight=0.5,
        )
        assert reward_high_quality > reward_low_quality

    def test_record_quality_weight_zero_equals_efficiency_only(self, router: HybridRouter):
        reward_eff = router.record_success(
            ComplexityTier.SIMPLE, "model-small", latency_seconds=0.5, completion_tokens=10
        )
        router2 = HybridRouter(HybridRouterConfig(tier_candidates=TIER_CANDIDATES, bandit="thompson"))
        reward_qw0 = router2.record_quality(
            ComplexityTier.SIMPLE,
            "model-small",
            latency_seconds=0.5,
            completion_tokens=10,
            quality_score=0.0,
            quality_weight=0.0,
        )
        assert abs(reward_eff - reward_qw0) < 1e-9

    def test_record_quality_weight_one_equals_quality_score(self):
        router = HybridRouter(HybridRouterConfig(tier_candidates=TIER_CANDIDATES, bandit="thompson"))
        reward = router.record_quality(
            ComplexityTier.SIMPLE,
            "model-small",
            latency_seconds=100.0,
            completion_tokens=1,
            quality_score=0.9,
            quality_weight=1.0,
        )
        assert abs(reward - 0.9) < 1e-9


class TestBanditConvergence:
    def test_thompson_converges_to_faster_arm(self):
        router = HybridRouter(HybridRouterConfig(tier_candidates=TIER_CANDIDATES, bandit="thompson"))
        for _ in range(100):
            router.record_success(ComplexityTier.SIMPLE, "model-small", latency_seconds=0.1, completion_tokens=50)
            router.record_success(ComplexityTier.SIMPLE, "model-medium", latency_seconds=1.0, completion_tokens=50)
        picks = [router.pick_model(ComplexityTier.SIMPLE) for _ in range(50)]
        assert picks.count("model-small") > 30

    def test_ucb_converges_to_faster_arm(self):
        router = HybridRouter(HybridRouterConfig(tier_candidates=TIER_CANDIDATES, bandit="ucb"))
        for _ in range(100):
            router.record_success(ComplexityTier.SIMPLE, "model-small", latency_seconds=0.1, completion_tokens=50)
            router.record_success(ComplexityTier.SIMPLE, "model-medium", latency_seconds=1.0, completion_tokens=50)
        picks = [router.pick_model(ComplexityTier.SIMPLE) for _ in range(50)]
        assert picks.count("model-small") > 30

    def test_epsilon_greedy_converges_to_faster_arm(self):
        router = HybridRouter(HybridRouterConfig(tier_candidates=TIER_CANDIDATES, bandit="epsilon_greedy"))
        for _ in range(100):
            router.record_success(ComplexityTier.SIMPLE, "model-small", latency_seconds=0.1, completion_tokens=50)
            router.record_success(ComplexityTier.SIMPLE, "model-medium", latency_seconds=1.0, completion_tokens=50)
        picks = [router.pick_model(ComplexityTier.SIMPLE) for _ in range(50)]
        assert picks.count("model-small") > 30


class TestTierPriors:
    def test_priors_bias_initial_selection(self):
        config = HybridRouterConfig(
            tier_candidates=TIER_CANDIDATES,
            bandit="thompson",
            tier_priors={"SIMPLE": {"model-small": (20.0, 1.0), "model-medium": (1.0, 20.0)}},
        )
        router = HybridRouter(config)
        picks = [router.pick_model(ComplexityTier.SIMPLE) for _ in range(50)]
        assert picks.count("model-small") > 40


class TestState:
    def test_state_includes_config(self, router: HybridRouter):
        state = router.state()
        assert "config" in state
        assert state["config"]["bandit"] == "thompson"

    def test_state_includes_tier_bandits(self, router: HybridRouter):
        state = router.state()
        assert "tier_bandits" in state
        assert "SIMPLE" in state["tier_bandits"]
        assert "MEDIUM" in state["tier_bandits"]
        assert "COMPLEX" not in state["tier_bandits"]


class TestHybridRouterConfig:
    def test_frozen_config(self):
        config = HybridRouterConfig(tier_candidates=TIER_CANDIDATES)
        with pytest.raises(AttributeError):
            config.bandit = "ucb"

    def test_default_bandit_is_thompson(self):
        config = HybridRouterConfig(tier_candidates=TIER_CANDIDATES)
        assert config.bandit == "thompson"

    def test_custom_keywords_override_defaults(self):
        config = HybridRouterConfig(tier_candidates=TIER_CANDIDATES, code_keywords=("custom_kw",))
        router = HybridRouter(config)
        assert router._code_keywords == ("custom_kw",)
