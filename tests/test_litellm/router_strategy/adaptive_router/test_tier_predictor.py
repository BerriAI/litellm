from typing import Final

import pytest

from litellm.router_strategy.adaptive_router.tier_predictor import (
    TierSuccessPredictor,
    resolve_tier_artifact,
    similarity_cohort,
)
from litellm.types.router import (
    AdaptiveRouterTierArtifact,
    AdaptiveRouterTierCohortStatistic,
    AdaptiveRouterTierDomainStatistic,
    AdaptiveRouterTierGlobalStatistic,
    RequestType,
)


def _artifact(
    global_successes: tuple[float, float, float, float] = (4.0, 5.0, 6.0, 7.0),
    threshold: float = 0.75,
    domain_statistics: tuple[AdaptiveRouterTierDomainStatistic, ...] = (),
    cohort_statistics: tuple[AdaptiveRouterTierCohortStatistic, ...] = (),
) -> AdaptiveRouterTierArtifact:
    return AdaptiveRouterTierArtifact(
        global_statistics=tuple(
            AdaptiveRouterTierGlobalStatistic(tier=tier, successes=successes, observations=10.0)
            for tier, successes in enumerate(global_successes, start=1)
        ),
        domain_statistics=domain_statistics,
        cohort_statistics=cohort_statistics,
        domain_prior_mass=10.0,
        cohort_prior_mass=10.0,
        routing_threshold=threshold,
    )


def test_predictions_are_monotonic_across_tiers() -> None:
    predictor: Final = TierSuccessPredictor(_artifact(global_successes=(9.0, 2.0, 7.0, 6.0)))

    prediction: Final = predictor.predict("hello", RequestType.GENERAL)

    probabilities: Final = tuple(prediction.probabilities.values())
    assert probabilities == tuple(sorted(probabilities))


def test_domain_and_cohort_statistics_back_off_hierarchically() -> None:
    matching_cohort: Final = similarity_cohort("hello", RequestType.GENERAL)
    artifact: Final = _artifact(
        global_successes=(1.0, 5.0, 6.0, 7.0),
        domain_statistics=(
            AdaptiveRouterTierDomainStatistic(
                tier=1,
                request_type=RequestType.GENERAL,
                successes=10.0,
                observations=10.0,
            ),
        ),
        cohort_statistics=(
            AdaptiveRouterTierCohortStatistic(
                tier=1,
                cohort=matching_cohort,
                successes=0.0,
                observations=10.0,
            ),
        ),
    )
    predictor: Final = TierSuccessPredictor(artifact)

    cohort_probability: Final = predictor.predict("hello", RequestType.GENERAL).probabilities[1]
    domain_probability: Final = predictor.predict("hello " * 100, RequestType.GENERAL).probabilities[1]
    global_probability: Final = predictor.predict("hello", RequestType.WRITING).probabilities[1]

    assert cohort_probability == pytest.approx(7.0 / 24.0)
    assert domain_probability == pytest.approx(7.0 / 12.0)
    assert global_probability == pytest.approx(1.0 / 6.0)


def test_selects_first_tier_above_probability_threshold() -> None:
    predictor: Final = TierSuccessPredictor(_artifact(global_successes=(4.0, 6.0, 8.0, 9.0), threshold=0.7))

    prediction: Final = predictor.predict("hello", RequestType.GENERAL)

    assert prediction.required_tier == 3


def test_builtin_ultrafeedback_artifact_is_loadable() -> None:
    artifact: Final = resolve_tier_artifact("ultrafeedback")

    assert artifact.routing_threshold == 0.75
    assert artifact.domain_prior_mass == 200.0
    assert artifact.cohort_prior_mass == 20.0
    assert artifact.datasets[0].license == "MIT"
