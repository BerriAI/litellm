from typing import Final

from litellm.router_strategy.adaptive_router.tier_training import (
    TierTrainingRecord,
    stable_split,
    train_tier_artifact,
)
from litellm.types.router import AdaptiveRouterTierDataset


def test_training_tunes_complete_artifact_on_hash_splits() -> None:
    records: Final = tuple(
        TierTrainingRecord(prompt=f"Evaluate request {prompt_index}", tier=tier, success=float(tier >= 3))
        for prompt_index in range(100)
        for tier in range(1, 5)
    )
    dataset: Final = AdaptiveRouterTierDataset(
        name="fixture",
        url="https://example.com/fixture",
        license="MIT",
        rows=len(records),
    )

    result: Final = train_tier_artifact(records, dataset)

    assert {stat.tier for stat in result.artifact.global_statistics} == {1, 2, 3, 4}
    assert result.artifact.domain_prior_mass > 0
    assert result.artifact.cohort_prior_mass > 0
    assert 0.0 <= result.artifact.routing_threshold <= 1.0
    assert result.report.router_benchmark.test.prompts > 0
    assert {stable_split(f"Evaluate request {index}") for index in range(100)} == {
        "train",
        "validation",
        "test",
    }
