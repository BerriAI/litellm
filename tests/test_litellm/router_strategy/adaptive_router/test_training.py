import random
from pathlib import Path
from typing import Final

from litellm.router_strategy.adaptive_router.bandit import (
    BanditCell,
    apply_evaluation_prior,
    pick_best,
)
from litellm.router_strategy.adaptive_router.training import training_config_fragment


def test_apply_evaluation_prior_preserves_quality_and_caps_mass() -> None:
    cell: Final = apply_evaluation_prior(
        BanditCell(alpha=5.0, beta=5.0),
        successes=90.0,
        failures=10.0,
        max_mass=50.0,
    )

    assert abs(cell.alpha + cell.beta - 50.0) < 0.001
    assert abs(cell.mean - (95.0 / 110.0)) < 0.001


def test_pick_best_uses_posterior_mean_without_exploration() -> None:
    chosen: Final = pick_best(
        cells={
            "reliable": BanditCell(alpha=90.0, beta=10.0),
            "unreliable": BanditCell(alpha=10.0, beta=90.0),
        },
        model_costs={"reliable": 1.0, "unreliable": 1.0},
        exploration_rate=0.0,
        rng=random.Random(42),
    )

    assert chosen == "reliable"


def test_training_config_fragment_aggregates_quality(tmp_path: Path) -> None:
    evaluation_records: Final = tmp_path / "evaluation.jsonl"
    evaluation_records.write_text(
        "\n".join(
            (
                '{"request_type":"code_generation","model":"fast","quality":1}',
                '{"request_type":"code_generation","model":"fast","quality":0.5}',
                '{"request_type":"code_generation","model":"fast","quality":0}',
                '{"request_type":"writing","model":"smart","quality":1}',
            )
        ),
        encoding="utf-8",
    )

    fragment: Final = training_config_fragment(evaluation_records).model_dump(mode="json")

    assert fragment == {
        "evaluation_priors": [
            {
                "request_type": "code_generation",
                "model": "fast",
                "successes": 1.5,
                "failures": 1.5,
            },
            {
                "request_type": "writing",
                "model": "smart",
                "successes": 1.0,
                "failures": 0.0,
            },
        ],
        "exploration_rate": 0.05,
    }
