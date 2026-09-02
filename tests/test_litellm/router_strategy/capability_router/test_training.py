import json
from pathlib import Path

import pytest

from litellm.router_strategy.capability_router.config import CapabilityRouterConfig
from litellm.router_strategy.capability_router.training import (
    CapabilityTrainingRecord,
    main,
    train_capability_artifact,
)


def training_config() -> CapabilityRouterConfig:
    return CapabilityRouterConfig.model_validate(
        {
            "candidates": [
                {
                    "model": "small",
                    "description": "Efficient model with bounded-task and open-ended-task rules",
                    "rules": [
                        {"boundary": "uncertain", "rule": "The task has an executable correctness check"},
                        {"boundary": "uncertain", "rule": "The task requires resolving hidden behavior"},
                    ],
                },
                {
                    "model": "strong",
                    "description": "Capable fallback model",
                    "rules": [
                        {"boundary": "supported", "rule": "The task has an executable correctness check"},
                        {"boundary": "supported", "rule": "The task requires resolving hidden behavior"},
                    ],
                },
            ],
            "classifier": {"model": "judge"},
            "probability_threshold": 0.5,
            "threshold_step": 0.0,
            "fallback_model": "strong",
        }
    )


def records() -> tuple[CapabilityTrainingRecord, ...]:
    split_sizes = {"train": 20, "validation": 4, "test": 4}
    return tuple(
        CapabilityTrainingRecord(
            benchmark="agent-bench",
            task_id=f"{split}-{difficulty}-{index}",
            split=split,
            model=model,
            primary_rule="R1" if difficulty == "bounded" else "R2",
            raw_p_solve=(0.9 if difficulty == "bounded" else 0.6) if model == "small" else 0.9,
            success=1.0 if model == "strong" or difficulty == "bounded" else 0.0,
            estimated_cost=1.0 if model == "small" else 10.0,
        )
        for split, count in split_sizes.items()
        for difficulty in ("bounded", "hidden")
        for index in range(count)
        for model in ("small", "strong")
    )


def test_training_learns_rule_boundaries_and_improves_held_out_routing() -> None:
    result = train_capability_artifact(records(), training_config())
    small = result.artifact.config.candidates[0]

    assert [rule.boundary for rule in small.rules] == ["supported", "unsupported"]
    assert [rule.observed_success_probability for rule in small.rules] == pytest.approx([21.0 / 22.0, 1.0 / 22.0])
    assert small.probability_calibration[-1].upper_bound == 1.0
    assert result.artifact.records == len(records())
    assert len(result.artifact.records_sha256) == 64
    assert result.report.test.success_rate == 1.0
    assert result.report.test_untrained.success_rate == 0.5
    assert result.report.test.quality_cost_utility > result.report.test_untrained.quality_cost_utility
    assert result.report.test_probability_calibrated.brier < result.report.test_probability_raw.brier
    assert result.report.test_always_candidates["small"].success_rate == 0.5
    assert result.report.test_always_candidates["strong"].success_rate == 1.0
    assert result.report.test_oracle.quality_cost_utility >= result.report.test.quality_cost_utility
    assert result.report.test_threshold_sweep


def test_training_requires_explicit_splits_and_exact_candidate_models() -> None:
    only_training = tuple(record for record in records() if record.split == "train")
    with pytest.raises(ValueError, match="train, validation, and test"):
        train_capability_artifact(only_training, training_config())

    without_strong = tuple(record for record in records() if record.model == "small")
    with pytest.raises(ValueError, match="exactly match"):
        train_capability_artifact(without_strong, training_config())


def test_training_rejects_a_task_that_crosses_splits() -> None:
    source = records()
    crossed = (*source, source[0].model_copy(update={"split": "test"}))

    with pytest.raises(ValueError, match="must not cross splits"):
        train_capability_artifact(crossed, training_config())


def test_training_rejects_a_task_family_that_crosses_splits() -> None:
    source = records()
    crossed = (
        source[0].model_copy(update={"task_family": "shared-repository"}),
        source[1].model_copy(update={"task_id": "different-task", "task_family": "shared-repository", "split": "test"}),
        *source[2:],
    )

    with pytest.raises(ValueError, match="task family must not cross splits"):
        train_capability_artifact(crossed, training_config())


def test_training_cli_writes_a_ready_to_use_artifact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    records_path = tmp_path / "outcomes.jsonl"
    config_path = tmp_path / "config.json"
    artifact_path = tmp_path / "artifact.json"
    records_path.write_text("".join(record.model_dump_json() + "\n" for record in records()))
    config_path.write_text(training_config().model_dump_json())

    exit_code = main(
        (
            str(records_path),
            "--config",
            str(config_path),
            "--artifact-output",
            str(artifact_path),
        )
    )

    artifact = json.loads(artifact_path.read_text())
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert artifact["config"]["candidates"][0]["probability_calibration"]
    assert report["test"]["success_rate"] == 1.0
