"""
Tests for pipeline type definitions.
"""

import pytest
from pydantic import ValidationError

from litellm.types.proxy.policy_engine.pipeline_types import (
    GuardrailPipeline,
    PipelineExecutionResult,
    PipelineStep,
    PipelineStepResult,
)
from litellm.types.proxy.policy_engine.policy_types import (
    Policy,
    PolicyGuardrails,
)


def test_pipeline_step_defaults():
    step = PipelineStep(guardrail="my-guard")
    assert step.on_fail == "block"
    assert step.on_pass == "allow"
    assert step.pass_data is False
    assert step.modify_response_message is None


def test_pipeline_step_valid_actions():
    step = PipelineStep(guardrail="my-guard", on_fail="next", on_pass="next")
    assert step.on_fail == "next"
    assert step.on_pass == "next"


def test_pipeline_step_all_action_types():
    for action in ("allow", "block", "next", "modify_response"):
        step = PipelineStep(guardrail="g", on_fail=action, on_pass=action)
        assert step.on_fail == action
        assert step.on_pass == action


def test_pipeline_step_invalid_action_rejected():
    with pytest.raises(ValidationError):
        PipelineStep(guardrail="my-guard", on_fail="invalid_action")


def test_pipeline_step_invalid_on_pass_rejected():
    with pytest.raises(ValidationError):
        PipelineStep(guardrail="my-guard", on_pass="skip")


def test_pipeline_requires_at_least_one_step():
    with pytest.raises(ValidationError):
        GuardrailPipeline(mode="pre_call", steps=[])


def test_pipeline_invalid_mode_rejected():
    with pytest.raises(ValidationError):
        GuardrailPipeline(
            mode="during_call",
            steps=[PipelineStep(guardrail="g")],
        )


def test_pipeline_valid_modes():
    for mode in ("pre_call", "post_call"):
        pipeline = GuardrailPipeline(
            mode=mode,
            steps=[PipelineStep(guardrail="g")],
        )
        assert pipeline.mode == mode


def test_pipeline_with_multiple_steps():
    pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[
            PipelineStep(guardrail="g1", on_fail="next", on_pass="allow"),
            PipelineStep(guardrail="g2", on_fail="block", on_pass="allow"),
        ],
    )
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0].guardrail == "g1"
    assert pipeline.steps[1].guardrail == "g2"


def test_policy_with_pipeline_parses():
    policy = Policy(
        guardrails=PolicyGuardrails(add=["g1", "g2"]),
        pipeline=GuardrailPipeline(
            mode="pre_call",
            steps=[
                PipelineStep(guardrail="g1", on_fail="next"),
                PipelineStep(guardrail="g2"),
            ],
        ),
    )
    assert policy.pipeline is not None
    assert len(policy.pipeline.steps) == 2


def test_policy_without_pipeline():
    policy = Policy(
        guardrails=PolicyGuardrails(add=["g1"]),
    )
    assert policy.pipeline is None


def test_pipeline_step_result():
    result = PipelineStepResult(
        guardrail_name="g1",
        outcome="fail",
        action_taken="next",
        error_detail="Content policy violation",
        duration_seconds=0.05,
    )
    assert result.outcome == "fail"
    assert result.action_taken == "next"


def test_pipeline_execution_result():
    result = PipelineExecutionResult(
        terminal_action="block",
        step_results=[
            PipelineStepResult(
                guardrail_name="g1",
                outcome="fail",
                action_taken="next",
            ),
            PipelineStepResult(
                guardrail_name="g2",
                outcome="fail",
                action_taken="block",
            ),
        ],
        error_message="Content blocked",
    )
    assert result.terminal_action == "block"
    assert len(result.step_results) == 2


def test_pipeline_step_extra_fields_rejected():
    with pytest.raises(ValidationError):
        PipelineStep(guardrail="g", unknown_field="value")


def test_pipeline_extra_fields_rejected():
    with pytest.raises(ValidationError):
        GuardrailPipeline(
            mode="pre_call",
            steps=[PipelineStep(guardrail="g")],
            unknown="value",
        )
