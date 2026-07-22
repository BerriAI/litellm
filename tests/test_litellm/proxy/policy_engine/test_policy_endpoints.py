import json

import pytest
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from litellm.proxy.policy_engine.policy_endpoints import (
    _dump_pipeline_result_for_response,
)
from litellm.types.proxy.policy_engine import (
    PipelineExecutionResult,
    PipelineStepResult,
)


class OpaqueSpan:
    __slots__ = ()

    def __str__(self) -> str:
        return "opaque-span"


class NestedPolicyPayload(BaseModel):
    value: str


def test_pipeline_test_response_removes_internal_non_json_values():
    opaque_span = OpaqueSpan()
    result = PipelineExecutionResult(
        terminal_action="allow",
        step_results=[
            PipelineStepResult(
                guardrail_name="pii",
                outcome="pass",
                action_taken="allow",
                modified_data={
                    "metadata": {
                        "parent_otel_span": opaque_span,
                        "litellm_parent_otel_span": opaque_span,
                        "safe": "kept",
                    },
                    "non_internal_object": opaque_span,
                },
            )
        ],
        modified_data={
            "metadata": {
                "parent_otel_span": opaque_span,
                "safe": "kept",
            },
        },
    )

    with pytest.raises(ValueError):
        jsonable_encoder(result.model_dump())

    dumped = _dump_pipeline_result_for_response(result)

    json.dumps(dumped)
    assert dumped["modified_data"]["metadata"] == {"safe": "kept"}
    assert dumped["step_results"][0]["modified_data"]["metadata"] == {"safe": "kept"}
    assert (
        dumped["step_results"][0]["modified_data"]["non_internal_object"]
        == "opaque-span"
    )


def test_pipeline_test_response_preserves_nested_pydantic_models():
    result = PipelineExecutionResult(
        terminal_action="allow",
        step_results=[],
        modified_data={"nested": NestedPolicyPayload(value="kept")},
    )

    dumped = _dump_pipeline_result_for_response(result)

    json.dumps(dumped)
    assert dumped["modified_data"]["nested"] == {"value": "kept"}
