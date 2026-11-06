from typing import Final

import pytest
from pydantic import ValidationError

from litellm.models.workflow_definition import WorkflowBundle
from tests.unit.proxy.workflow_definition_rows import bundle


@pytest.mark.parametrize("node_type", ["parallel", "if", "shell", "http"])
def test_v1_contract_rejects_nodes_outside_sequential_chat_and_python(node_type: str) -> None:
    original: Final = bundle().model_dump(mode="json", by_alias=True, exclude_unset=True)
    unsupported: Final = {**original, "nodes": [{**original["nodes"][0], "type": node_type}]}
    with pytest.raises(ValidationError) as rejected:
        WorkflowBundle.model_validate(unsupported)
    assert rejected.value.errors(include_input=False)[0]["type"] == "union_tag_invalid"


def test_binding_cannot_supply_both_literal_and_reference() -> None:
    original: Final = bundle().model_dump(mode="json", by_alias=True, exclude_unset=True)
    ambiguous: Final = {
        **original,
        "nodes": [
            {
                **original["nodes"][0],
                "messages": {"from": "$input", "output": "messages", "value": [{"role": "user", "content": "hello"}]},
            }
        ],
    }
    with pytest.raises(ValidationError) as rejected:
        WorkflowBundle.model_validate(ambiguous)
    assert {
        error["loc"][-1] for error in rejected.value.errors(include_input=False) if "messages" in error["loc"]
    } == {"from", "output", "value"}


def test_nonfinite_values_are_rejected_before_digest_serialization() -> None:
    original: Final = bundle().model_dump(mode="json", by_alias=True, exclude_unset=True)
    with pytest.raises(ValidationError) as rejected:
        WorkflowBundle.model_validate({**original, "input_schema": {"minimum": float("nan")}})
    assert rejected.value.errors(include_input=False)[0]["type"] == "finite_number"
