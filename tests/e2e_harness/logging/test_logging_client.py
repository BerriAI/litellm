import json
from typing import Final

import pytest
from pydantic import JsonValue

from logging_client import LangfuseObservation, is_otel_v2_generation, observation_spend


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("cost", [0.0, 0.000023])
def test_observation_preserves_v2_content_usage_and_cost(encoded: bool, cost: float) -> None:
    content: Final[JsonValue] = [{"role": "user", "content": "unique-request-marker"}]
    output: Final[JsonValue] = [{"role": "assistant", "content": "response"}]
    metadata: Final[JsonValue] = {
        "attributes": {
            "langfuse.observation.type": "generation",
            "litellm.metadata.user_api_key_alias": "test-key-alias",
        }
    }
    usage: Final[JsonValue] = {"input": 17, "output": 3}
    observation: Final = LangfuseObservation.model_validate(
        {
            "id": "observation",
            "input": json.dumps(content) if encoded else content,
            "output": json.dumps(output) if encoded else output,
            "metadata": json.dumps(metadata) if encoded else metadata,
            "usageDetails": usage,
            "totalCost": cost,
        }
    )

    assert observation.input == content
    assert observation.output == output
    assert observation.usage_details == usage
    assert observation_spend(observation) == cost
    assert is_otel_v2_generation(observation, key_alias="test-key-alias")
    assert not is_otel_v2_generation(observation, key_alias="another-key")


def test_otel_v2_generation_matches_metadata_flattened_by_the_v2_observations_api() -> None:
    observation: Final = LangfuseObservation.model_validate(
        {
            "id": "observation",
            "metadata": {
                "scope.name": "litellm",
                "attributes.langfuse.observation.type": "generation",
                "attributes.litellm.metadata.user_api_key_alias": "test-key-alias",
                "attributes.gen_ai.usage.input_tokens": 17,
            },
        }
    )

    assert is_otel_v2_generation(observation, key_alias="test-key-alias")
    assert not is_otel_v2_generation(observation, key_alias="another-key")


def test_observation_preserves_plain_text_and_legacy_cost() -> None:
    cost: Final = 0.000031
    observation: Final = LangfuseObservation.model_validate(
        {"id": "observation", "input": "plain prompt", "output": "plain reply", "calculatedTotalCost": cost}
    )

    assert observation.input == "plain prompt"
    assert observation.output == "plain reply"
    assert observation_spend(observation) == cost
