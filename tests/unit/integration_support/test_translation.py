import base64
from typing import Final

import pytest
from pydantic import JsonValue

from integration.translation.runner import expected_response_for_deployment


def test_response_id_uses_the_deployment_id_and_preserves_provider_identity_and_the_complete_body() -> None:
    original: Final = "litellm:custom_llm_provider:openai;model_id:recorded-deployment;response_id:resp_original"
    expected: Final[dict[str, JsonValue]] = {
        "id": "resp_" + base64.b64encode(original.encode()).decode(),
        "content": [{"text": "answer"}],
    }
    result: Final = expected_response_for_deployment(expected, "isolated-deployment")
    updated: Final = "litellm:custom_llm_provider:openai;model_id:isolated-deployment;response_id:resp_original"
    assert result == {**expected, "id": "resp_" + base64.b64encode(updated.encode()).decode()}
    assert expected["id"] == "resp_" + base64.b64encode(original.encode()).decode()
    with pytest.raises(AssertionError, match="must identify the selected deployment"):
        expected_response_for_deployment(expected, None)


def test_unencoded_provider_identity_is_preserved_exactly() -> None:
    expected: Final[dict[str, JsonValue]] = {"id": "resp_original", "content": "answer"}
    assert expected_response_for_deployment(expected, "isolated-deployment") is expected
