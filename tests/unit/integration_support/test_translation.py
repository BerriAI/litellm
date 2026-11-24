import base64
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue


@pytest.mark.parametrize("response_id", ("resp_original", "", "resp_original;part", "resp_response_id:original"))
def test_response_id_uses_the_deployment_id_and_preserves_provider_identity_and_the_complete_body(
    response_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration.translation.runner import expected_response_for_deployment

    original: Final = f"litellm:custom_llm_provider:openai;model_id:recorded-deployment;response_id:{response_id}"
    expected: Final[dict[str, JsonValue]] = {
        "id": "resp_" + base64.b64encode(original.encode()).decode(),
        "content": [{"text": "answer"}],
    }
    result: Final = expected_response_for_deployment(expected, "isolated-deployment")
    updated: Final = f"litellm:custom_llm_provider:openai;model_id:isolated-deployment;response_id:{response_id}"
    assert result == {**expected, "id": "resp_" + base64.b64encode(updated.encode()).decode()}
    assert expected["id"] == "resp_" + base64.b64encode(original.encode()).decode()
    with pytest.raises(AssertionError, match="must identify the selected deployment"):
        expected_response_for_deployment(expected, None)


def test_unencoded_provider_identity_is_preserved_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration.translation.runner import expected_response_for_deployment

    expected: Final[dict[str, JsonValue]] = {"id": "resp_original", "content": "answer"}
    assert expected_response_for_deployment(expected, "isolated-deployment") is expected
