from typing import Final

import pytest

from litellm.llms.databricks.decisions.transformation import DatabricksDecisionsConfig
from litellm.types.decisions import (
    DecisionsIRPredicateAnswer,
    DecisionsIRPredicateQuestion,
    DecisionsIRRequest,
    DecisionsIRState,
)

_HOST: Final = "https://workspace.example"
_ROUTE: Final = f"{_HOST}/api/2.0/ai-functions/ai-decide"
_CONFIG: Final = DatabricksDecisionsConfig()
_REQUEST: Final = DecisionsIRRequest(
    input=DecisionsIRState(state="export hangs"),
    questions=(DecisionsIRPredicateQuestion(name="is_defect", instructions="Is this a defect?", criteria=None),),
)


@pytest.mark.parametrize("api_base", [_HOST, f"{_HOST}/", f"{_HOST}/serving-endpoints", f"{_HOST}/serving-endpoints/"])
def test_the_complete_url_is_the_workspace_ai_decide_route(api_base: str) -> None:
    assert _CONFIG.get_complete_url(api_base, "ai_decide") == _ROUTE


@pytest.mark.parametrize("model", ["", "databricks-openjev-qwen35-4b", "ai-decide", "AI_DECIDE", "ai_decide/x"])
def test_a_model_other_than_ai_decide_is_rejected_before_any_request(model: str) -> None:
    with pytest.raises(ValueError, match="'ai_decide'"):
        _ = _CONFIG.canonical_model(model)


def test_the_request_body_omits_model_because_the_route_rejects_unexpected_parameters() -> None:
    body: Final = _CONFIG.transform_decisions_request("ai_decide", _REQUEST, "databricks")
    assert body == {
        "state": "export hangs",
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }


def test_the_response_envelope_is_unwrapped_and_noul_probability_becomes_the_noul_answer() -> None:
    payload: Final = {
        "response": {"answers": {"is_defect": {"type": "noul", "probability": 0.9}}},
        "metadata": {"version": "1.0"},
    }
    parsed: Final = _CONFIG.parse_response(payload, _REQUEST)
    assert parsed.answers == (DecisionsIRPredicateAnswer(probability=0.9),)
    assert parsed.model is None
    assert dict(parsed.extra) == {"metadata": {"version": "1.0"}}


def test_connection_prefers_the_configured_values_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", "https://env.example")
    monkeypatch.setenv("DATABRICKS_API_KEY", "dapi-env")
    connection: Final = _CONFIG.connection(f"{_HOST}/serving-endpoints/", "dapi-configured")
    assert (connection.api_base, connection.api_key) == (_HOST, "dapi-configured")


def test_connection_falls_back_to_the_workspace_token_when_the_api_key_variable_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", _HOST)
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-token")
    connection: Final = _CONFIG.connection(None, None)
    assert (connection.api_base, connection.api_key) == (_HOST, "dapi-token")


def test_connection_never_sends_the_environment_key_to_a_configured_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", _HOST)
    monkeypatch.setenv("DATABRICKS_API_KEY", "dapi-env")
    with pytest.raises(ValueError, match="DATABRICKS_API_BASE"):
        _ = _CONFIG.connection("https://collector.example", None)


def test_classifier_response_unwraps_the_envelope_and_accounts_the_configured_model() -> None:
    choice: Final = {"type": "choice", "choice": "simple", "confidence": 0.9, "probabilities": {"simple": 0.9}}
    body: Final = {"response": {"answers": {"tier": choice}}, "metadata": {"version": "1.0"}}
    normalized: Final = _CONFIG.classifier_response(body, "ai_decide")
    assert dict(normalized) == {"answers": {"tier": choice}, "metadata": {"version": "1.0"}, "model": "ai_decide"}
