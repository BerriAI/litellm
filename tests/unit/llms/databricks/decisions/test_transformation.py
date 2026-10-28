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
_BASE: Final = f"{_HOST}/serving-endpoints"
_AI_DECIDE_ROUTE: Final = f"{_HOST}/api/2.0/ai-functions/ai-decide"
_CONFIG: Final = DatabricksDecisionsConfig()
_REQUEST: Final = DecisionsIRRequest(
    input=DecisionsIRState(state="export hangs"),
    questions=(DecisionsIRPredicateQuestion(name="is_defect", instructions="Is this a defect?", criteria=None),),
)
_QUESTIONS: Final = {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}}


@pytest.mark.parametrize("api_base", [_BASE, f"{_BASE}/"])
def test_a_serving_endpoint_name_is_called_on_its_invocations_route(api_base: str) -> None:
    assert _CONFIG.get_complete_url(api_base, "openjev") == f"{_BASE}/openjev/invocations"


@pytest.mark.parametrize("api_base", [_HOST, f"{_HOST}/", _BASE, f"{_BASE}/"])
def test_ai_decide_is_called_on_the_workspace_ai_function_route(api_base: str) -> None:
    assert _CONFIG.get_complete_url(api_base, "ai_decide") == _AI_DECIDE_ROUTE


@pytest.mark.parametrize(
    "name",
    ["", "serving-endpoints/openjev", "openjev?x=1", "openjev#frag", ".", "..", ".openjev", "open jev", "openjev%2Fx"],
)
def test_a_name_that_would_rewrite_the_url_is_rejected_before_any_request(name: str) -> None:
    with pytest.raises(ValueError, match="bare serving endpoint name"):
        _ = _CONFIG.get_complete_url(_BASE, name)
    with pytest.raises(ValueError, match="bare serving endpoint name"):
        _ = _CONFIG.canonical_model(name)


def test_the_rejection_does_not_echo_the_rejected_name() -> None:
    name: Final = "openjev?token=" + "x" * 4096
    with pytest.raises(ValueError, match="bare serving endpoint name") as raised:
        _ = _CONFIG.canonical_model(name)
    assert "openjev?token=" not in str(raised.value)


def test_a_serving_endpoint_request_keeps_the_model_the_endpoint_expects() -> None:
    body: Final = _CONFIG.transform_decisions_request("openjev", _REQUEST, "databricks")
    assert body == {"model": "openjev", "state": "export hangs", "questions": _QUESTIONS}


def test_the_ai_decide_request_omits_model_because_the_route_rejects_unexpected_parameters() -> None:
    body: Final = _CONFIG.transform_decisions_request("ai_decide", _REQUEST, "databricks")
    assert body == {"state": "export hangs", "questions": _QUESTIONS}


def test_the_ai_decide_envelope_is_unwrapped_and_noul_probability_becomes_the_noul_answer() -> None:
    payload: Final = {
        "response": {"answers": {"is_defect": {"type": "noul", "probability": 0.9}}},
        "metadata": {"version": "1.0"},
    }
    parsed: Final = _CONFIG.parse_response(payload, _REQUEST)
    assert parsed.answers == (DecisionsIRPredicateAnswer(probability=0.9),)
    assert parsed.model is None
    assert dict(parsed.extra) == {"metadata": {"version": "1.0"}}


def test_a_serving_endpoint_answer_is_parsed_as_sent() -> None:
    payload: Final = {
        "model": "openjev",
        "answers": {"is_defect": {"type": "noul", "noul": 0.25}},
        "usage": {"input_tokens": 12, "output_tokens": 0},
    }
    parsed: Final = _CONFIG.parse_response(payload, _REQUEST)
    assert parsed.answers == (DecisionsIRPredicateAnswer(probability=0.25),)
    assert parsed.model == "openjev"


def test_connection_prefers_the_configured_values_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", "https://env.example/serving-endpoints")
    monkeypatch.setenv("DATABRICKS_API_KEY", "dapi-env")
    connection: Final = _CONFIG.connection(f"{_BASE}/", "dapi-configured")
    assert (connection.api_base, connection.api_key) == (_BASE, "dapi-configured")


def test_connection_falls_back_to_the_workspace_token_when_the_api_key_variable_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", _BASE)
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-token")
    connection: Final = _CONFIG.connection(None, None)
    assert (connection.api_base, connection.api_key) == (_BASE, "dapi-token")


def test_connection_never_sends_the_environment_key_to_a_configured_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", _BASE)
    monkeypatch.setenv("DATABRICKS_API_KEY", "dapi-env")
    with pytest.raises(ValueError, match="DATABRICKS_API_BASE"):
        _ = _CONFIG.connection("https://collector.example/serving-endpoints", None)


def test_classifier_response_accounts_the_serving_endpoint_instead_of_the_container_model() -> None:
    body: Final = {"model": "/mosaicml/local_model", "answers": {}, "usage": {"input_tokens": 3, "output_tokens": 0}}
    normalized: Final = _CONFIG.classifier_response(body, "openjev")
    assert normalized["model"] == "openjev"
    assert normalized["usage"] == body["usage"]
    assert body["model"] == "/mosaicml/local_model"


def test_classifier_response_unwraps_the_ai_decide_envelope_and_accounts_the_configured_model() -> None:
    choice: Final = {"type": "choice", "choice": "simple", "confidence": 0.9, "probabilities": {"simple": 0.9}}
    body: Final = {"response": {"answers": {"tier": choice}}, "metadata": {"version": "1.0"}}
    normalized: Final = _CONFIG.classifier_response(body, "ai_decide")
    assert dict(normalized) == {"answers": {"tier": choice}, "metadata": {"version": "1.0"}, "model": "ai_decide"}
