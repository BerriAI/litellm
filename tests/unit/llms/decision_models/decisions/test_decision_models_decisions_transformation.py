from collections.abc import Mapping

import pytest

from litellm.llms.decision_models.decisions.transformation import DecisionModelsDecisionsConfig


@pytest.mark.parametrize(
    ("environment", "connection"),
    (
        ({}, ("https://api.decisionmodels.io", None)),
        (
            {"DECISION_MODELS_API_BASE": "https://gateway.example/v1", "DECISION_MODELS_API_KEY": "environment-key"},
            ("https://gateway.example/v1", "environment-key"),
        ),
    ),
    ids=("public_api_without_a_key", "environment"),
)
def test_connection_comes_from_the_environment_before_the_public_api(
    environment: Mapping[str, str],
    connection: tuple[str, str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DECISION_MODELS_API_BASE", raising=False)
    monkeypatch.delenv("DECISION_MODELS_API_KEY", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert DecisionModelsDecisionsConfig().resolve_connection(None, None) == connection


@pytest.mark.parametrize(
    "api_base",
    ("https://api.decisionmodels.io", "https://api.decisionmodels.io/v1", "https://api.decisionmodels.io/v1/"),
)
def test_requests_go_to_v1_systemone_whether_or_not_the_base_ends_in_v1(api_base: str) -> None:
    assert (
        DecisionModelsDecisionsConfig().get_complete_url(api_base=api_base, model="s1-fast")
        == "https://api.decisionmodels.io/v1/systemone"
    )
