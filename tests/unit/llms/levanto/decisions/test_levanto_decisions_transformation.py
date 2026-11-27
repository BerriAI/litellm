from collections.abc import Mapping

import pytest

from litellm.llms.levanto.decisions.transformation import LevantoDecisionsConfig


@pytest.mark.parametrize(
    ("environment", "connection"),
    (
        ({}, ("https://api.levanto.ai", None)),
        (
            {"LEVANTO_BASE_URL": "https://gateway.example/v1", "LEVANTO_API_KEY": "environment-key"},
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
    monkeypatch.delenv("LEVANTO_BASE_URL", raising=False)
    monkeypatch.delenv("LEVANTO_API_KEY", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert LevantoDecisionsConfig().resolve_connection(None, None) == connection


@pytest.mark.parametrize(
    "api_base",
    ("https://api.levanto.ai", "https://api.levanto.ai/v1", "https://api.levanto.ai/v1/"),
)
def test_requests_go_to_v1_systemone_whether_or_not_the_base_ends_in_v1(api_base: str) -> None:
    assert (
        LevantoDecisionsConfig().get_complete_url(api_base=api_base, model="levanto-sage-2")
        == "https://api.levanto.ai/v1/systemone"
    )
