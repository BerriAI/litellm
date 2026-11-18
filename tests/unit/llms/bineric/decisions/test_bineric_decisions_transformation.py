from collections.abc import Mapping
from typing import Final

import pytest

from litellm.llms.bineric.decisions.transformation import BinericDecisionsConfig


@pytest.mark.parametrize(
    ("environment", "connection"),
    (
        ({}, ("https://api.bineric.com/v1", None)),
        (
            {"BINERIC_API_BASE": "https://gateway.example/v1", "BINERIC_API_KEY": "environment-key"},
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
    monkeypatch.delenv("BINERIC_API_BASE", raising=False)
    monkeypatch.delenv("BINERIC_API_KEY", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert BinericDecisionsConfig().resolve_connection(None, None) == connection


@pytest.mark.parametrize(
    "environment",
    (
        {},
        {"BINERIC_API_BASE": "https://api.bineric.com"},
        {"BINERIC_API_BASE": "https://api.bineric.com/v1/"},
    ),
    ids=("public_api", "base_without_v1", "base_with_v1_and_slash"),
)
def test_requests_go_to_v1_systemone_whether_or_not_the_base_ends_in_v1(
    environment: Mapping[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BINERIC_API_BASE", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    config: Final = BinericDecisionsConfig()
    api_base, _ = config.resolve_connection(None, None)

    assert api_base is not None
    assert config.get_complete_url(api_base=api_base, model="zuse-27b") == "https://api.bineric.com/v1/systemone"
