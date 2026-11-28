from collections.abc import Mapping
from typing import Final

import pytest

from litellm.llms.openai_like.decisions.transformation import JSONProviderDecisionsConfig
from litellm.llms.openai_like.json_loader import SimpleProviderConfig

_ENTRY: Final[Mapping[str, object]] = {
    "base_url": "https://decisions.example",
    "api_key_env": ["EXAMPLE_DECISIONS_API_KEY", "EXAMPLE_DECISIONS_TOKEN"],
    "api_base_env": "EXAMPLE_DECISIONS_API_BASE",
    "supported_endpoints": ["/v1/systemone"],
}
_ENV_NAMES: Final = ("EXAMPLE_DECISIONS_API_KEY", "EXAMPLE_DECISIONS_TOKEN", "EXAMPLE_DECISIONS_API_BASE")


def _config(entry: Mapping[str, object] = _ENTRY, model: str = "") -> JSONProviderDecisionsConfig:
    return JSONProviderDecisionsConfig(SimpleProviderConfig("example_decisions", dict(entry)), model)


@pytest.mark.parametrize(
    ("environment", "connection"),
    (
        ({}, ("https://decisions.example", None)),
        ({"EXAMPLE_DECISIONS_TOKEN": "second-key"}, ("https://decisions.example", "second-key")),
        (
            {"EXAMPLE_DECISIONS_API_BASE": "https://gateway.example", "EXAMPLE_DECISIONS_API_KEY": "first-key"},
            ("https://gateway.example", "first-key"),
        ),
    ),
    ids=("entry_base_url_without_a_key", "any_listed_key_env", "environment_overrides"),
)
def test_connection_comes_from_the_entry_env_vars_before_its_base_url(
    environment: Mapping[str, str],
    connection: tuple[str, str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert _config().resolve_connection(None, None) == connection


def test_entry_without_an_api_base_env_ignores_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE_DECISIONS_API_BASE", "https://gateway.example")

    entry: Final = {key: value for key, value in _ENTRY.items() if key != "api_base_env"}

    assert _config(entry).resolve_api_base(None) == "https://decisions.example"


@pytest.mark.parametrize(
    ("model", "api_base"),
    (
        ("moved-model", "https://moved.example"),
        ("example_decisions/moved-model", "https://moved.example"),
        ("other-model", "https://decisions.example"),
    ),
)
def test_a_model_listed_in_model_base_urls_defaults_to_its_own_base(
    model: str, api_base: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EXAMPLE_DECISIONS_API_BASE", raising=False)

    entry: Final = {**_ENTRY, "model_base_urls": {"moved-model": "https://moved.example"}}

    assert _config(entry, model).resolve_api_base(None) == api_base
