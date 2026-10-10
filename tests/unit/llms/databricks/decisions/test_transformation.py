from typing import Final

import pytest

from litellm.llms.databricks.decisions.transformation import DatabricksDecisionsConfig

_BASE: Final = "https://workspace.example/serving-endpoints"
_CONFIG: Final = DatabricksDecisionsConfig()


@pytest.mark.parametrize("api_base", [_BASE, f"{_BASE}/"])
def test_the_complete_url_is_the_serving_endpoint_invocations_route(api_base: str) -> None:
    assert _CONFIG.get_complete_url(api_base, "openjev") == f"{_BASE}/openjev/invocations"


@pytest.mark.parametrize(
    "name",
    ["", "serving-endpoints/openjev", "openjev?x=1", "openjev#frag", ".", "..", ".openjev", "open jev", "openjev%2Fx"],
)
def test_a_name_that_would_rewrite_the_url_is_rejected_before_any_request(name: str) -> None:
    with pytest.raises(ValueError, match="bare endpoint name"):
        _ = _CONFIG.get_complete_url(_BASE, name)
    with pytest.raises(ValueError, match="bare endpoint name"):
        _ = _CONFIG.canonical_model(name)


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


@pytest.mark.parametrize("name", ["openjev", "databricks-openjev-qwen35-4b", "a.b_c-d", "a" * 1_000_000])
def test_a_bare_name_of_any_length_is_accepted_unchanged(name: str) -> None:
    assert _CONFIG.canonical_model(name) == name
    assert _CONFIG.get_complete_url(_BASE, name) == f"{_BASE}/{name}/invocations"


def test_a_short_invalid_name_is_echoed_in_full() -> None:
    with pytest.raises(ValueError) as refused:
        _ = _CONFIG.canonical_model("open/jev")
    assert str(refused.value).startswith("Databricks serving endpoint name 'open/jev' must be the bare endpoint name")


def test_a_megabyte_invalid_name_is_refused_with_a_bounded_message_that_keeps_the_rule() -> None:
    name: Final = "a" * 1_000_000 + "/"
    with pytest.raises(ValueError) as refused:
        _ = _CONFIG.canonical_model(name)
    message: Final = str(refused.value)
    assert len(message) < 1_000, len(message)
    assert f"'{'a' * 100}'... ({len(name)} characters) must be the bare endpoint name" in message
