"""Tests for the OpenAI data-residency inference helper."""

import pytest

import litellm
from litellm.llms.openai.data_residency import infer_openai_data_residency
from litellm.types.utils import DataResidency


@pytest.mark.parametrize(
    "api_base, expected",
    [
        ("https://eu.api.openai.com/v1", "eu"),
        ("https://eu.api.openai.com", "eu"),
        ("https://us.api.openai.com/v1", "us"),
        ("https://us.api.openai.com", "us"),
        ("https://EU.api.openai.com/v1", "eu"),
        ("https://gov.api.openai.com/v1", DataResidency.FEDRAMP.value),
        ("https://GOV.api.openai.com/v1", DataResidency.FEDRAMP.value),
        ("https://api.openai.com/v1", None),
        ("https://api.openai.com", None),
        ("https://example.com/v1", None),
        ("https://my-azure-endpoint.openai.azure.com/openai/deployments/foo", None),
        ("", None),
        (None, None),
        ("not a url", None),
    ],
)
def test_infer_openai_data_residency(api_base, expected):
    assert infer_openai_data_residency("openai", api_base) == expected


@pytest.mark.parametrize("custom_llm_provider", [None, "anthropic", "azure", "bedrock"])
def test_infer_openai_data_residency_non_openai_provider(custom_llm_provider):
    assert infer_openai_data_residency(custom_llm_provider, "https://eu.api.openai.com/v1") is None


@pytest.mark.parametrize("env_var", ["OPENAI_BASE_URL", "OPENAI_API_BASE"])
def test_infer_openai_data_residency_from_env_when_api_base_unset(monkeypatch, env_var):
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv(env_var, "https://gov.api.openai.com/v1")
    assert infer_openai_data_residency("openai", None) == DataResidency.FEDRAMP.value


def test_infer_openai_data_residency_explicit_api_base_wins_over_env(monkeypatch):
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gov.api.openai.com/v1")
    assert infer_openai_data_residency("openai", "https://eu.api.openai.com/v1") == "eu"


def test_infer_openai_data_residency_from_litellm_api_base(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", "https://us.api.openai.com/v1")
    assert infer_openai_data_residency("openai", None) == "us"
