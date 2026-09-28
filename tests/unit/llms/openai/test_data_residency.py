"""Tests for the OpenAI data-residency inference helper."""

import pytest

from litellm.llms.openai.data_residency import infer_openai_data_residency


@pytest.mark.parametrize(
    "api_base, expected",
    [
        ("https://eu.api.openai.com/v1", "eu"),
        ("https://eu.api.openai.com", "eu"),
        ("https://us.api.openai.com/v1", "us"),
        ("https://us.api.openai.com", "us"),
        ("https://EU.api.openai.com/v1", "eu"),
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
    assert (
        infer_openai_data_residency(custom_llm_provider, "https://eu.api.openai.com/v1")
        is None
    )
