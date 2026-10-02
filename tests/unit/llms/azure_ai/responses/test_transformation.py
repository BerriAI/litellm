import pytest

import litellm
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.azure_ai.responses.transformation import AzureAIResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


def _responses_params(**fields: object) -> ResponsesAPIOptionalRequestParams:
    return ResponsesAPIOptionalRequestParams(**fields)  # pyright: ignore[reportArgumentType]  # TypedDict kwargs from a test helper


def test_foundry_responses_effort_gate_reads_the_azure_ai_row(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
):
    """A Foundry deployment and its Azure OpenAI namesake are different products, so the Foundry
    Responses route answers from the azure_ai/ row the Foundry chat route already reads, never from
    the azure/ twin. The rows are made to disagree on low, a level neither real row flags, so a
    leak from the wrong row cannot hide behind rows that happen to agree."""
    monkeypatch.setitem(
        litellm.model_cost,
        "azure_ai/gpt-6-sol",
        {**litellm.model_cost["azure_ai/gpt-6-sol"], "supports_low_reasoning_effort": False},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "azure/gpt-6-sol",
        {**litellm.model_cost["azure/gpt-6-sol"], "supports_low_reasoning_effort": True},
    )

    dropped = AzureAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=_responses_params(reasoning={"effort": "low"}),
        model="gpt-6-sol",
        drop_params=True,
    )
    assert "reasoning" not in dropped

    with pytest.raises(litellm.UnsupportedParamsError):
        AzureAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params=_responses_params(reasoning={"effort": "low"}),
            model="gpt-6-sol",
            drop_params=False,
        )

    kept_by_azure_openai = AzureOpenAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=_responses_params(reasoning={"effort": "low"}),
        model="gpt-6-sol",
        drop_params=True,
    )
    assert kept_by_azure_openai["reasoning"] == {"effort": "low"}


def test_foundry_responses_none_unlocks_temperature_off_the_azure_ai_row(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
):
    """The sampling-param unlock that rides on none reads the same azure_ai/ row as the effort gate:
    with the azure/ twin turned off for none, the Foundry row still lets none through and, only
    then, a non-default temperature."""
    monkeypatch.setitem(
        litellm.model_cost,
        "azure_ai/gpt-6-sol",
        {**litellm.model_cost["azure_ai/gpt-6-sol"], "supports_none_reasoning_effort": True},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "azure/gpt-6-sol",
        {**litellm.model_cost["azure/gpt-6-sol"], "supports_none_reasoning_effort": False},
    )

    params = AzureAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=_responses_params(temperature=0.2, reasoning={"effort": "none"}),
        model="gpt-6-sol",
        drop_params=False,
    )

    assert params["temperature"] == 0.2
    assert params["reasoning"] == {"effort": "none"}
