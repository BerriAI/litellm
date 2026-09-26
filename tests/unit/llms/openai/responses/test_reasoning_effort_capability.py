import re

import pytest

import litellm
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig


@pytest.mark.parametrize("effort", ["minimal", "low"])
def test_rejects_explicitly_unsupported_lower_reasoning_effort(effort: str) -> None:
    config = OpenAIResponsesAPIConfig()

    with pytest.raises(litellm.UnsupportedParamsError, match=f"reasoning.effort={effort}"):
        config.map_openai_params(
            response_api_optional_params={"reasoning": {"effort": effort}},
            model="gpt-5.5-pro",
            drop_params=False,
        )


def test_keeps_supported_reasoning_effort() -> None:
    config = OpenAIResponsesAPIConfig()

    result = config.map_openai_params(
        response_api_optional_params={"reasoning": {"effort": "medium"}},
        model="gpt-5.5-pro",
        drop_params=False,
    )

    assert result["reasoning"] == {"effort": "medium"}


def test_drop_params_removes_only_unsupported_effort() -> None:
    config = OpenAIResponsesAPIConfig()

    result = config.map_openai_params(
        response_api_optional_params={"reasoning": {"effort": "minimal", "summary": "detailed"}},
        model="gpt-5.5-pro",
        drop_params=True,
    )

    assert result["reasoning"] == {"summary": "detailed"}


def test_drop_params_revalidates_temperature_after_effort_drop() -> None:
    """Dropping an unsupported effort revalidates temperature against the emptied effort.

    `gpt-5.1` really declares `xhigh` unsupported while supporting (and defaulting to)
    `none`, so no capability stubbing is needed: the dropped effort resolves to `none`
    and a valid non-default temperature survives.
    """
    config = OpenAIResponsesAPIConfig()

    result = config.map_openai_params(
        response_api_optional_params={"reasoning": {"effort": "xhigh"}, "temperature": 0.5},
        model="gpt-5.1",
        drop_params=True,
    )

    assert "reasoning" not in result
    assert result["temperature"] == 0.5


def test_azure_config_uses_azure_capability_map_for_effort() -> None:
    """A bare Azure deployment name must resolve against the Azure map entry.

    OpenAI's `gpt-5.4` entry explicitly disables `minimal` while Azure's enables it;
    the OpenAI lookup would wrongly reject an effort Azure supports.
    """
    openai_config = OpenAIResponsesAPIConfig()
    with pytest.raises(litellm.UnsupportedParamsError, match=re.escape("reasoning.effort=minimal")):
        openai_config.map_openai_params(
            response_api_optional_params={"reasoning": {"effort": "minimal"}},
            model="gpt-5.4",
            drop_params=False,
        )

    azure_config = AzureOpenAIResponsesAPIConfig()
    result = azure_config.map_openai_params(
        response_api_optional_params={"reasoning": {"effort": "minimal"}},
        model="gpt-5.4",
        drop_params=False,
    )

    assert result["reasoning"] == {"effort": "minimal"}
