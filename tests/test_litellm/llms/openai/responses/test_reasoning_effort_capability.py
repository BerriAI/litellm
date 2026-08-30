import pytest

import litellm
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
