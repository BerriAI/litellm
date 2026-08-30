import pytest

import litellm
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
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


def test_drop_params_revalidates_temperature_after_effort_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    config = OpenAIResponsesAPIConfig()

    monkeypatch.setattr(OpenAIGPT5Config, "_supports_reasoning_effort_level", lambda model, level: False)
    monkeypatch.setattr(config, "_supports_reasoning_effort_none", lambda model: True)
    monkeypatch.setattr(config, "_effort_resolves_to_none", lambda model, effort: effort is None)

    result = config.map_openai_params(
        response_api_optional_params={"reasoning": {"effort": "xhigh"}, "temperature": 0.5},
        model="gpt-5-test",
        drop_params=True,
    )

    assert "reasoning" not in result
    assert result["temperature"] == 0.5
