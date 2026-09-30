"""gpt-6.1-sol rejects reasoning efforts OpenAI does not list for that model.

The public contract is low, medium, high, xhigh, and max. none and minimal are
unsupported, on both Chat Completions and the Responses API. The cost map
records that with supports_none_reasoning_effort and
supports_minimal_reasoning_effort set to false. Unknown models still pass
those levels through, because a missing key is not an explicit disable.
"""

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.llms.openai.openai import OpenAIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig


@pytest.fixture()
def config() -> OpenAIConfig:
    return OpenAIConfig()


@pytest.fixture()
def responses_config() -> OpenAIResponsesAPIConfig:
    return OpenAIResponsesAPIConfig()


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url))
    litellm.add_known_models(model_cost_map=litellm.model_cost)


def test_gpt61_sol_is_a_reasoning_series_model():
    assert OpenAIGPT5Config.is_model_gpt_5_model("gpt-6.1-sol")


@pytest.mark.parametrize("effort", ["none", "minimal"])
def test_gpt61_sol_drops_unsupported_reasoning_effort(config: OpenAIConfig, effort: str):
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": effort},
        optional_params={},
        model="gpt-6.1-sol",
        drop_params=True,
    )
    assert "reasoning_effort" not in params


@pytest.mark.parametrize("effort", ["none", "minimal"])
def test_gpt61_sol_rejects_unsupported_reasoning_effort(config: OpenAIConfig, effort: str):
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"reasoning_effort": effort},
            optional_params={},
            model="gpt-6.1-sol",
            drop_params=False,
        )


@pytest.mark.parametrize("effort", ["low", "max"])
def test_gpt61_sol_keeps_supported_reasoning_effort(config: OpenAIConfig, effort: str):
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": effort},
        optional_params={},
        model="gpt-6.1-sol",
        drop_params=False,
    )
    assert params["reasoning_effort"] == effort


@pytest.mark.parametrize("effort", ["none", "minimal"])
def test_responses_gpt61_sol_drops_unsupported_effort(responses_config: OpenAIResponsesAPIConfig, effort: str):
    params = responses_config.map_openai_params(
        response_api_optional_params={"reasoning": {"effort": effort}},
        model="gpt-6.1-sol",
        drop_params=True,
    )
    assert "reasoning" not in params


@pytest.mark.parametrize("effort", ["none", "minimal"])
def test_responses_gpt61_sol_rejects_unsupported_effort(responses_config: OpenAIResponsesAPIConfig, effort: str):
    with pytest.raises(litellm.UnsupportedParamsError):
        responses_config.map_openai_params(
            response_api_optional_params={"reasoning": {"effort": effort}},
            model="gpt-6.1-sol",
            drop_params=False,
        )


def test_responses_gpt61_sol_keeps_a_reasoning_summary_when_effort_is_dropped(
    responses_config: OpenAIResponsesAPIConfig,
):
    params = responses_config.map_openai_params(
        response_api_optional_params={"reasoning": {"effort": "none", "summary": "auto"}},
        model="gpt-6.1-sol",
        drop_params=True,
    )
    assert params["reasoning"] == {"summary": "auto"}
