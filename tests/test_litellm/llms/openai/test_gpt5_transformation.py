import pytest

import litellm
from litellm.llms.openai.openai import OpenAIConfig


@pytest.fixture()
def config() -> OpenAIConfig:
    return OpenAIConfig()

def test_gpt5_supports_reasoning_effort(config: OpenAIConfig):
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5")
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5-mini")

def test_gpt5_maps_max_tokens(config: OpenAIConfig):
    params = config.map_openai_params(
        non_default_params={"max_tokens": 10},
        optional_params={},
        model="gpt-5",
        drop_params=False,
    )
    assert params["max_completion_tokens"] == 10
    assert "max_tokens" not in params


def test_gpt5_temperature_drop(config: OpenAIConfig):
    params = config.map_openai_params(
        non_default_params={"temperature": 0.2},
        optional_params={},
        model="gpt-5",
        drop_params=True,
    )
    assert "temperature" not in params


def test_gpt5_temperature_error(config: OpenAIConfig):
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.2},
            optional_params={},
            model="gpt-5",
            drop_params=False,
        )


def test_gpt5_unsupported_params_drop(config: OpenAIConfig):
    assert "top_p" not in config.get_supported_openai_params(model="gpt-5")
    params = config.map_openai_params(
        non_default_params={"top_p": 0.5},
        optional_params={},
        model="gpt-5",
        drop_params=True,
    )
    assert "top_p" not in params
