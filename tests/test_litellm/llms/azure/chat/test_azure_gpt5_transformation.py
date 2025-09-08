import pytest

import litellm
from litellm.llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config


@pytest.fixture()
def config() -> AzureOpenAIGPT5Config:
    return AzureOpenAIGPT5Config()


def test_azure_gpt5_supports_reasoning_effort(config: AzureOpenAIGPT5Config):
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5")
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt5_series/my-deployment")


def test_azure_gpt5_maps_max_tokens(config: AzureOpenAIGPT5Config):
    params = config.map_openai_params(
        non_default_params={"max_tokens": 5},
        optional_params={},
        model="gpt5_series/gpt-5",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["max_completion_tokens"] == 5
    assert "max_tokens" not in params


def test_azure_gpt5_temperature_error(config: AzureOpenAIGPT5Config):
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.2},
            optional_params={},
            model="gpt-5",
            drop_params=False,
            api_version="2024-05-01-preview",
        )


def test_azure_gpt5_series_transform_request(config: AzureOpenAIGPT5Config):
    request = config.transform_request(
        model="gpt5_series/gpt-5",
        messages=[],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "gpt-5"
