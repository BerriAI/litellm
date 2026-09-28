import pytest
from litellm.llms.cohere.chat.transformation import CohereChatConfig
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
from litellm.utils import UnsupportedParamsError, get_optional_params


@pytest.mark.parametrize("config_cls", [CohereChatConfig, CohereV2ChatConfig])
def test_tool_choice_not_declared(config_cls):
    params = config_cls().get_supported_openai_params(model="command-r")
    assert "tool_choice" not in params


def test_tool_choice_raises_unsupported_params_error():
    # tool_choice used to be declared but silently dropped; with drop_params=False
    # (the default) litellm must surface that instead of ignoring it
    with pytest.raises(UnsupportedParamsError):
        get_optional_params(
            model="command-r",
            custom_llm_provider="cohere",
            tool_choice="auto",
            request_timeout=10,
            num_retries=0,
        )


def test_stop_still_maps_to_stop_sequences():
    result = CohereChatConfig().map_openai_params(
        non_default_params={"stop": ["X"]},
        optional_params={},
        model="command-r",
        drop_params=False,
    )
    assert result["stop_sequences"] == ["X"]
