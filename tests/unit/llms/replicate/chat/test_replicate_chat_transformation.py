import pytest
from litellm.llms.replicate.chat.transformation import ReplicateConfig
from litellm.utils import UnsupportedParamsError, get_optional_params

REMOVED_PARAMS = {
    "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}],
    "tool_choice": "auto",
    "functions": [{"name": "f", "parameters": {}}],
    "function_call": "auto",
    "seed": 42,
}


def test_tool_params_not_declared():
    supported = ReplicateConfig().get_supported_openai_params(model="meta/meta-llama-3-70b-instruct")
    for param in REMOVED_PARAMS:
        assert param not in supported


@pytest.mark.parametrize("param,value", sorted(REMOVED_PARAMS.items()))
def test_removed_params_raise_unsupported_params_error(param, value):
    # these used to be declared but were silently dropped, degrading tool-calling
    # requests to plain completions; with drop_params=False (the default) litellm
    # must surface that instead of ignoring it
    with pytest.raises(UnsupportedParamsError):
        get_optional_params(
            model="meta/meta-llama-3-70b-instruct",
            custom_llm_provider="replicate",
            **{param: value},
            request_timeout=10,
            num_retries=0,
        )


def test_stop_still_maps_to_stop_sequences():
    result = ReplicateConfig().map_openai_params(
        non_default_params={"stop": ["X"]},
        optional_params={},
        model="meta/meta-llama-3-70b-instruct",
        drop_params=False,
    )
    assert result["stop_sequences"] == ["X"]
