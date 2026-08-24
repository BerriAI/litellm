"""`tool_choice="any"` is Bedrock's own native value and must be accepted."""

import pytest

import litellm
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


class TestBedrockToolChoiceAny:
    def setup_method(self):
        self.config = AmazonConverseConfig()

    @pytest.mark.parametrize("tool_choice", ["required", "any"])
    def test_required_and_any_both_map_to_bedrock_any(self, tool_choice):
        result = self.config.map_tool_choice_values(model=MODEL, tool_choice=tool_choice, drop_params=False)

        assert result == {"any": {}}

    def test_auto_is_unchanged(self):
        result = self.config.map_tool_choice_values(model=MODEL, tool_choice="auto", drop_params=False)

        assert result == {"auto": {}}

    def test_specific_tool_is_unchanged(self):
        result = self.config.map_tool_choice_values(
            model=MODEL,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            drop_params=False,
        )

        assert result == {"tool": {"name": "get_weather"}}

    def test_unknown_value_still_raises_and_lists_any(self):
        with pytest.raises(litellm.utils.UnsupportedParamsError) as exc_info:
            self.config.map_tool_choice_values(model=MODEL, tool_choice="definitely_not_supported", drop_params=False)

        # The error text is the discoverability surface for these values.
        assert "'any'" in str(exc_info.value)

    def test_none_still_drops_when_drop_params_is_set(self):
        assert self.config.map_tool_choice_values(model=MODEL, tool_choice="none", drop_params=True) is None
