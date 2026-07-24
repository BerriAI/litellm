import os
import pytest
from unittest.mock import patch
from litellm.llms.opencode_go.chat import OpenCodeGoConfig


class TestOpenCodeGoParamMapping:
    """Test OpenCode Go param mapping — fails if mapping breaks or params silently get dropped."""

    def test_supported_params_are_mapped(self):
        config = OpenCodeGoConfig()
        result = config.map_openai_params(
            non_default_params={"max_tokens": 100, "temperature": 0.7},
            optional_params={},
            model="opencode_go/some-model",
            drop_params=False,
        )
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7

    def test_unsupported_param_is_dropped(self):
        config = OpenCodeGoConfig()
        result = config.map_openai_params(
            non_default_params={"not_a_real_param": "x"},
            optional_params={},
            model="opencode_go/some-model",
            drop_params=False,
        )
        assert "not_a_real_param" not in result


if __name__ == "__main__":
    pytest.main([__file__])
