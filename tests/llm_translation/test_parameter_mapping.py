import sys
import os
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.llms.parameter_mapping import ParameterMapping, ThinkingConfig


class TestParameterMapping:
    """Tests for the ParameterMapping class that handles OpenAI-to-Claude parameter mapping"""

    @pytest.mark.parametrize(
        "reasoning_effort, expected_budget",
        [
            ("low", 8000),
            ("medium", 16000),
            ("high", 24000),
            (None, None),
        ],
    )
    def test_map_reasoning_to_thinking(self, reasoning_effort, expected_budget):
        """Test that reasoning_effort values map to the correct thinking config"""
        result = ParameterMapping.map_reasoning_to_thinking(reasoning_effort)
        
        if reasoning_effort is None:
            assert result is None
        else:
            assert isinstance(result, dict)
            assert result["type"] == "enabled"
            assert result["budget_tokens"] == expected_budget