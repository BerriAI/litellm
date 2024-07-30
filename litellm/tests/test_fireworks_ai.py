import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.llms.fireworks_ai import FireworksAIConfig

fireworks = FireworksAIConfig()


def test_map_openai_params_tool_choice():
    # Test case 1: tool_choice is "required"
    result = fireworks.map_openai_params({"tool_choice": "required"}, {}, "some_model")
    assert result == {"tool_choice": "any"}

    # Test case 2: tool_choice is "auto"
    result = fireworks.map_openai_params({"tool_choice": "auto"}, {}, "some_model")
    assert result == {"tool_choice": "auto"}

    # Test case 3: tool_choice is not present
    result = fireworks.map_openai_params(
        {"some_other_param": "value"}, {}, "some_model"
    )
    assert result == {}

    # Test case 4: tool_choice is None
    result = fireworks.map_openai_params({"tool_choice": None}, {}, "some_model")
    assert result == {"tool_choice": None}
