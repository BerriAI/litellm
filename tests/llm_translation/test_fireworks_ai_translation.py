import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.llms.fireworks_ai.chat.fireworks_ai_transformation import FireworksAIConfig

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


def test_map_response_format():
    response_format = {
        'type': 'json_schema',
        'json_schema': {
            'schema': {
                'properties': {
                    'result': {
                        'type': 'boolean'
                    }
                },
                'required': [
                    'result'
                ],
                'type': 'object',
            },
            'name': 'BooleanResponse',
            'strict': True
        }
    }
    result = fireworks.map_openai_params({"response_format":response_format}, {}, "some_model")
    assert result == {
        "response_format" : {
            'type': 'json_object',
            'schema': {
                'properties': {
                    'result': {
                        'type': 'boolean'
                    }
                },
                'required': [
                    'result'
                ],
                'type': 'object',
            },
        }
    }