import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import transcription
from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig
from base_llm_unit_tests import BaseLLMChatTest
from base_audio_transcription_unit_tests import BaseLLMAudioTranscriptionTest

fireworks = FireworksAIConfig()


def test_map_openai_params_tool_choice():
    # Test case 1: tool_choice is "required"
    result = fireworks.map_openai_params(
        {"tool_choice": "required"}, {}, "some_model", drop_params=False
    )
    assert result == {"tool_choice": "any"}

    # Test case 2: tool_choice is "auto"
    result = fireworks.map_openai_params(
        {"tool_choice": "auto"}, {}, "some_model", drop_params=False
    )
    assert result == {"tool_choice": "auto"}

    # Test case 3: tool_choice is not present
    result = fireworks.map_openai_params(
        {"some_other_param": "value"}, {}, "some_model", drop_params=False
    )
    assert result == {}

    # Test case 4: tool_choice is None
    result = fireworks.map_openai_params(
        {"tool_choice": None}, {}, "some_model", drop_params=False
    )
    assert result == {"tool_choice": None}


def test_map_response_format():
    """
    Test that the response format is translated correctly.

    h/t to https://github.com/DaveDeCaprio (@DaveDeCaprio) for the test case

    Relevant Issue: https://github.com/BerriAI/litellm/issues/6797
    Fireworks AI Ref: https://docs.fireworks.ai/structured-responses/structured-response-formatting#step-1-import-libraries
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "properties": {"result": {"type": "boolean"}},
                "required": ["result"],
                "type": "object",
            },
            "name": "BooleanResponse",
            "strict": True,
        },
    }
    result = fireworks.map_openai_params(
        {"response_format": response_format}, {}, "some_model", drop_params=False
    )
    assert result == {
        "response_format": {
            "type": "json_object",
            "schema": {
                "properties": {"result": {"type": "boolean"}},
                "required": ["result"],
                "type": "object",
            },
        }
    }


class TestFireworksAIChatCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "fireworks_ai/accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_multilingual_requests(self):
        """
        Fireworks AI raises a 500 BadRequest error when the request contains invalid utf-8 sequences.
        """
        pass


class TestFireworksAIAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> dict:
        return {
            "model": "fireworks_ai/whisper-v3",
            "api_base": "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.FIREWORKS_AI
