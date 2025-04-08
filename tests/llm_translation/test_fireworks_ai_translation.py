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
            "model": "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct"
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_multilingual_requests(self):
        """
        Fireworks AI raises a 500 BadRequest error when the request contains invalid utf-8 sequences.
        """
        pass

    @pytest.mark.parametrize(
        "response_format",
        [
            {"type": "json_object"},
            {"type": "text"},
        ],
    )
    @pytest.mark.flaky(retries=6, delay=1)
    def test_json_response_format(self, response_format):
        """
        Test that the JSON response format is supported by the LLM API
        """
        from litellm.utils import supports_response_schema
        from openai import OpenAI
        from unittest.mock import patch

        client = OpenAI()

        base_completion_call_args = self.get_base_completion_call_args()
        litellm.set_verbose = True

        messages = [
            {
                "role": "system",
                "content": "Your output should be a JSON object with no additional properties.  ",
            },
            {
                "role": "user",
                "content": "Respond with this in json. city=San Francisco, state=CA, weather=sunny, temp=60",
            },
        ]

        with patch.object(
            client.chat.completions.with_raw_response, "create"
        ) as mock_post:
            response = self.completion_function(
                **base_completion_call_args,
                messages=messages,
                response_format=response_format,
                client=client,
            )

            mock_post.assert_called_once()
            if response_format["type"] == "json_object":
                assert (
                    mock_post.call_args.kwargs["response_format"]["type"]
                    == "json_object"
                )
            else:
                assert mock_post.call_args.kwargs["response_format"]["type"] == "text"


class TestFireworksAIAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> dict:
        return {
            "model": "fireworks_ai/whisper-v3",
            "api_base": "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.FIREWORKS_AI


@pytest.mark.parametrize(
    "disable_add_transform_inline_image_block",
    [True, False],
)
def test_document_inlining_example(disable_add_transform_inline_image_block):
    litellm.set_verbose = True
    if disable_add_transform_inline_image_block is True:
        with pytest.raises(Exception):
            completion = litellm.completion(
                model="fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://storage.googleapis.com/fireworks-public/test/sample_resume.pdf"
                                },
                            },
                            {
                                "type": "text",
                                "text": "What are the candidate's BA and MBA GPAs?",
                            },
                        ],
                    }
                ],
                disable_add_transform_inline_image_block=disable_add_transform_inline_image_block,
            )
    else:
        completion = litellm.completion(
            model="fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": "this is a test request, write a short poem",
                },
            ],
            disable_add_transform_inline_image_block=disable_add_transform_inline_image_block,
        )
        print(completion)


@pytest.mark.parametrize(
    "content, model, expected_url",
    [
        (
            {"image_url": "http://example.com/image.png"},
            "gpt-4",
            "http://example.com/image.png#transform=inline",
        ),
        (
            {"image_url": {"url": "http://example.com/image.png"}},
            "gpt-4",
            {"url": "http://example.com/image.png#transform=inline"},
        ),
        (
            {"image_url": "http://example.com/image.png"},
            "vision-gpt",
            "http://example.com/image.png",
        ),
    ],
)
def test_transform_inline(content, model, expected_url):

    result = litellm.FireworksAIConfig()._add_transform_inline_image_block(
        content=content, model=model, disable_add_transform_inline_image_block=False
    )
    if isinstance(expected_url, str):
        assert result["image_url"] == expected_url
    else:
        assert result["image_url"]["url"] == expected_url["url"]


@pytest.mark.parametrize(
    "model, is_disabled, expected_url",
    [
        ("gpt-4", True, "http://example.com/image.png"),
        ("vision-gpt", False, "http://example.com/image.png"),
        ("gpt-4", False, "http://example.com/image.png#transform=inline"),
    ],
)
def test_global_disable_flag(model, is_disabled, expected_url):
    content = {"image_url": "http://example.com/image.png"}
    result = litellm.FireworksAIConfig()._add_transform_inline_image_block(
        content=content,
        model=model,
        disable_add_transform_inline_image_block=is_disabled,
    )
    assert result["image_url"] == expected_url
    litellm.disable_add_transform_inline_image_block = False  # Reset for other tests


def test_global_disable_flag_with_transform_messages_helper(monkeypatch):
    from openai import OpenAI
    from unittest.mock import patch
    from litellm import completion

    monkeypatch.setattr(litellm, "disable_add_transform_inline_image_block", True)

    client = OpenAI()

    with patch.object(
        client.chat.completions.with_raw_response,
        "create",
    ) as mock_post:
        try:
            completion(
                model="fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                },
                            },
                        ],
                    }
                ],
                client=client,
            )
        except Exception as e:
            print(e)

        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs)
        assert (
            "#transform=inline"
            not in mock_post.call_args.kwargs["messages"][0]["content"][1]["image_url"][
                "url"
            ]
        )
