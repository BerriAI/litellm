import os
import sys
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(
    "../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.centml.completion.transformation import CentmlTextCompletionConfig
from litellm.types.llms.openai import OpenAITextCompletionUserMessage


class TestCentmlCompletionTransformation:

    def setup_method(self):
        self.config = CentmlTextCompletionConfig()
        self.model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        self.logging_obj = MagicMock()

    def test_transform_prompt_single_string(self):
        """Test transforming a single string message"""
        messages = [OpenAITextCompletionUserMessage(content="Hello, world!")]

        result = self.config._transform_prompt(messages)

        assert result == "Hello, world!"
        assert isinstance(result, str)

    def test_transform_prompt_single_string_in_list(self):
        """Test transforming a list with a single string (as a proper message object)"""
        messages = [{"content": "Hello, world!"}]

        result = self.config._transform_prompt(messages)

        assert result == "Hello, world!"
        assert isinstance(result, str)

    def test_transform_prompt_multiple_strings_raises_error(self):
        """Test that multiple prompts raise an error"""
        messages = [{"content": "Hello"}, {"content": "World"}]

        with pytest.raises(ValueError,
                           match="CentML does not support multiple prompts"):
            self.config._transform_prompt(messages)

    def test_transform_prompt_integers_raises_error(self):
        """Test that integer tokens raise an error"""
        messages = [{"content": [123, 456, 789]}]  # Token IDs as content

        with pytest.raises(ValueError,
                           match="CentML does not support integers as input"):
            self.config._transform_prompt(messages)

    def test_transform_prompt_mixed_integers_raises_error(self):
        """Test that token arrays raise an error (realistic token scenario)"""
        # Test with pure token array (which should get to our validation)
        messages = [{"content": [123, 456, 789]}]  # Pure token array

        with pytest.raises(ValueError,
                           match="CentML does not support integers as input"):
            self.config._transform_prompt(messages)

    def test_transform_text_completion_request(self):
        """Test the complete text completion request transformation"""
        messages = [
            OpenAITextCompletionUserMessage(
                content="Complete this sentence: The weather today is")
        ]
        optional_params = {"max_tokens": 100, "temperature": 0.7, "top_p": 0.9}
        headers = {}

        result = self.config.transform_text_completion_request(
            model=self.model,
            messages=messages,
            optional_params=optional_params,
            headers=headers)

        expected = {
            "model": self.model,
            "prompt": "Complete this sentence: The weather today is",
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }

        assert result == expected

    def test_transform_text_completion_request_empty_optional_params(self):
        """Test transformation with empty optional parameters"""
        messages = [OpenAITextCompletionUserMessage(content="Tell me a joke")]
        optional_params = {}
        headers = {}

        result = self.config.transform_text_completion_request(
            model=self.model,
            messages=messages,
            optional_params=optional_params,
            headers=headers)

        expected = {"model": self.model, "prompt": "Tell me a joke"}

        assert result == expected

    def test_transform_prompt_complex_user_message(self):
        """Test transforming OpenAI text completion user message objects"""
        messages = [
            OpenAITextCompletionUserMessage(
                content="What is the capital of France?"),
        ]

        result = self.config._transform_prompt(messages)

        assert result == "What is the capital of France?"
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        "prompt_input,expected_output",
        [
            # Single OpenAI message format
            ([OpenAITextCompletionUserMessage(content="Hello world")
              ], "Hello world"),

            # Single dict message format
            ([{
                "content": "Hello world"
            }], "Hello world"),

            # OpenAI message format
            ([OpenAITextCompletionUserMessage(content="Hello world")
              ], "Hello world"),

            # Multi-line prompt
            ([
                OpenAITextCompletionUserMessage(
                    content="Line 1\nLine 2\nLine 3")
            ], "Line 1\nLine 2\nLine 3"),

            # Empty string
            ([OpenAITextCompletionUserMessage(content="")], ""),

            # Prompt with special characters
            ([
                OpenAITextCompletionUserMessage(
                    content="Hello! @#$%^&*()_+ world?")
            ], "Hello! @#$%^&*()_+ world?"),
        ])
    def test_transform_prompt_valid_inputs(self, prompt_input,
                                           expected_output):
        """Test various valid prompt inputs"""
        # prompt_input is already in the correct message list format
        result = self.config._transform_prompt(prompt_input)
        assert result == expected_output

    @pytest.mark.parametrize(
        "invalid_input",
        [
            # Integer tokens as content
            [{
                "content": [123, 456]
            }],

            # Pure token list to trigger integer validation
            [{
                "content": [789, 101112]
            }],

            # Multiple string messages
            [{
                "content": "Hello"
            }, {
                "content": "World"
            }],

            # Multiple OpenAI messages
            [
                OpenAITextCompletionUserMessage(content="Hello"),
                OpenAITextCompletionUserMessage(content="World")
            ],
        ])
    def test_transform_prompt_invalid_inputs(self, invalid_input):
        """Test various invalid prompt inputs that should raise errors"""
        with pytest.raises(ValueError):
            self.config._transform_prompt(invalid_input)
