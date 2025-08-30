import datetime
import json
import os
import sys
import unittest
from typing import List, Optional, Tuple
from unittest.mock import ANY, MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
    OpenAiResponsesToChatCompletionStreamIterator,
)
from litellm.types.llms.openai import Reasoning
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


class TestLiteLLMResponsesTransformation:
    def setup_method(self):
        self.handler = LiteLLMResponsesTransformationHandler()
        self.model = "responses-api-model"
        self.logging_obj = MagicMock()

    def test_transform_request_reasoning_effort(self):
        """
        Test that reasoning_effort is mapped to reasoning parameter correctly.
        """
        # Case 1: reasoning_effort = "high"
        optional_params_high = {"reasoning_effort": "high"}
        result_high = self.handler.transform_request(
            model=self.model,
            messages=[],
            optional_params=optional_params_high,
            litellm_params={},
            headers={},
            litellm_logging_obj=self.logging_obj,
        )
        assert "reasoning" in result_high
        assert result_high["reasoning"] == Reasoning(effort="high", summary="detailed")

        # Case 2: reasoning_effort = "medium"
        optional_params_medium = {"reasoning_effort": "medium"}
        result_medium = self.handler.transform_request(
            model=self.model,
            messages=[],
            optional_params=optional_params_medium,
            litellm_params={},
            headers={},
            litellm_logging_obj=self.logging_obj,
        )
        assert "reasoning" in result_medium
        assert result_medium["reasoning"] == Reasoning(effort="medium", summary="auto")

        # Case 3: reasoning_effort = "low"
        optional_params_low = {"reasoning_effort": "low"}
        result_low = self.handler.transform_request(
            model=self.model,
            messages=[],
            optional_params=optional_params_low,
            litellm_params={},
            headers={},
            litellm_logging_obj=self.logging_obj,
        )
        assert "reasoning" in result_low
        assert result_low["reasoning"] == Reasoning(effort="low", summary="auto")

        # Case 4: no reasoning_effort
        optional_params_none = {}
        result_none = self.handler.transform_request(
            model=self.model,
            messages=[],
            optional_params=optional_params_none,
            litellm_params={},
            headers={},
            litellm_logging_obj=self.logging_obj,
        )
        assert "reasoning" in result_none
        assert result_none["reasoning"] == Reasoning(summary="auto")

        # Case 5: reasoning_effort = None
        optional_params_explicit_none = {"reasoning_effort": None}
        result_explicit_none = self.handler.transform_request(
            model=self.model,
            messages=[],
            optional_params=optional_params_explicit_none,
            litellm_params={},
            headers={},
            litellm_logging_obj=self.logging_obj,
        )
        assert "reasoning" in result_explicit_none
        assert result_explicit_none["reasoning"] == Reasoning(summary="auto")

    def test_convert_chat_completion_messages_to_responses_api_image_input(self):
        """
        Test that chat completion messages with image inputs are converted correctly.
        """
        user_content = "What's in this image?"
        user_image = "https://w7.pngwing.com/pngs/666/274/png-transparent-image-pictures-icon-photo-thumbnail.png"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_content,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": user_image},
                    },
                ],
            },
        ]

        response, _ = self.handler.convert_chat_completion_messages_to_responses_api(messages)

        response_str = json.dumps(response)

        assert user_content in response_str
        assert user_image in response_str

        print("response: ", response)
        assert response[0]["content"][1]["image_url"] == user_image

    def test_openai_responses_chunk_parser_reasoning_summary(self):
        """
        Test that OpenAI responses chunk parser handles reasoning summary correctly.
        """
        iterator = OpenAiResponsesToChatCompletionStreamIterator(
            streaming_response=None, sync_stream=True
        )

        chunk = {
            "delta": "**Compar",
            "item_id": "rs_686d544208748198b6912e27b7c299c00e24bd875d35bade",
            "output_index": 0,
            "sequence_number": 4,
            "summary_index": 0,
            "type": "response.reasoning_summary_text.delta",
        }

        result = iterator.chunk_parser(chunk)

        assert isinstance(result, ModelResponseStream)
        assert len(result.choices) == 1
        choice = result.choices[0]
        assert isinstance(choice, StreamingChoices)
        assert choice.index == 0
        delta = choice.delta
        assert isinstance(delta, Delta)
        assert delta.content is None
        assert delta.reasoning_content == "**Compar"
        assert delta.tool_calls is None
        assert delta.function_call is None
