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
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path
import litellm


def test_convert_chat_completion_messages_to_responses_api_image_input():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    handler = LiteLLMResponsesTransformationHandler()

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

    response, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    response_str = json.dumps(response)

    assert user_content in response_str
    assert user_image in response_str

    print("response: ", response)
    assert response[0]["content"][1]["image_url"] == user_image
