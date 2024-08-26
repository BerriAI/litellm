import os
import sys
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from dotenv import load_dotenv

import litellm


def generate_text():
    try:
        litellm.set_verbose = True
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://avatars.githubusercontent.com/u/17561003?v=4"
                        },
                    },
                ],
            }
        ]
        response = litellm.completion(
            model="gemini/gemini-pro-vision",
            messages=messages,
            stop="Hello world",
            num_retries=3,
        )
        print(response)
        assert isinstance(response.choices[0].message.content, str) == True
    except Exception as exception:
        raise Exception("An error occurred during text generation:", exception)


# generate_text()


from unittest.mock import AsyncMock, patch

import litellm


@pytest.mark.asyncio
async def test_fine_tuned_model_mock():
    mock_response = AsyncMock()

    def return_val():
        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "text": "Hello! I'm doing well, thank you for asking. How can I assist you today?"
                            }
                        ],
                    },
                    "finishReason": "STOP",
                    "safetyRatings": [
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "NEGLIGIBLE",
                            "probabilityScore": 0.01,
                            "severity": "HARM_SEVERITY_NEGLIGIBLE",
                            "severityScore": 0.02,
                        },
                    ],
                    "avgLogprobs": -0.5,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 15,
                "totalTokenCount": 20,
            },
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    expected_payload = {
        "contents": [{"role": "user", "parts": [{"text": "Hello, how are you?"}]}],
        "generationConfig": {},
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.acompletion(
            model="gemini/tunedModels/test-hkx8uhx16ylg",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            gemini_project="gen-lang-client-0594097422",
            api_key="anything",
        )

        mock_post.assert_called_once()
        url, kwargs = mock_post.call_args
        print("url = ", url)

        assert (
            url[0]
            == "https://generativelanguage.googleapis.com/v1beta/tunedModels/test-hkx8uhx16ylg:generateContent"
        )

        assert kwargs["headers"] == {
            "Authorization": "Bearer anything",
            "x-goog-user-project": "gen-lang-client-0594097422",
        }

        print("call args = ", kwargs)
        args_to_gemini = kwargs["json"]

        print("args to Gemini call:", args_to_gemini)

        assert args_to_gemini == expected_payload
        assert response.choices[0].message.content.startswith("Hello! I'm doing well")
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.total_tokens == 20

        print("Arguments passed to Gemini:", args_to_gemini)
        print("Response:", response)
