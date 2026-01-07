"""
Test for google_endpoints/endpoints.py
"""
import pytest
import sys, os
from dotenv import load_dotenv


from litellm.proxy.google_endpoints.endpoints import google_count_tokens
from litellm.types.llms.vertex_ai import TokenCountDetailsResponse
from starlette.requests import Request

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../../../..")
)

@pytest.mark.asyncio
async def test_proxy_gemini_to_openai_like_model_token_counting():
    """
    Test the token counting endpoint for proxing gemini to openai-like models.
    """
    response: TokenCountDetailsResponse = await google_count_tokens(
        request=Request(
            scope={
                "type": "http",
                "parsed_body": (
                    [
                        "contents"
                    ],
                    {
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": "Hello, how are you?"
                                    }
                                ]
                            }
                        ]
                    }
                )
            }
        ),
        model_name="volcengine/foo",
    )

    assert response.get("totalTokens") > 0