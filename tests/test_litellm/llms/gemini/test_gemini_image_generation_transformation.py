import httpx

from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.llms.gemini.image_generation.transformation import GoogleImageGenConfig
from litellm.types.utils import ImageResponse


def test_gemini_image_generation_usage_includes_chat_token_details():
    config = GoogleImageGenConfig()
    raw_response = httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "fake-image",
                                }
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 35,
                "candidatesTokenCount": 1716,
                "totalTokenCount": 1751,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 30},
                    {"modality": "IMAGE", "tokenCount": 5},
                ],
            },
        },
    )

    result = config.transform_image_generation_response(
        model="gemini-3.1-flash-image-preview",
        raw_response=raw_response,
        model_response=ImageResponse(data=[]),
        logging_obj=None,
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    usage = result.model_dump()["usage"]

    assert usage["input_tokens"] == 35
    assert usage["output_tokens"] == 1716
    assert usage["prompt_tokens"] == 35
    assert usage["completion_tokens"] == 1716
    assert usage["prompt_tokens_details"]["image_tokens"] == 5
    assert usage["completion_tokens_details"]["image_tokens"] == 1716
    assert usage["output_tokens_details"]["image_tokens"] == 1716

    logging_usage = StandardLoggingPayloadSetup.get_usage_as_dict(
        response_obj=result.model_dump()
    )
    assert logging_usage["completion_tokens_details"]["image_tokens"] == 1716
