import httpx

from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.llms.gemini.image_generation.transformation import GoogleImageGenConfig
from litellm.types.utils import ImageResponse


def test_gemini_image_generation_request_uses_shared_generation_config():
    config = GoogleImageGenConfig()

    request = config.transform_image_generation_request(
        model="gemini-3.1-flash-image-preview",
        prompt="Generate a simple app icon",
        optional_params={
            "sampleCount": 2,
            "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"},
        },
        litellm_params={},
        headers={},
    )

    assert request["contents"][0]["parts"] == [{"text": "Generate a simple app icon"}]
    assert request["generationConfig"] == {
        "response_modalities": ["IMAGE", "TEXT"],
        "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"},
        "candidateCount": 2,
    }


def test_gemini_image_generation_map_openai_params_maps_n_size_and_image_config():
    config = GoogleImageGenConfig()

    mapped = config.map_openai_params(
        non_default_params={
            "n": 2,
            "size": "768x1376",
            "imageConfig": {"aspectRatio": "1:1", "imageSize": "512"},
        },
        optional_params={},
        model="gemini-3.1-flash-image-preview",
        drop_params=False,
    )

    assert mapped == {
        "sampleCount": 2,
        "imageConfig": {"aspectRatio": "1:1", "imageSize": "512"},
    }


def test_imagen_generation_with_provider_prefix_uses_imagen_params_and_response():
    config = GoogleImageGenConfig()

    mapped = config.map_openai_params(
        non_default_params={
            "n": 1,
            "size": "1024x1024",
        },
        optional_params={},
        model="gemini/imagen-4.0-generate-001",
        drop_params=False,
    )
    assert mapped == {
        "sampleCount": 1,
        "aspectRatio": "1:1",
        "imageSize": "1K",
    }

    request = config.transform_image_generation_request(
        model="gemini/imagen-4.0-generate-001",
        prompt="Generate a simple app icon",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )
    assert request == {
        "instances": [{"prompt": "Generate a simple app icon"}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "1:1",
            "imageSize": "1K",
        },
    }

    result = config.transform_image_generation_response(
        model="gemini/imagen-4.0-generate-001",
        raw_response=httpx.Response(
            status_code=200,
            json={
                "predictions": [
                    {
                        "bytesBase64Encoded": "fake-imagen-image",
                    }
                ]
            },
        ),
        model_response=ImageResponse(data=[]),
        logging_obj=None,
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert result.data is not None
    assert result.data[0].b64_json == "fake-imagen-image"


def test_imagen_generation_forwards_mapped_openai_size_image_size():
    config = GoogleImageGenConfig()

    mapped = config.map_openai_params(
        non_default_params={
            "size": "512x512",
        },
        optional_params={},
        model="gemini/imagen-4.0-generate-001",
        drop_params=False,
    )
    assert mapped == {"aspectRatio": "1:1", "imageSize": "512"}

    request = config.transform_image_generation_request(
        model="gemini/imagen-4.0-generate-001",
        prompt="Generate a simple app icon",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert request == {
        "instances": [{"prompt": "Generate a simple app icon"}],
        "parameters": {"aspectRatio": "1:1", "imageSize": "512"},
    }


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
                "candidatesTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 213},
                    {"modality": "IMAGE", "tokenCount": 1120},
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
    assert usage["completion_tokens_details"]["text_tokens"] == 596
    assert usage["completion_tokens_details"]["image_tokens"] == 1120
    assert usage["output_tokens_details"]["text_tokens"] == 596
    assert usage["output_tokens_details"]["image_tokens"] == 1120

    logging_usage = StandardLoggingPayloadSetup.get_usage_as_dict(
        response_obj=result.model_dump()
    )
    assert logging_usage["completion_tokens_details"]["text_tokens"] == 596
    assert logging_usage["completion_tokens_details"]["image_tokens"] == 1120


def test_gemini_image_generation_usage_without_output_details_treats_output_as_image():
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
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 35}],
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
    assert usage["completion_tokens_details"]["text_tokens"] == 0
    assert usage["completion_tokens_details"]["image_tokens"] == 1716
