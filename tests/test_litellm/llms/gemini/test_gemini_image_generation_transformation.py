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


def test_gemini_image_generation_web_search_options_maps_to_google_search_tool():
    config = GoogleImageGenConfig()

    mapped = config.map_openai_params(
        non_default_params={"web_search_options": {}},
        optional_params={},
        model="gemini-3.1-flash-image-preview",
        drop_params=False,
    )

    assert "tools" in mapped
    assert mapped["tools"] == [{"googleSearch": {}}]

    request = config.transform_image_generation_request(
        model="gemini-3.1-flash-image-preview",
        prompt="Generate an image of the latest iPhone",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert request["tools"] == [{"googleSearch": {}}]


def test_gemini_image_generation_openai_web_search_tool_maps_to_google_search():
    config = GoogleImageGenConfig()

    mapped = config.map_openai_params(
        non_default_params={"tools": [{"type": "web_search"}]},
        optional_params={},
        model="gemini-3.1-flash-image-preview",
        drop_params=False,
    )

    assert mapped["tools"] == [{"googleSearch": {}}]

    request = config.transform_image_generation_request(
        model="gemini-3.1-flash-image-preview",
        prompt="Generate an image of the latest iPhone",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert request["tools"] == [{"googleSearch": {}}]


def test_gemini_image_generation_dedupes_search_tools_from_tools_and_web_search_options():
    config = GoogleImageGenConfig()

    mapped = config.map_openai_params(
        non_default_params={
            "tools": [{"type": "web_search"}],
            "web_search_options": {},
        },
        optional_params={},
        model="gemini-3.1-flash-image-preview",
        drop_params=False,
    )

    assert mapped["tools"] == [{"googleSearch": {}}]


def test_gemini_image_generation_preserves_tool_config_side_effect():
    config = GoogleImageGenConfig()

    mapped = config.map_openai_params(
        non_default_params={
            "tools": [{"googleMaps": {"latitude": 37.7, "longitude": -122.4}}]
        },
        optional_params={},
        model="gemini-3.1-flash-image-preview",
        drop_params=False,
    )

    assert mapped["tools"] == [{"googleMaps": {}}]
    assert mapped["toolConfig"] == {
        "retrievalConfig": {"latLng": {"latitude": 37.7, "longitude": -122.4}}
    }

    request = config.transform_image_generation_request(
        model="gemini-3.1-flash-image-preview",
        prompt="Generate an image of a coffee shop nearby",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert request["tools"] == [{"googleMaps": {}}]
    assert request["toolConfig"] == {
        "retrievalConfig": {"latLng": {"latitude": 37.7, "longitude": -122.4}}
    }


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


def test_gemini_image_generation_response_tracks_web_search_requests():
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
                    },
                    "groundingMetadata": {
                        "webSearchQueries": ["latest iphone", "iphone colors"]
                    },
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

    assert result.usage.web_search_requests == 2


def test_gemini_image_generation_response_without_grounding_has_no_web_search_requests():
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

    assert getattr(result.usage, "web_search_requests", None) is None
