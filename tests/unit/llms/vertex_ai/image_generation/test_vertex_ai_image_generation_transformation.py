import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError


import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.vertex_ai.image_generation import (
    get_vertex_ai_image_generation_config,
)
from litellm.llms.vertex_ai.image_generation.vertex_gemini_transformation import (
    VertexAIGeminiImageGenerationConfig,
)
from litellm.llms.vertex_ai.image_generation.vertex_imagen_transformation import (
    VertexAIImagenImageGenerationConfig,
)
from litellm.types.utils import ImageResponse


class TestVertexAIGeminiImageGenerationConfig:
    def setup_method(self):
        """Set up test fixtures"""
        self.config = VertexAIGeminiImageGenerationConfig()

    def test_get_supported_openai_params(self):
        """Test get_supported_openai_params returns correct params"""
        supported = self.config.get_supported_openai_params("gemini-2.5-flash-image")
        assert "n" in supported
        assert "size" in supported

    def test_map_openai_params_n(self):
        """Test mapping n parameter to candidate_count"""
        non_default_params = {"n": 3}
        optional_params = {}
        result = self.config.map_openai_params(non_default_params, optional_params, "gemini-2.5-flash-image", False)
        assert result.get("candidate_count") == 3

    def test_map_openai_params_size(self):
        """Test mapping size parameter to aspectRatio"""
        non_default_params = {"size": "1024x1024"}
        optional_params = {}
        result = self.config.map_openai_params(non_default_params, optional_params, "gemini-2.5-flash-image", False)
        assert result.get("aspectRatio") == "1:1"

    def test_map_openai_params_size_16_9(self):
        """Test mapping 16:9 size"""
        non_default_params = {"size": "1792x1024"}
        optional_params = {}
        result = self.config.map_openai_params(non_default_params, optional_params, "gemini-2.5-flash-image", False)
        assert result.get("aspectRatio") == "16:9"

    def test_map_size_to_aspect_ratio(self):
        """Test size to aspect ratio mapping"""
        assert self.config._map_size_to_aspect_ratio("1024x1024") == "1:1"
        assert self.config._map_size_to_aspect_ratio("1792x1024") == "16:9"
        assert self.config._map_size_to_aspect_ratio("1024x1792") == "9:16"
        assert self.config._map_size_to_aspect_ratio("1280x896") == "4:3"
        assert self.config._map_size_to_aspect_ratio("896x1280") == "3:4"
        assert self.config._map_size_to_aspect_ratio("unknown") == "1:1"  # default

    def test_get_supported_openai_params_includes_native_gemini_params(self):
        """Test that native Gemini imageConfig params are supported"""
        supported = self.config.get_supported_openai_params("gemini-3-pro-image-preview")
        assert "aspectRatio" in supported
        assert "aspect_ratio" in supported
        assert "imageSize" in supported
        assert "image_size" in supported
        assert "imageConfig" in supported

    def test_map_openai_params_aspect_ratio_camel_case(self):
        """Test mapping native aspectRatio parameter"""
        result = self.config.map_openai_params({"aspectRatio": "9:16"}, {}, "gemini-3-pro-image-preview", False)
        assert result["aspectRatio"] == "9:16"

    def test_map_openai_params_aspect_ratio_snake_case(self):
        """Test mapping native aspect_ratio parameter"""
        result = self.config.map_openai_params({"aspect_ratio": "16:9"}, {}, "gemini-3-pro-image-preview", False)
        assert result["aspectRatio"] == "16:9"

    def test_map_openai_params_image_size_camel_case(self):
        """Test mapping native imageSize parameter"""
        result = self.config.map_openai_params({"imageSize": "4K"}, {}, "gemini-3-pro-image-preview", False)
        assert result["imageSize"] == "4K"

    def test_map_openai_params_image_size_snake_case(self):
        """Test mapping native image_size parameter"""
        result = self.config.map_openai_params({"image_size": "2K"}, {}, "gemini-3-pro-image-preview", False)
        assert result["imageSize"] == "2K"

    def test_map_openai_params_image_config_dict_stored_whole(self):
        """imageConfig dict is stored as-is so all fields survive"""
        result = self.config.map_openai_params(
            {"imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"}},
            {},
            "gemini-3.1-flash-image",
            False,
        )
        assert result["imageConfig"] == {"aspectRatio": "16:9", "imageSize": "2K"}

    def test_map_openai_params_image_config_all_fields(self):
        """All ImageConfig fields (personGeneration, imageOutputOptions) pass through"""
        payload = {
            "imageConfig": {
                "aspectRatio": "9:16",
                "imageSize": "4K",
                "personGeneration": "DONT_ALLOW",
                "imageOutputOptions": {
                    "mimeType": "image/jpeg",
                    "compressionQuality": 80,
                },
            }
        }
        result = self.config.map_openai_params(payload, {}, "gemini-3.1-flash-image", False)
        assert result["imageConfig"] == payload["imageConfig"]

    def test_map_openai_params_image_config_non_dict_warns_and_drops(self):
        """Non-dict imageConfig is dropped with a warning, not silently discarded"""
        with patch("litellm.llms.vertex_ai.image_generation.vertex_gemini_transformation.verbose_logger") as mock_log:
            result = self.config.map_openai_params(
                {"imageConfig": "bad-string-value"}, {}, "gemini-3.1-flash-image", False
            )
        assert "imageConfig" not in result
        mock_log.warning.assert_called_once()

    def test_transform_image_generation_request_from_image_config(self):
        """Full imageConfig dict is forwarded verbatim into generationConfig"""
        full_config = {
            "aspectRatio": "16:9",
            "imageSize": "2K",
            "personGeneration": "DONT_ALLOW",
            "imageOutputOptions": {"mimeType": "image/jpeg", "compressionQuality": 85},
        }
        mapped = self.config.map_openai_params(
            {"imageConfig": full_config},
            {},
            "gemini-3.1-flash-image",
            False,
        )
        request = self.config.transform_image_generation_request(
            model="gemini-3.1-flash-image",
            prompt="A nano banana on a desk",
            optional_params=mapped,
            litellm_params={},
            headers={},
        )
        assert request["generationConfig"]["imageConfig"] == full_config

    def test_transform_image_generation_flat_params_override_image_config(self):
        """Explicit flat params win over the same key inside imageConfig"""
        request = self.config.transform_image_generation_request(
            model="gemini-3.1-flash-image",
            prompt="A nano banana",
            optional_params={
                "imageConfig": {"aspectRatio": "1:1", "personGeneration": "DONT_ALLOW"},
                "aspectRatio": "16:9",  # should win
            },
            litellm_params={},
            headers={},
        )
        assert request["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
        assert request["generationConfig"]["imageConfig"]["personGeneration"] == "DONT_ALLOW"

    def test_transform_image_generation_request_basic(self):
        """Test basic request transformation"""
        request = self.config.transform_image_generation_request(
            model="gemini-2.5-flash-image",
            prompt="A nano banana",
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert "contents" in request
        assert "generationConfig" in request
        assert request["generationConfig"]["responseModalities"] == ["IMAGE"]
        assert request["contents"][0]["parts"][0]["text"] == "A nano banana"

    def test_transform_image_generation_request_with_aspect_ratio(self):
        """Test request transformation with aspectRatio"""
        request = self.config.transform_image_generation_request(
            model="gemini-2.5-flash-image",
            prompt="A nano banana",
            optional_params={"aspectRatio": "16:9"},
            litellm_params={},
            headers={},
        )
        assert request["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"

    def test_transform_image_generation_request_with_image_size(self):
        """Test request transformation with imageSize (Gemini 3 Pro)"""
        request = self.config.transform_image_generation_request(
            model="gemini-3-pro-image-preview",
            prompt="A nano banana",
            optional_params={"imageSize": "4K"},
            litellm_params={},
            headers={},
        )
        assert request["generationConfig"]["imageConfig"]["imageSize"] == "4K"

    def test_map_openai_params_web_search_options(self):
        """Test web_search_options maps to googleSearch tool"""
        result = self.config.map_openai_params({"web_search_options": {}}, {}, "gemini-3.1-flash-image-preview", False)
        assert result["tools"] == [{"googleSearch": {}}]

    def test_transform_image_generation_request_with_web_search_tools(self):
        """Test request transformation includes googleSearch tools"""
        request = self.config.transform_image_generation_request(
            model="gemini-3.1-flash-image-preview",
            prompt="Generate an image of the latest iPhone",
            optional_params={"tools": [{"googleSearch": {}}]},
            litellm_params={},
            headers={},
        )
        assert request["tools"] == [{"googleSearch": {}}]

    def test_transform_image_generation_request_forwards_tool_config(self):
        """Test request transformation forwards toolConfig side-effects from tool mapping"""
        mapped = self.config.map_openai_params(
            {"tools": [{"googleMaps": {"latitude": 37.7, "longitude": -122.4}}]},
            {},
            "gemini-3.1-flash-image-preview",
            False,
        )
        request = self.config.transform_image_generation_request(
            model="gemini-3.1-flash-image-preview",
            prompt="Generate an image of a coffee shop nearby",
            optional_params=mapped,
            litellm_params={},
            headers={},
        )
        assert request["tools"] == [{"googleMaps": {}}]
        assert request["toolConfig"] == {"retrievalConfig": {"latLng": {"latitude": 37.7, "longitude": -122.4}}}

    def test_transform_image_generation_request_with_candidate_count(self):
        """Test request transformation with candidate_count"""
        request = self.config.transform_image_generation_request(
            model="gemini-2.5-flash-image",
            prompt="A nano banana",
            optional_params={"candidate_count": 2},
            litellm_params={},
            headers={},
        )
        assert request["generationConfig"]["candidateCount"] == 2

    def test_transform_image_generation_request_with_n(self):
        """Test request transformation with n parameter"""
        request = self.config.transform_image_generation_request(
            model="gemini-2.5-flash-image",
            prompt="A nano banana",
            optional_params={"n": 2},
            litellm_params={},
            headers={},
        )
        assert request["generationConfig"]["candidateCount"] == 2

    def test_transform_image_generation_response(self):
        """Test response transformation"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "base64_encoded_image_data",
                                }
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 93,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 54,
                    },
                    {
                        "modality": "IMAGE",
                        "tokenCount": 39,
                    },
                ],
                "candidatesTokenCount": 17,
                "totalTokenCount": 110,
            },
        }
        mock_response.headers = {}

        from litellm.types.utils import ImageResponse

        model_response = ImageResponse()
        result = self.config.transform_image_generation_response(
            model="gemini-2.5-flash-image",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "base64_encoded_image_data"
        assert result.data[0].url is None
        assert result.usage.input_tokens == 93
        assert result.usage.input_tokens_details.text_tokens == 54
        assert result.usage.input_tokens_details.image_tokens == 39
        assert result.usage.output_tokens == 17
        assert result.usage.total_tokens == 110

    def test_transform_image_generation_response_multiple_images(self):
        """Test response transformation with multiple images"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "image1",
                                }
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "image2",
                                }
                            },
                        ]
                    }
                }
            ]
        }
        mock_response.headers = {}

        from litellm.types.utils import ImageResponse

        model_response = ImageResponse()
        result = self.config.transform_image_generation_response(
            model="gemini-2.5-flash-image",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 2
        assert result.data[0].b64_json == "image1"
        assert result.data[1].b64_json == "image2"

    def test_transform_image_generation_response_signature(self):
        """Test response transformation includes thoughtSignature for Gemini 3 Pro"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "base64_encoded_image_data",
                                },
                                "thoughtSignature": "test_signature_abc123",
                            }
                        ]
                    }
                }
            ]
        }
        mock_response.headers = {}

        from litellm.types.utils import ImageResponse

        model_response = ImageResponse()
        result = self.config.transform_image_generation_response(
            model="gemini-3-pro-image-preview",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "base64_encoded_image_data"
        assert result.data[0].provider_specific_fields["thought_signature"] == "test_signature_abc123"

    def test_transform_image_generation_response_tracks_web_search_requests(self):
        """Grounding queries are carried onto usage so search spend can be billed"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "base64_encoded_image_data",
                                }
                            }
                        ]
                    },
                    "groundingMetadata": {"webSearchQueries": ["eiffel tower", "paris skyline"]},
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 93,
                "candidatesTokenCount": 17,
                "totalTokenCount": 110,
            },
        }
        mock_response.headers = {}

        from litellm.types.utils import ImageResponse

        result = self.config.transform_image_generation_response(
            model="gemini-2.5-flash-image",
            raw_response=mock_response,
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.usage.web_search_requests == 2


class TestVertexAIImagenImageGenerationConfig:
    def setup_method(self):
        """Set up test fixtures"""
        self.config = VertexAIImagenImageGenerationConfig()

    def test_get_supported_openai_params(self):
        """Test get_supported_openai_params returns correct params"""
        supported = self.config.get_supported_openai_params("imagegeneration@006")
        assert "n" in supported
        assert "size" in supported

    def test_map_openai_params_n(self):
        """Test mapping n parameter to sampleCount"""
        non_default_params = {"n": 3}
        optional_params = {}
        result = self.config.map_openai_params(non_default_params, optional_params, "imagegeneration@006", False)
        assert result.get("sampleCount") == 3

    def test_map_openai_params_size(self):
        """Test mapping size parameter to aspectRatio"""
        non_default_params = {"size": "1024x1024"}
        optional_params = {}
        result = self.config.map_openai_params(non_default_params, optional_params, "imagegeneration@006", False)
        assert result.get("aspectRatio") == "1:1"

    def test_map_size_to_aspect_ratio(self):
        """Test size to aspect ratio mapping"""
        assert self.config._map_size_to_aspect_ratio("1024x1024") == "1:1"
        assert self.config._map_size_to_aspect_ratio("1792x1024") == "16:9"
        assert self.config._map_size_to_aspect_ratio("unknown") == "1:1"  # default

    def test_transform_image_generation_request_basic(self):
        """Test basic request transformation"""
        request = self.config.transform_image_generation_request(
            model="imagegeneration@006",
            prompt="A cat",
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert "instances" in request
        assert "parameters" in request
        assert request["instances"][0]["prompt"] == "A cat"
        assert request["parameters"]["sampleCount"] == 1

    def test_transform_image_generation_request_with_params(self):
        """Test request transformation with parameters"""
        request = self.config.transform_image_generation_request(
            model="imagegeneration@006",
            prompt="A cat",
            optional_params={"sampleCount": 2, "aspectRatio": "16:9"},
            litellm_params={},
            headers={},
        )
        assert request["parameters"]["sampleCount"] == 2
        assert request["parameters"]["aspectRatio"] == "16:9"

    def test_transform_image_generation_request_labels_from_metadata(self):
        """Billing labels from litellm_params.metadata.requester_metadata on predict body."""
        request = self.config.transform_image_generation_request(
            model="imagegeneration@006",
            prompt="A cat",
            optional_params={},
            litellm_params={"metadata": {"requester_metadata": {"team": "platform", "env": "prod"}}},
            headers={},
        )
        assert request["labels"] == {"team": "platform", "env": "prod"}
        assert "labels" not in request["parameters"]

    def test_transform_image_generation_response(self):
        """Test response transformation"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"predictions": [{"bytesBase64Encoded": "base64_encoded_image_data"}]}
        mock_response.headers = {}

        from litellm.types.utils import ImageResponse

        model_response = ImageResponse()
        result = self.config.transform_image_generation_response(
            model="imagegeneration@006",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "base64_encoded_image_data"
        assert result.data[0].url is None

    def test_transform_image_generation_response_multiple_images(self):
        """Test response transformation with multiple images"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {"bytesBase64Encoded": "image1"},
                {"bytesBase64Encoded": "image2"},
            ]
        }
        mock_response.headers = {}

        from litellm.types.utils import ImageResponse

        model_response = ImageResponse()
        result = self.config.transform_image_generation_response(
            model="imagegeneration@006",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 2
        assert result.data[0].b64_json == "image1"
        assert result.data[1].b64_json == "image2"


class TestGetVertexAIImageGenerationConfig:
    """Test the router function that selects the correct config"""

    def test_get_gemini_model_config(self):
        """Test that Gemini models return Gemini config"""
        config = get_vertex_ai_image_generation_config("gemini-2.5-flash-image")
        assert isinstance(config, VertexAIGeminiImageGenerationConfig)

        config = get_vertex_ai_image_generation_config("gemini-3-pro-image-preview")
        assert isinstance(config, VertexAIGeminiImageGenerationConfig)

        config = get_vertex_ai_image_generation_config("vertex_ai/gemini-2.5-flash-image")
        assert isinstance(config, VertexAIGeminiImageGenerationConfig)

    def test_get_imagen_model_config(self):
        """Test that Imagen models return Imagen config"""
        config = get_vertex_ai_image_generation_config("imagegeneration@006")
        assert isinstance(config, VertexAIImagenImageGenerationConfig)

        config = get_vertex_ai_image_generation_config("imagen-4.0-generate-001")
        assert isinstance(config, VertexAIImagenImageGenerationConfig)

        config = get_vertex_ai_image_generation_config("vertex_ai/imagegeneration@006")
        assert isinstance(config, VertexAIImagenImageGenerationConfig)

    def test_get_non_gemini_model_config(self):
        """Test that non-Gemini models default to Imagen config"""
        config = get_vertex_ai_image_generation_config("some-other-model")
        assert isinstance(config, VertexAIImagenImageGenerationConfig)


class TestVertexAIImageGenerationIntegration:
    """Integration tests for Vertex AI image generation"""


    def test_gemini_get_complete_url(self):
        """Test Gemini config URL generation"""
        config = VertexAIGeminiImageGenerationConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gemini-2.5-flash-image",
            optional_params={},
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "us-central1",
            },
        )
        assert "test-project" in url
        assert "us-central1" in url
        assert "gemini-2.5-flash-image" in url
        assert "generateContent" in url

    def test_imagen_get_complete_url(self):
        """Test Imagen config URL generation"""
        config = VertexAIImagenImageGenerationConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="imagegeneration@006",
            optional_params={},
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "us-central1",
            },
        )
        assert "test-project" in url
        assert "us-central1" in url
        assert "imagegeneration@006" in url
        assert "predict" in url


def _transform_gemini_response(payload: object) -> ImageResponse:
    return VertexAIGeminiImageGenerationConfig().transform_image_generation_response(
        model="gemini-2.5-flash-image",
        raw_response=httpx.Response(200, json=payload),
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_gemini_image_generation_response_maps_usage_by_modality():
    response = _transform_gemini_response(
        {
            "candidates": [{"content": {"parts": [{"inlineData": {"data": "aGVsbG8="}}]}}],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 7,
                "totalTokenCount": 12,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 3},
                    {"modality": "IMAGE", "tokenCount": 2},
                ],
            },
        }
    )

    assert [image.b64_json for image in response.data] == ["aGVsbG8="]
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 7
    assert response.usage.total_tokens == 12
    assert response.usage.input_tokens_details.text_tokens == 3
    assert response.usage.input_tokens_details.image_tokens == 2


@pytest.mark.parametrize("usage_metadata", [None, {}, [], "", 0])
def test_gemini_image_generation_response_with_falsy_usage_metadata_keeps_zeroed_usage(usage_metadata: object):
    response = _transform_gemini_response(
        {
            "candidates": [{"content": {"parts": []}, "groundingMetadata": {"webSearchQueries": ["a"]}}],
            "usageMetadata": usage_metadata,
        }
    )

    assert response.data == []
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (0, 0, 0)
    assert response.usage.web_search_requests == 1


@pytest.mark.parametrize("usage_metadata", ["not an object", ["not", "an", "object"], 7, True])
def test_gemini_image_generation_response_rejects_non_object_usage_metadata_without_echoing_it(
    usage_metadata: object,
):
    with pytest.raises(ValidationError) as exc_info:
        _transform_gemini_response({"candidates": [], "usageMetadata": usage_metadata})

    assert "input_value" not in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"candidates": []},
        {"candidates": ""},
        {"candidates": {}},
        {"candidates": [{}]},
        {"candidates": [{"finishReason": "SAFETY"}]},
        {"candidates": [{"content": {}}]},
        {"candidates": [{"content": {"parts": ""}}]},
        {"candidates": [{"content": {"parts": "plain text"}}]},
        {"candidates": [{"content": {"parts": {"text": "only text"}}}]},
        {"candidates": [{"content": {"parts": [{"text": "only text"}, "plain text", ["text"], [], ""]}}]},
        {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png"}}]}}]},
        {"candidates": [{"content": {"parts": [{"inlineData": "no image"}, {"inlineData": ["mimeType"]}]}}]},
        {"candidates": [{"content": {"parts": [{"inlineData": ""}, {"inlineData": []}, {"inlineData": {}}]}}]},
        {"candidates": [{"content": {"parts": [{"inline_data": {"data": "snake-case-is-not-read"}}]}}]},
    ],
)
def test_gemini_image_generation_response_without_inline_image_data_has_no_images(payload: object):
    assert _transform_gemini_response(payload).data == []


def test_gemini_image_generation_response_keeps_image_order_and_thought_signatures():
    response = _transform_gemini_response(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "caption"},
                            {"inlineData": {"mimeType": "image/png", "data": "first"}, "thoughtSignature": "sig-1"},
                            "plain text",
                        ]
                    }
                },
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"data": "second"}},
                            {"inlineData": {"data": None}, "thoughtSignature": ""},
                            {"inlineData": {"data": "fourth"}, "thoughtSignature": {"nested": ["sig"]}},
                        ]
                    }
                },
            ]
        }
    )

    assert [(image.b64_json, image.provider_specific_fields) for image in response.data or []] == [
        ("first", {"thought_signature": "sig-1"}),
        ("second", None),
        (None, None),
        ("fourth", {"thought_signature": {"nested": ["sig"]}}),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        ["vertex-secret"],
        "vertex-secret",
        {"candidates": None},
        {"candidates": 7},
        {"candidates": "vertex-secret"},
        {"candidates": ["vertex-secret"]},
        {"candidates": [{"content": None}]},
        {"candidates": [{"content": "vertex-secret"}]},
        {"candidates": [{"content": ["vertex-secret"]}]},
        {"candidates": [{"content": {"parts": None}}]},
        {"candidates": [{"content": {"parts": [None]}}]},
        {"candidates": [{"content": {"parts": [7]}}]},
        {"candidates": [{"content": {"parts": ["inlineData vertex-secret"]}}]},
        {"candidates": [{"content": {"parts": [["inlineData", "vertex-secret"]]}}]},
        {"candidates": [{"content": {"parts": {"inlineData": "vertex-secret"}}}]},
        {"candidates": [{"content": {"parts": [{"inlineData": None}]}}]},
        {"candidates": [{"content": {"parts": [{"inlineData": 7}]}}]},
        {"candidates": [{"content": {"parts": [{"inlineData": "data vertex-secret"}]}}]},
        {"candidates": [{"content": {"parts": [{"inlineData": ["data", "vertex-secret"]}]}}]},
    ],
)
def test_gemini_image_generation_response_rejects_malformed_candidates_without_echoing_them(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform_gemini_response(payload)

    assert "vertex-secret" not in str(exc_info.value)


def _gemini_rejection_text(payload: object) -> str:
    with pytest.raises(ValidationError) as exc_info:
        _transform_gemini_response(payload)

    return str(exc_info.value)


@pytest.mark.parametrize("position", [403, 429])
@pytest.mark.parametrize("malformed_entry", [None, 7, "inlineData"])
def test_gemini_image_generation_response_rejection_text_is_the_same_wherever_the_malformed_entry_sits(
    position: int, malformed_entry: object
):
    late_parts = [*[{"text": "caption"}] * position, malformed_entry]
    late_candidates = [*[{"finishReason": "STOP"}] * position, malformed_entry]

    assert _gemini_rejection_text({"candidates": [{"content": {"parts": late_parts}}]}) == _gemini_rejection_text(
        {"candidates": [{"content": {"parts": [malformed_entry]}}]}
    )
    assert _gemini_rejection_text({"candidates": late_candidates}) == _gemini_rejection_text(
        {"candidates": [malformed_entry]}
    )


_IMAGEN_IMAGE_KEY = "bytesBase64Encoded"


def _imagen_predict_response(payload: object) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode(), headers={"content-type": "application/json"})


def _transform_imagen_response(payload: object, model_response: ImageResponse) -> ImageResponse:
    return VertexAIImagenImageGenerationConfig().transform_image_generation_response(
        model="imagegeneration@006",
        raw_response=_imagen_predict_response(payload),
        model_response=model_response,
        logging_obj=Logging(
            model="imagegeneration@006",
            messages=[],
            stream=False,
            call_type="image_generation",
            start_time=datetime(2026, 1, 1),
            litellm_call_id="vertex-imagen-generation-test",
            function_id="vertex-imagen-generation-test",
        ),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def _imagen_rejection_text(payload: object) -> str:
    with pytest.raises(ValidationError) as exc_info:
        _transform_imagen_response(payload, ImageResponse())

    return str(exc_info.value)


def _generate_imagen_image(payload: object) -> ImageResponse:
    transport = httpx.MockTransport(lambda request: _imagen_predict_response(payload))
    return litellm.image_generation(
        model="vertex_ai/imagegeneration@006",
        prompt="a cat",
        api_base="https://vertex.invalid/v1/predict",
        client=HTTPHandler(client=httpx.Client(transport=transport)),
    )


def test_imagen_image_generation_response_appends_prediction_images_in_order_to_the_given_response():
    model_response = ImageResponse(data=[{"b64_json": "seeded", "url": None}])
    seeded_image = model_response.data[0]

    result = _transform_imagen_response(
        {
            "predictions": [
                {_IMAGEN_IMAGE_KEY: "first", "mimeType": "image/png"},
                {"raiFilteredReason": "filtered"},
                "plain text",
                [],
                {_IMAGEN_IMAGE_KEY: "second"},
            ],
            "deployedModelId": "123",
        },
        model_response,
    )

    assert result is model_response
    assert result.data[0] is seeded_image
    assert [(image.b64_json, image.url) for image in result.data] == [
        ("seeded", None),
        ("first", None),
        ("second", None),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"deployedModelId": "123"},
        {"predictions": []},
        {"predictions": ""},
        {"predictions": "plain text"},
        {"predictions": _IMAGEN_IMAGE_KEY},
        {"predictions": {}},
        {"predictions": {"mimeType": "image/png", "prompt": "a cat"}},
        {"predictions": [{}]},
        {"predictions": [{"mimeType": "image/png"}, {"raiFilteredReason": "filtered"}]},
        {"predictions": ["plain text", ""]},
        {"predictions": [["mimeType", "image/png"], []]},
        {"predictions": [[[_IMAGEN_IMAGE_KEY]]]},
        {"predictions": [{"bytes_base64_encoded": "snake-case-is-not-read"}]},
    ],
)
def test_imagen_image_generation_response_without_prediction_image_bytes_has_no_images(payload: object):
    assert _transform_imagen_response(payload, ImageResponse()).data == []


@pytest.mark.parametrize("b64_value", [None, "", "aGVsbG8=", "1.5", "true"])
def test_imagen_image_generation_response_keeps_the_image_bytes_value_exactly_as_sent(b64_value: object):
    result = _transform_imagen_response({"predictions": [{_IMAGEN_IMAGE_KEY: b64_value}]}, ImageResponse())

    assert [repr(image.b64_json) for image in result.data] == [repr(b64_value)]


@pytest.mark.parametrize("b64_value", [5, 1.5, True, ["aGVsbG8="], {"data": "aGVsbG8="}])
def test_imagen_image_generation_response_rejects_non_string_image_bytes_naming_b64_json(b64_value: object):
    with pytest.raises(ValidationError, match="b64_json"):
        _transform_imagen_response({"predictions": [{_IMAGEN_IMAGE_KEY: b64_value}]}, ImageResponse())


@pytest.mark.parametrize(
    "payload",
    [
        ["vertex-secret"],
        "vertex-secret",
        7,
        None,
        {"predictions": None},
        {"predictions": 7},
        {"predictions": True},
        {"predictions": [None]},
        {"predictions": [7]},
        {"predictions": [False]},
        {"predictions": [f"{_IMAGEN_IMAGE_KEY} vertex-secret"]},
        {"predictions": [[_IMAGEN_IMAGE_KEY, "vertex-secret"]]},
        {"predictions": {_IMAGEN_IMAGE_KEY: "vertex-secret"}},
        {"predictions": [{_IMAGEN_IMAGE_KEY: "kept"}, None]},
    ],
)
def test_imagen_image_generation_response_rejects_malformed_predictions_without_echoing_them(payload: object):
    rejection_text = _imagen_rejection_text(payload)

    assert "vertex-secret" not in rejection_text
    assert "input_value" not in rejection_text


@pytest.mark.parametrize("position", [403, 429])
@pytest.mark.parametrize("malformed_entry", [None, 7, _IMAGEN_IMAGE_KEY, [_IMAGEN_IMAGE_KEY]])
def test_imagen_image_generation_response_rejection_text_is_the_same_wherever_the_malformed_prediction_sits(
    position: int, malformed_entry: object
):
    late_predictions = [*[{_IMAGEN_IMAGE_KEY: "kept"}] * position, malformed_entry]

    assert _imagen_rejection_text({"predictions": late_predictions}) == _imagen_rejection_text(
        {"predictions": [malformed_entry]}
    )


def test_imagen_image_generation_returns_the_prediction_images_of_the_predict_endpoint():
    result = _generate_imagen_image(
        {"predictions": [{_IMAGEN_IMAGE_KEY: "first"}, {"mimeType": "image/png"}, {_IMAGEN_IMAGE_KEY: "second"}]}
    )

    assert [(image.b64_json, image.url) for image in result.data] == [("first", None), ("second", None)]


@pytest.mark.parametrize(
    "payload",
    [
        ["vertex-secret"],
        "vertex-secret",
        {"predictions": None},
        {"predictions": [*[{_IMAGEN_IMAGE_KEY: "kept"}] * 403, f"{_IMAGEN_IMAGE_KEY} vertex-secret"]},
        {"predictions": [*[{_IMAGEN_IMAGE_KEY: "kept"}] * 429, None]},
    ],
)
def test_imagen_image_generation_maps_a_malformed_predict_body_to_a_connection_error_without_echoing_it(
    payload: object,
):
    with pytest.raises(litellm.APIConnectionError) as exc_info:
        _generate_imagen_image(payload)

    assert type(exc_info.value) is litellm.APIConnectionError
    assert exc_info.value.status_code == 500
    assert "vertex-secret" not in str(exc_info.value)
