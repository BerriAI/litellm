import base64
import json
import os
from datetime import datetime
from io import BytesIO
from typing import Dict
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.vertex_ai.image_edit.vertex_gemini_transformation import (
    VertexAIGeminiImageEditConfig,
)
from litellm.llms.vertex_ai.image_edit.vertex_imagen_transformation import (
    VertexAIImagenImageEditConfig,
)


class TestVertexAIGeminiImageEditTransformation:
    def setup_method(self) -> None:
        self.config = VertexAIGeminiImageEditConfig()
        self.model = "vertex_ai/gemini-2.5-flash"
        self.prompt = "Add neon lights in the background"
        self.logging_obj = MagicMock()

    def test_map_openai_params(self) -> None:
        """Test mapping OpenAI parameters to Vertex AI Gemini format"""
        optional_params: Dict[str, object] = {
            "size": "1792x1024",
        }

        mapped = self.config.map_openai_params(
            image_edit_optional_params=optional_params,  # type: ignore[arg-type]
            model=self.model,
            drop_params=False,
        )

        assert mapped["aspectRatio"] == "16:9"

    def test_get_complete_url(self) -> None:
        """Test URL generation for Vertex AI Gemini"""
        with patch.dict(
            os.environ,
            {
                "VERTEXAI_PROJECT": "test-project",
                "VERTEXAI_LOCATION": "us-central1",
            },
        ):
            url = self.config.get_complete_url(
                model="gemini-2.5-flash",
                api_base=None,
                litellm_params={},
            )
            assert "test-project" in url
            assert "us-central1" in url
            assert "generateContent" in url

    def test_transform_image_edit_request(self) -> None:
        """Test request transformation for Vertex AI Gemini"""
        image_bytes = b"fake_image_data"
        image = BytesIO(image_bytes)
        optional_params = {
            "aspectRatio": "1:1",
        }

        request_body_str, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=image,
            image_edit_optional_request_params=optional_params,
            litellm_params=MagicMock(),
            headers={},
        )

        assert files == []
        assert isinstance(request_body_str, str)

        request_body = json.loads(request_body_str)
        assert "contents" in request_body
        assert request_body["contents"]["role"] == "USER"

        parts = request_body["contents"]["parts"]
        assert parts[-1]["text"] == self.prompt

        inline_data = parts[0]["inlineData"]
        assert inline_data["mimeType"] == "image/png"
        assert base64.b64decode(inline_data["data"]) == image_bytes

        generation_config = request_body["generationConfig"]
        assert generation_config["response_modalities"] == ["IMAGE"]
        assert generation_config["image_config"]["aspect_ratio"] == "1:1"

    def test_transform_image_edit_response(self) -> None:
        """Test response transformation for Vertex AI Gemini"""
        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(b"image-one").decode(
                                        "utf-8"
                                    ),
                                }
                            }
                        ]
                    }
                }
            ]
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_payload
        mock_response.status_code = 200
        mock_response.headers = {}

        image_response = self.config.transform_image_edit_response(
            model=self.model,
            raw_response=mock_response,
            logging_obj=self.logging_obj,
        )

        assert image_response.data is not None
        assert len(image_response.data) == 1
        assert image_response.data[0].b64_json == base64.b64encode(b"image-one").decode(
            "utf-8"
        )

    def test_transform_image_edit_request_without_image_raises(self) -> None:
        """Test that missing image raises ValueError"""
        optional_params = {}

        with pytest.raises(ValueError, match="requires at least one image"):
            self.config.transform_image_edit_request(
                model=self.model,
                prompt=self.prompt,
                image=[],
                image_edit_optional_request_params=optional_params,
                litellm_params=MagicMock(),
                headers={},
            )

    def test_validate_environment_with_litellm_params(self) -> None:
        """Test validate_environment uses credentials from litellm_params"""
        with patch.object(
            self.config,
            "_ensure_access_token",
            return_value=("test-token", "test-expiry"),
        ) as mock_token:
            with patch.object(
                self.config,
                "set_headers",
                return_value={"Authorization": "Bearer test-token"},
            ) as mock_headers:
                litellm_params = {
                    "vertex_ai_project": "custom-project",
                    "vertex_ai_credentials": "/path/to/custom/credentials.json",
                }

                result = self.config.validate_environment(
                    headers={"X-Custom": "header"},
                    model=self.model,
                    litellm_params=litellm_params,
                    api_base=None,
                )

                # Verify that safe_get_vertex_ai_project and safe_get_vertex_ai_credentials were used
                mock_token.assert_called_once()
                call_kwargs = mock_token.call_args[1]
                assert call_kwargs["credentials"] == "/path/to/custom/credentials.json"
                assert call_kwargs["project_id"] == "custom-project"
                assert result == {"Authorization": "Bearer test-token"}

    def test_get_complete_url_from_litellm_params(self) -> None:
        """Test vertex_project/vertex_location read from litellm_params first"""
        url = self.config.get_complete_url(
            model="gemini-2.5-flash",
            api_base=None,
            litellm_params={
                "vertex_project": "params-project",
                "vertex_location": "us-east1",
            },
        )
        assert "params-project" in url
        assert "us-east1" in url

    def test_get_complete_url_global_location(self) -> None:
        """Test global location uses correct base URL without region prefix"""
        url = self.config.get_complete_url(
            model="gemini-2.5-flash",
            api_base=None,
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "global",
            },
        )
        assert "aiplatform.googleapis.com" in url
        assert "global-aiplatform.googleapis.com" not in url
        assert "/locations/global/" in url

    def test_get_complete_url_litellm_params_overrides_env(self) -> None:
        """Test litellm_params takes precedence over environment variables"""
        with patch.dict(
            os.environ,
            {
                "VERTEXAI_PROJECT": "env-project",
                "VERTEXAI_LOCATION": "us-central1",
            },
        ):
            url = self.config.get_complete_url(
                model="gemini-2.5-flash",
                api_base=None,
                litellm_params={
                    "vertex_project": "params-project",
                    "vertex_location": "eu-west1",
                },
            )
            assert "params-project" in url
            assert "eu-west1" in url
            assert "env-project" not in url
            assert "us-central1" not in url


class TestVertexAIImagenImageEditTransformation:
    def setup_method(self) -> None:
        self.config = VertexAIImagenImageEditConfig()
        self.model = "vertex_ai/imagen-3.0-capability-001"
        self.prompt = "Turn this into watercolor style scenery"
        self.logging_obj = MagicMock()

    def test_map_openai_params(self) -> None:
        """Test mapping OpenAI parameters to Vertex AI Imagen format"""
        optional_params: Dict[str, object] = {
            "n": 2,
            "size": "1024x1024",
            "mask": BytesIO(b"mask_data"),
        }

        mapped = self.config.map_openai_params(
            image_edit_optional_params=optional_params,  # type: ignore[arg-type]
            model=self.model,
            drop_params=False,
        )

        assert mapped["sampleCount"] == 2
        assert mapped["aspectRatio"] == "1:1"
        assert "mask" in mapped

    def test_get_complete_url(self) -> None:
        """Test URL generation for Vertex AI Imagen"""
        with patch.dict(
            os.environ,
            {
                "VERTEXAI_PROJECT": "test-project",
                "VERTEXAI_LOCATION": "us-central1",
            },
        ):
            url = self.config.get_complete_url(
                model="imagen-3.0-capability-001",
                api_base=None,
                litellm_params={},
            )
            assert "test-project" in url
            assert "us-central1" in url
            assert "predict" in url

    def test_transform_image_edit_request(self) -> None:
        """Test request transformation for Vertex AI Imagen"""
        image_bytes = b"fake_image_data"
        image = BytesIO(image_bytes)
        optional_params = {
            "sampleCount": 1,
        }

        request_body_str, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=image,
            image_edit_optional_request_params=optional_params,
            litellm_params=MagicMock(),
            headers={},
        )

        assert files == []
        assert isinstance(request_body_str, str)

        request_body = json.loads(request_body_str)
        assert "instances" in request_body
        assert "parameters" in request_body

        instance = request_body["instances"][0]
        assert instance["prompt"] == self.prompt
        assert "referenceImages" in instance

        reference_image = instance["referenceImages"][0]
        assert reference_image["referenceType"] == "REFERENCE_TYPE_RAW"
        assert reference_image["referenceId"] == 1
        assert "referenceImage" in reference_image
        assert "bytesBase64Encoded" in reference_image["referenceImage"]

        parameters = request_body["parameters"]
        assert parameters["sampleCount"] == 1
        assert parameters["editMode"] == "EDIT_MODE_INPAINT_INSERTION"
        assert "editConfig" in parameters

    def test_transform_image_edit_request_with_mask(self) -> None:
        """Test request transformation with mask for inpainting"""
        image_bytes = b"fake_image_data"
        mask_bytes = b"mask_data"
        image = BytesIO(image_bytes)
        mask = BytesIO(mask_bytes)
        optional_params = {
            "sampleCount": 2,
            "mask": mask,
        }

        request_body_str, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=image,
            image_edit_optional_request_params=optional_params,
            litellm_params=MagicMock(),
            headers={},
        )

        request_body = json.loads(request_body_str)
        reference_images = request_body["instances"][0]["referenceImages"]

        # Should have both base image and mask
        assert len(reference_images) == 2

        # First should be RAW reference
        assert reference_images[0]["referenceType"] == "REFERENCE_TYPE_RAW"
        assert reference_images[0]["referenceId"] == 1

        # Second should be MASK reference
        assert reference_images[1]["referenceType"] == "REFERENCE_TYPE_MASK"
        assert "maskImageConfig" in reference_images[1]
        assert (
            reference_images[1]["maskImageConfig"]["maskMode"]
            == "MASK_MODE_USER_PROVIDED"
        )

    def test_transform_image_edit_response(self) -> None:
        """Test response transformation for Vertex AI Imagen"""
        response_payload = {
            "predictions": [
                {
                    "bytesBase64Encoded": base64.b64encode(b"image-one").decode(
                        "utf-8"
                    ),
                    "mimeType": "image/png",
                },
                {
                    "bytesBase64Encoded": base64.b64encode(b"image-two").decode(
                        "utf-8"
                    ),
                    "mimeType": "image/png",
                },
            ]
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_payload
        mock_response.status_code = 200
        mock_response.headers = {}

        image_response = self.config.transform_image_edit_response(
            model=self.model,
            raw_response=mock_response,
            logging_obj=self.logging_obj,
        )

        assert image_response.data is not None
        assert len(image_response.data) == 2
        assert image_response.data[0].b64_json == base64.b64encode(b"image-one").decode(
            "utf-8"
        )
        assert image_response.data[1].b64_json == base64.b64encode(b"image-two").decode(
            "utf-8"
        )

    def test_transform_image_edit_request_without_image_raises(self) -> None:
        """Test that missing image raises ValueError"""
        optional_params = {}

        with pytest.raises(ValueError, match="requires at least one reference image"):
            self.config.transform_image_edit_request(
                model=self.model,
                prompt=self.prompt,
                image=[],
                image_edit_optional_request_params=optional_params,
                litellm_params=MagicMock(),
                headers={},
            )

    def test_read_all_bytes_handles_various_types(self) -> None:
        """Test that _read_all_bytes handles different file types"""
        # Test with bytes
        assert self.config._read_all_bytes(b"test_bytes") == b"test_bytes"

        # Test with BytesIO
        bio = BytesIO(b"test_bytesio")
        assert self.config._read_all_bytes(bio) == b"test_bytesio"

        # Test with bytearray
        assert (
            self.config._read_all_bytes(bytearray(b"test_bytearray"))
            == b"test_bytearray"
        )


_IMAGEN_EDIT_IMAGE_KEY = "bytesBase64Encoded"


def _imagen_edit_predict_response(payload: object) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode(), headers={"content-type": "application/json"})


def _imagen_edit_response_images(payload: object) -> list[tuple[object, object]]:
    response = VertexAIImagenImageEditConfig().transform_image_edit_response(
        model="imagen-3.0-capability-001",
        raw_response=_imagen_edit_predict_response(payload),
        logging_obj=Logging(
            model="imagen-3.0-capability-001",
            messages=[],
            stream=False,
            call_type="image_edit",
            start_time=datetime(2026, 1, 1),
            litellm_call_id="imagen-edit-call",
            function_id="imagen-edit-function",
        ),
    )
    return [(image.b64_json, image.url) for image in response.data]


def _imagen_edit_rejection_text(payload: object) -> str:
    with pytest.raises(ValidationError) as rejection:
        _imagen_edit_response_images(payload)
    return str(rejection.value)


def _edited_imagen_images(payload: object) -> list[tuple[object, object]]:
    transport = httpx.MockTransport(lambda request: _imagen_edit_predict_response(payload))
    response = litellm.image_edit(
        model="vertex_ai/imagen-3.0-capability-001",
        image=BytesIO(b"image-bytes"),
        prompt="add a hat",
        api_base="https://vertex.invalid",
        vertex_project="test-project",
        vertex_location="us-central1",
        client=HTTPHandler(client=httpx.Client(transport=transport)),
    )
    return [(image.b64_json, image.url) for image in response.data]


def test_imagen_image_edit_response_lists_prediction_images_in_order_and_skips_entries_without_image_bytes():
    images = _imagen_edit_response_images(
        {
            "predictions": [
                {_IMAGEN_EDIT_IMAGE_KEY: "first", "mimeType": "image/png"},
                "plain text",
                [],
                {},
                ["other"],
                {"raiFilteredReason": "blocked"},
                {_IMAGEN_EDIT_IMAGE_KEY: "second"},
            ],
            "deployedModelId": "1",
        }
    )

    assert images == [("first", None), ("second", None)]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"predictions": []},
        {"predictions": ""},
        {"predictions": "plain text"},
        {"predictions": _IMAGEN_EDIT_IMAGE_KEY},
        {"predictions": {}},
        {"predictions": {"mimeType": "image/png"}},
        {"predictions": [{}]},
        {"predictions": [{"mimeType": "image/png"}]},
        {"predictions": ["plain", ""]},
        {"predictions": [["a", "b"], []]},
        {"predictions": [[[_IMAGEN_EDIT_IMAGE_KEY]]]},
        {"Predictions": [{_IMAGEN_EDIT_IMAGE_KEY: "a"}]},
        {"predictions": [{"bytes_base64_encoded": "a"}]},
    ],
)
def test_imagen_image_edit_response_without_prediction_image_bytes_has_no_images(payload: object):
    assert _imagen_edit_response_images(payload) == []


@pytest.mark.parametrize("image_bytes", [None, "", "aGVsbG8=", "1.5", "true", "é中"])
def test_imagen_image_edit_response_keeps_the_image_bytes_value_exactly_as_sent(image_bytes: str | None):
    images = _imagen_edit_response_images({"predictions": [{_IMAGEN_EDIT_IMAGE_KEY: image_bytes}]})

    assert repr(images) == repr([(image_bytes, None)])


def test_imagen_image_edit_response_finds_the_image_after_hundreds_of_imageless_predictions():
    images = _imagen_edit_response_images({"predictions": [*[{}] * 403, {_IMAGEN_EDIT_IMAGE_KEY: "late"}]})

    assert images == [("late", None)]


@pytest.mark.parametrize("image_bytes", [5, 1.5, True, ["a"], {"a": 1}])
def test_imagen_image_edit_response_rejects_non_string_image_bytes_naming_b64_json(image_bytes: object):
    with pytest.raises(ValidationError, match="b64_json"):
        _imagen_edit_response_images({"predictions": [{_IMAGEN_EDIT_IMAGE_KEY: image_bytes}]})


@pytest.mark.parametrize(
    "payload",
    [
        {"predictions": None},
        {"predictions": 7},
        {"predictions": 1.5},
        {"predictions": True},
        {"predictions": {_IMAGEN_EDIT_IMAGE_KEY: "vertex-secret"}},
        {"predictions": [None]},
        {"predictions": [7]},
        {"predictions": [1.5]},
        {"predictions": [False]},
        {"predictions": ["vertex-secret " + _IMAGEN_EDIT_IMAGE_KEY]},
        {"predictions": [[_IMAGEN_EDIT_IMAGE_KEY, "vertex-secret"]]},
        {"predictions": [{_IMAGEN_EDIT_IMAGE_KEY: "vertex-secret"}, None]},
    ],
)
def test_imagen_image_edit_response_rejects_malformed_predictions_without_echoing_them(payload: object):
    rejection_text = _imagen_edit_rejection_text(payload)

    assert "vertex-secret" not in rejection_text and "input_value" not in rejection_text, rejection_text


@pytest.mark.parametrize("position", [403, 429])
@pytest.mark.parametrize("malformed", [None, 7, _IMAGEN_EDIT_IMAGE_KEY, [_IMAGEN_EDIT_IMAGE_KEY]])
def test_imagen_image_edit_response_rejection_text_is_the_same_wherever_the_malformed_prediction_sits(
    position: int, malformed: object
):
    rejection_at_start = _imagen_edit_rejection_text({"predictions": [malformed]})

    rejection_at_position = _imagen_edit_rejection_text({"predictions": [*[{}] * position, malformed]})

    assert rejection_at_position == rejection_at_start and str(position) not in rejection_at_position


@pytest.mark.parametrize("body", [[], [{_IMAGEN_EDIT_IMAGE_KEY: "a"}], "predictions", 7, 2.5, True, None])
def test_imagen_image_edit_response_rejects_a_predict_body_that_is_not_an_object_with_attribute_error(body: object):
    with pytest.raises(AttributeError, match="object has no attribute 'get'"):
        _imagen_edit_response_images(body)


def test_imagen_image_edit_returns_the_prediction_images_of_the_predict_endpoint():
    images = _edited_imagen_images(
        {
            "predictions": [
                {_IMAGEN_EDIT_IMAGE_KEY: "first"},
                {"mimeType": "image/png"},
                {_IMAGEN_EDIT_IMAGE_KEY: "second"},
            ]
        }
    )

    assert images == [("first", None), ("second", None)]


@pytest.mark.parametrize(
    "payload",
    [
        {"predictions": None},
        {"predictions": 7},
        {"predictions": [None]},
        {"predictions": ["vertex-secret " + _IMAGEN_EDIT_IMAGE_KEY]},
        {"predictions": [*[{}] * 403, [_IMAGEN_EDIT_IMAGE_KEY, "vertex-secret"]]},
    ],
)
def test_imagen_image_edit_maps_malformed_predictions_to_a_connection_error_without_echoing_them(payload: object):
    with pytest.raises(litellm.APIConnectionError) as mapped:
        _edited_imagen_images(payload)

    assert (type(mapped.value), mapped.value.status_code, "vertex-secret" in str(mapped.value)) == (
        litellm.APIConnectionError,
        500,
        False,
    )
