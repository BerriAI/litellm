import base64
import json
import os
from io import BytesIO
from typing import Dict
from unittest.mock import MagicMock, patch

import httpx
import pytest

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
                                    "data": base64.b64encode(b"image-one").decode("utf-8"),
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
            self.config, "_ensure_access_token", return_value=("test-token", "test-expiry")
        ) as mock_token:
            with patch.object(self.config, "set_headers", return_value={"Authorization": "Bearer test-token"}) as mock_headers:
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
        assert reference_images[1]["maskImageConfig"]["maskMode"] == "MASK_MODE_USER_PROVIDED"

    def test_transform_image_edit_response(self) -> None:
        """Test response transformation for Vertex AI Imagen"""
        response_payload = {
            "predictions": [
                {
                    "bytesBase64Encoded": base64.b64encode(b"image-one").decode("utf-8"),
                    "mimeType": "image/png",
                },
                {
                    "bytesBase64Encoded": base64.b64encode(b"image-two").decode("utf-8"),
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
        assert self.config._read_all_bytes(bytearray(b"test_bytearray")) == b"test_bytearray"

