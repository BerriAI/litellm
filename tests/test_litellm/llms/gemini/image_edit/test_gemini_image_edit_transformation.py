import base64
import json
from io import BytesIO
from typing import Dict
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.gemini.image_edit.transformation import GeminiImageEditConfig


class TestGeminiImageEditTransformation:
    def setup_method(self) -> None:
        self.config = GeminiImageEditConfig()
        self.model = "gemini-2.5-flash-image-preview"
        self.prompt = "Enhance this photo with a dramatic night sky."
        self.logging_obj = MagicMock()

    def test_map_openai_params(self) -> None:
        optional_params: Dict[str, object] = {
            "size": "1792x1024",
            "response_format": "b64_json",
            "quality": "high",
        }

        mapped = self.config.map_openai_params(
            image_edit_optional_params=optional_params,  # type: ignore[arg-type]
            model=self.model,
            drop_params=False,
        )

        assert mapped["aspectRatio"] == "16:9"
        assert "response_format" not in mapped
        assert "quality" not in mapped

    def test_transform_image_edit_request(self) -> None:
        image_bytes = b"fake_image_data"
        image = BytesIO(image_bytes)
        optional_params = {
            "sampleCount": 2,
            "aspectRatio": "16:9",
        }

        request_body, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=[image],  # Gemini pipeline passes list of images
            image_edit_optional_request_params=optional_params,
            litellm_params=MagicMock(),
            headers={},
        )

        assert files == []

        parts = request_body["contents"][0]["parts"]
        assert parts[-1]["text"] == self.prompt

        inline_data = parts[0]["inlineData"]
        assert inline_data["mimeType"] == "image/png"
        assert base64.b64decode(inline_data["data"]) == image_bytes

        generation_config = request_body["generationConfig"]
        assert generation_config["aspectRatio"] == "16:9"

    def test_transform_image_edit_request_multiple_images(self) -> None:
        image_one = BytesIO(b"image_one")
        image_two = BytesIO(b"image_two")

        request_body, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=[image_one, image_two],
            image_edit_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )

        assert files == []
        parts = request_body["contents"][0]["parts"]

        assert len(parts) == 3  # two images + text prompt
        assert parts[-1]["text"] == self.prompt
        assert base64.b64decode(parts[0]["inlineData"]["data"]) == b"image_one"
        assert base64.b64decode(parts[1]["inlineData"]["data"]) == b"image_two"

    def test_transform_image_edit_response(self) -> None:
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
                },
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(b"image-two").decode("utf-8"),
                                }
                            }
                        ]
                    }
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
        optional_params = {}

        with pytest.raises(ValueError):
            self.config.transform_image_edit_request(
                model=self.model,
                prompt=self.prompt,
                image=[],
                image_edit_optional_request_params=optional_params,
                litellm_params=MagicMock(),
                headers={},
            )

