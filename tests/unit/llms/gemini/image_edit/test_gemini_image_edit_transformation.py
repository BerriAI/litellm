import base64
import json
from collections.abc import Mapping
from io import BytesIO
from typing import Dict, Final
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.gemini.image_edit.transformation import GeminiImageEditConfig


class TestGeminiImageEditTransformation:
    def setup_method(self) -> None:
        self.config = GeminiImageEditConfig()
        self.model = "gemini-2.5-flash-image-preview"
        self.prompt = "Enhance this photo with a dramatic night sky."
        self.logging_obj = MagicMock()

    def test_map_openai_params(self) -> None:
        optional_params: Dict[str, object] = {
            "n": 2,
            "size": "1792x1024",
            "response_format": "b64_json",
            "quality": "high",
        }

        mapped = self.config.map_openai_params(
            image_edit_optional_params=optional_params,  # type: ignore[arg-type]
            model=self.model,
            drop_params=False,
        )

        assert mapped["imageConfig"] == {"aspectRatio": "16:9"}
        assert mapped["sampleCount"] == 2
        assert "response_format" not in mapped
        assert "quality" not in mapped

    def test_map_openai_params_with_image_size_for_gemini_3(self) -> None:
        optional_params: Dict[str, object] = {
            "size": "768x1376",
        }

        mapped = self.config.map_openai_params(
            image_edit_optional_params=optional_params,  # type: ignore[arg-type]
            model="gemini-3-pro-image-preview",
            drop_params=False,
        )

        assert mapped["imageConfig"] == {"aspectRatio": "9:16", "imageSize": "1K"}

    def test_map_openai_params_forwards_image_config_as_is(self) -> None:
        optional_params: Dict[str, object] = {
            "size": "1024x1024",
            "imageConfig": {"aspectRatio": "16:9", "imageSize": "512px"},
        }

        mapped = self.config.map_openai_params(
            image_edit_optional_params=optional_params,  # type: ignore[arg-type]
            model="gemini-3-pro-image-preview",
            drop_params=False,
        )

        assert mapped["imageConfig"] == {"aspectRatio": "16:9", "imageSize": "512px"}

    def test_map_openai_params_parses_form_image_config_json(self) -> None:
        optional_params: Dict[str, object] = {
            "imageConfig": '{"aspectRatio":"16:9","imageSize":"1K"}',
        }

        mapped = self.config.map_openai_params(
            image_edit_optional_params=optional_params,  # type: ignore[arg-type]
            model="gemini-3-pro-image-preview",
            drop_params=False,
        )

        assert mapped["imageConfig"] == {"aspectRatio": "16:9", "imageSize": "1K"}

    def test_map_openai_params_rejects_malformed_form_image_config_json(
        self,
    ) -> None:
        optional_params: Dict[str, object] = {
            "imageConfig": "{bad",
        }

        with pytest.raises(litellm.UnsupportedParamsError) as exc_info:
            self.config.map_openai_params(
                image_edit_optional_params=optional_params,  # type: ignore[arg-type]
                model="gemini-3-pro-image-preview",
                drop_params=False,
            )

        assert "`imageConfig` must be valid JSON" in str(exc_info.value)

    def test_transform_image_edit_request(self) -> None:
        image_bytes = b"fake_image_data"
        image = BytesIO(image_bytes)
        optional_params = {
            "sampleCount": 2,
            "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"},
        }

        request_body, files = self.config.transform_image_edit_request(
            model="gemini-3-pro-image-preview",
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
        assert "candidateCount" not in generation_config
        assert generation_config["imageConfig"]["aspectRatio"] == "16:9"
        assert generation_config["imageConfig"]["imageSize"] == "2K"

    def test_transform_image_edit_request_omits_image_size_for_gemini_25(self) -> None:
        image = BytesIO(b"fake_image_data")
        optional_params = {
            "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"},
        }

        request_body, _ = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=[image],
            image_edit_optional_request_params=optional_params,
            litellm_params=MagicMock(),
            headers={},
        )

        assert request_body["generationConfig"]["imageConfig"] == {
            "aspectRatio": "16:9"
        }

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
                                    "data": base64.b64encode(b"image-one").decode(
                                        "utf-8"
                                    ),
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
                                    "data": base64.b64encode(b"image-two").decode(
                                        "utf-8"
                                    ),
                                }
                            }
                        ]
                    }
                },
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

        usage = image_response.model_dump()["usage"]
        assert usage["input_tokens"] == 35
        assert usage["output_tokens"] == 1716
        assert usage["prompt_tokens"] == 35
        assert usage["completion_tokens"] == 1716
        assert usage["prompt_tokens_details"]["image_tokens"] == 5
        assert usage["completion_tokens_details"] == {}

        logging_usage = StandardLoggingPayloadSetup.get_usage_as_dict(
            response_obj=image_response.model_dump()
        )
        assert all(value is None for value in logging_usage["completion_tokens_details"].values())

    @pytest.mark.parametrize(
        ("body", "error", "message"),
        [
            (["not", "an", "object"], AttributeError, "'list' object has no attribute 'get'"),
            ({"candidates": None}, TypeError, "'NoneType' object is not iterable"),
            ({"candidates": [{"content": {"parts": None}}]}, TypeError, "'NoneType' object is not iterable"),
        ],
    )
    def test_transform_image_edit_response_malformed_body_raises_the_error_of_the_failing_lookup(
        self, body: object, error: type[Exception], message: str
    ) -> None:
        with pytest.raises(error, match=message) as raised:
            self.config.transform_image_edit_response(
                model=self.model,
                raw_response=httpx.Response(200, json=body),
                logging_obj=self.logging_obj,
            )

        assert raised.type is error

    @pytest.mark.parametrize("image", [None, []])
    def test_transform_image_edit_request_without_image_is_a_bad_request(self, image: list[object] | None) -> None:
        with pytest.raises(litellm.BadRequestError, match="Gemini image edit requires at least one image") as raised:
            self.config.transform_image_edit_request(
                model=self.model,
                prompt=self.prompt,
                image=image,
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )

        assert raised.value.status_code == 400
        assert raised.value.llm_provider == "gemini"

    def test_use_multipart_form_data_returns_false(self) -> None:
        """
        Gemini uses JSON requests, not multipart/form-data.
        This is critical because httpx sends data differently:
        - data=dict sends form-encoded
        - json=dict sends JSON

        Without this, Gemini returns: "Invalid JSON payload received. Unexpected token."
        """
        assert self.config.use_multipart_form_data() is False


_SENT_BODY: Final = TypeAdapter(Mapping[str, object])


def _sent_gemini_edit_generation_config(**params: object) -> Mapping[str, object]:
    sent: Final[list[httpx.Request]] = []  # mutable-ok: MockTransport hands each request to a callback

    def reply(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1n"}}]}}]},
        )

    litellm.image_edit(
        model="gemini/gemini-3.1-flash-image",
        image=BytesIO(b"\x89PNG\r\n\x1a\nimage"),
        prompt="mountains",
        api_key="test-key",
        api_base="https://gemini.test/v1beta",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(reply))),
        **params,
    )
    return _SENT_BODY.validate_python(_SENT_BODY.validate_json(sent[0].content)["generationConfig"])


@pytest.mark.parametrize(
    ("fields", "image_config"),
    (
        (
            {"imageConfig[aspectRatio]": "9:16", "imageConfig[image_size]": "2K"},
            {"aspectRatio": "9:16", "imageSize": "2K"},
        ),
        (
            {"imageConfig": '{"aspectRatio": "1:1", "imageSize": "2K"}', "imageConfig[aspectRatio]": "9:16"},
            {"aspectRatio": "9:16", "imageSize": "2K"},
        ),
        (
            {"imageConfig": '{"aspect_ratio": "9:16"}', "imageConfig[aspectRatio]": "1:1"},
            {"aspectRatio": "1:1"},
        ),
        (
            {
                "imageConfig": '{"imageOutputOptions": {"mimeType": "image/jpeg"}}',
                "imageConfig[image_output_options][mime_type]": "image/png",
            },
            {"imageOutputOptions": {"mimeType": "image/png"}},
        ),
    ),
)
def test_gemini_edit_folds_bracket_image_config_fields(
    fields: Mapping[str, object], image_config: Mapping[str, object]
) -> None:
    assert _sent_gemini_edit_generation_config(**fields)["imageConfig"] == image_config


def test_gemini_edit_rejects_more_than_64_bracket_fields_unless_dropping_params() -> None:
    fields: Final = {f"imageConfig[k{index}]": "v" for index in range(65)}

    with pytest.raises(litellm.UnsupportedParamsError, match="`imageConfig` takes at most 64 form fields"):
        _sent_gemini_edit_generation_config(**fields)

    assert "imageConfig" not in _sent_gemini_edit_generation_config(**fields, drop_params=True)


def test_gemini_edit_rejects_bracket_fields_beside_a_non_object_image_config_unless_dropping_params() -> None:
    fields: Final = {"imageConfig": "[1]", "imageConfig[aspectRatio]": "9:16"}

    with pytest.raises(litellm.UnsupportedParamsError, match="`imageConfig` must be a JSON object"):
        _sent_gemini_edit_generation_config(**fields)

    assert _sent_gemini_edit_generation_config(**fields, drop_params=True)["imageConfig"] == {"aspectRatio": "9:16"}
