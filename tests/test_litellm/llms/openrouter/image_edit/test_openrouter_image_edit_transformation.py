import base64
import json
import os
import sys
from io import BytesIO
from typing import Optional

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.image_edit.transformation import OpenRouterImageEditConfig
from litellm.types.router import GenericLiteLLMParams

# Verbatim shape of a `POST https://openrouter.ai/api/v1/images` response.
OPENROUTER_IMAGES_RESPONSE = {
    "created": 0,
    "data": [{"b64_json": "iVBORw0KGgoAAAANS", "media_type": "image/png"}],
    "usage": {
        "prompt_tokens": 300,
        "completion_tokens": 1299,
        "total_tokens": 1599,
        "cost": 0.05,
        "is_byok": False,
        "prompt_tokens_details": {"image_tokens": 258, "cached_tokens": 0},
        "cost_details": {"upstream_inference_cost": 0.05},
        "completion_tokens_details": {"reasoning_tokens": 0, "image_tokens": 1290},
    },
}

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


class RequestRecorder:
    """httpx.MockTransport handler that keeps the request it was called with."""

    def __init__(self, response_payload: dict, status_code: int = 200):
        self.response_payload = response_payload
        self.status_code = status_code
        self.request: Optional[httpx.Request] = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return httpx.Response(status_code=self.status_code, json=self.response_payload)

    def body(self) -> dict:
        assert self.request is not None, "no request was sent"
        return json.loads(self.request.content)


def make_client(recorder: RequestRecorder) -> HTTPHandler:
    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(recorder)))


def make_httpx_response(payload: object, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request(method="POST", url="https://openrouter.ai/api/v1/images"),
    )


class TestOpenRouterImageEditEndpoint:
    """OpenRouter edits images from /api/v1/images, never /chat/completions."""

    def setup_method(self):
        self.config = OpenRouterImageEditConfig()
        self.model = "openai/gpt-image-2"

    def test_get_complete_url_appends_images_path(self):
        result = self.config.get_complete_url(
            model=self.model,
            api_base="https://openrouter.ai/api/v1",
            litellm_params={},
        )

        assert result == "https://openrouter.ai/api/v1/images"

    def test_get_complete_url_defaults_to_openrouter(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)

        result = self.config.get_complete_url(model=self.model, api_base=None, litellm_params={})

        assert result == "https://openrouter.ai/api/v1/images"

    def test_get_complete_url_reads_api_base_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_BASE", "https://gateway.internal/openrouter/v1/")

        result = self.config.get_complete_url(model=self.model, api_base=None, litellm_params={})

        assert result == "https://gateway.internal/openrouter/v1/images"

    def test_get_complete_url_does_not_duplicate_images_path(self):
        result = self.config.get_complete_url(
            model=self.model,
            api_base="https://openrouter.ai/api/v1/images",
            litellm_params={},
        )

        assert result == "https://openrouter.ai/api/v1/images"

    def test_uses_json_not_multipart(self):
        assert self.config.use_multipart_form_data() is False


class TestOpenRouterImageEditParams:
    def setup_method(self):
        self.config = OpenRouterImageEditConfig()
        self.model = "openai/gpt-image-2"

    def test_get_supported_openai_params(self):
        assert set(self.config.get_supported_openai_params(self.model)) == {
            "background",
            "n",
            "quality",
            "response_format",
            "size",
        }

    @pytest.mark.parametrize(
        "size, expected_aspect_ratio",
        [
            ("256x256", "1:1"),
            ("1024x1024", "1:1"),
            ("1536x1024", "3:2"),
            ("1792x1024", "16:9"),
            ("1024x1536", "2:3"),
            ("1024x1792", "9:16"),
            ("1184x864", "4:3"),
            ("1536x672", "21:9"),
        ],
    )
    def test_size_maps_to_nearest_supported_aspect_ratio(self, size, expected_aspect_ratio):
        result = self.config.map_openai_params(
            image_edit_optional_params={"size": size},
            model=self.model,
            drop_params=False,
        )

        assert result == {"aspect_ratio": expected_aspect_ratio}

    def test_aspect_ratio_is_not_nested_under_image_config(self):
        """`image_config` is a chat completions concept; /images takes the params flat."""
        result = self.config.map_openai_params(
            image_edit_optional_params={"size": "1024x1024"},
            model=self.model,
            drop_params=False,
        )

        assert "image_config" not in result

    @pytest.mark.parametrize("size", ["auto", "", "large", "1024", "0x0"])
    def test_size_without_a_pixel_value_is_omitted(self, size):
        """No aspect_ratio is better than guessing 1:1 and silently squaring the output."""
        result = self.config.map_openai_params(
            image_edit_optional_params={"size": size},
            model=self.model,
            drop_params=False,
        )

        assert result == {}

    @pytest.mark.parametrize("quality", ["auto", "low", "medium", "high"])
    def test_quality_passes_through_untranslated(self, quality):
        """OpenRouter reuses OpenAI's quality enum; translating it to an `image_size` tier
        silently drops the caller's intent on models that take `quality`."""
        result = self.config.map_openai_params(
            image_edit_optional_params={"quality": quality},
            model=self.model,
            drop_params=False,
        )

        assert result == {"quality": quality}

    def test_image_size_tier_is_not_synthesized_from_quality(self):
        result = self.config.map_openai_params(
            image_edit_optional_params={"quality": "high"},
            model=self.model,
            drop_params=False,
        )

        assert "image_size" not in result
        assert "resolution" not in result

    def test_n_and_background_pass_through(self):
        result = self.config.map_openai_params(
            image_edit_optional_params={"n": 2, "background": "transparent"},
            model=self.model,
            drop_params=False,
        )

        assert result == {"n": 2, "background": "transparent"}

    def test_b64_json_response_format_is_accepted_and_not_sent(self):
        result = self.config.map_openai_params(
            image_edit_optional_params={"response_format": "b64_json"},
            model=self.model,
            drop_params=False,
        )

        assert result == {}

    def test_url_response_format_raises(self):
        with pytest.raises(litellm.UnsupportedParamsError) as exc_info:
            self.config.map_openai_params(
                image_edit_optional_params={"response_format": "url"},
                model=self.model,
                drop_params=False,
            )

        assert "response_format='url' is not supported" in str(exc_info.value)

    def test_url_response_format_is_dropped_when_drop_params_is_set(self):
        result = self.config.map_openai_params(
            image_edit_optional_params={"response_format": "url", "size": "1024x1024"},
            model=self.model,
            drop_params=True,
        )

        assert result == {"aspect_ratio": "1:1"}

    def test_mask_is_rejected_because_openrouter_has_no_mask(self):
        with pytest.raises(litellm.UnsupportedParamsError) as exc_info:
            ImageEditRequestUtils.get_optional_params_image_edit(
                model=self.model,
                image_edit_provider_config=self.config,
                image_edit_optional_params={"mask": "data:image/png;base64,abc"},
            )

        assert "mask" in str(exc_info.value)


class TestOpenRouterImageEditRequest:
    def setup_method(self):
        self.config = OpenRouterImageEditConfig()
        self.model = "openai/gpt-image-2"

    def _transform(self, image=PNG_BYTES, prompt="Add a sunset", optional_params=None):
        return self.config.transform_image_edit_request(
            model=self.model,
            prompt=prompt,
            image=image,
            image_edit_optional_request_params=optional_params or {},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    def test_source_image_travels_in_input_references_not_in_messages(self):
        data, files = self._transform()

        assert data["model"] == self.model
        assert data["prompt"] == "Add a sunset"
        assert "messages" not in data
        assert "modalities" not in data
        assert list(files) == []

        assert data["input_references"] == [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"},
            }
        ]

    def test_multiple_source_images_become_multiple_references(self):
        data, _ = self._transform(image=[PNG_BYTES, PNG_BYTES])

        assert len(data["input_references"]) == 2
        assert all(ref["type"] == "image_url" for ref in data["input_references"])

    def test_bytesio_image_is_read_without_moving_the_caller_position(self):
        image = BytesIO(PNG_BYTES)
        image.seek(4)

        data, _ = self._transform(image=image)

        assert data["input_references"][0]["image_url"]["url"].endswith(base64.b64encode(PNG_BYTES).decode())
        assert image.tell() == 4

    def test_jpeg_content_type_is_detected(self):
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100

        data, _ = self._transform(image=jpeg_bytes)

        assert data["input_references"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_mapped_params_are_sent_flat(self):
        data, _ = self._transform(optional_params={"aspect_ratio": "16:9", "n": 2, "quality": "high"})

        assert data["aspect_ratio"] == "16:9"
        assert data["n"] == 2
        assert data["quality"] == "high"

    def test_extra_headers_are_not_leaked_into_the_request_body(self):
        data, _ = self._transform(optional_params={"extra_headers": {"X-Title": "litellm"}})

        assert "extra_headers" not in data

    def test_missing_image_raises_instead_of_generating_a_new_one(self):
        """Without input_references OpenRouter would happily generate an unrelated image."""
        with pytest.raises(ValueError, match="image is required"):
            self._transform(image=[])

    def test_unsupported_image_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported image type"):
            self._transform(image="not_an_image")

    def test_prompt_is_omitted_when_absent(self):
        data, _ = self._transform(prompt=None)

        assert "prompt" not in data


class TestOpenRouterImageEditResponse:
    def setup_method(self):
        self.config = OpenRouterImageEditConfig()
        self.model = "openai/gpt-image-2"

    def _transform(self, payload: object, status_code: int = 200):
        return self.config.transform_image_edit_response(
            model=self.model,
            raw_response=make_httpx_response(payload, status_code=status_code),
            logging_obj=None,
        )

    def test_extracts_base64_image_from_data_array(self):
        result = self._transform(OPENROUTER_IMAGES_RESPONSE)

        assert len(result.data) == 1
        assert result.data[0].b64_json == "iVBORw0KGgoAAAANS"
        assert result.data[0].url is None

    def test_extracts_multiple_images(self):
        payload = {"created": 0, "data": [{"b64_json": "image1data"}, {"b64_json": "image2data"}]}

        result = self._transform(payload)

        assert [image.b64_json for image in result.data] == ["image1data", "image2data"]

    def test_extracts_url_image(self):
        payload = {"created": 0, "data": [{"url": "https://example.com/edited.png"}]}

        result = self._transform(payload)

        assert result.data[0].url == "https://example.com/edited.png"
        assert result.data[0].b64_json is None

    def test_maps_usage_and_cost(self):
        result = self._transform(OPENROUTER_IMAGES_RESPONSE)

        assert result.usage is not None
        assert result.usage.input_tokens == 300
        assert result.usage.output_tokens == 1290
        assert result.usage.total_tokens == 1599
        assert result.usage.input_tokens_details.image_tokens == 258
        assert result.usage.input_tokens_details.text_tokens == 42
        assert result._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.05
        assert result._hidden_params["response_cost_details"]["upstream_inference_cost"] == 0.05

    def test_falls_back_to_completion_tokens_when_image_tokens_absent(self):
        payload = {
            "data": [{"b64_json": "abc"}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 1299, "total_tokens": 1305},
        }

        result = self._transform(payload)

        assert result.usage.output_tokens == 1299
        assert result.usage.input_tokens_details.image_tokens == 0

    def test_zero_created_does_not_overwrite_litellm_timestamp(self):
        result = self._transform(OPENROUTER_IMAGES_RESPONSE)

        assert result.created > 0

    def test_reports_response_model(self):
        payload = {"data": [{"b64_json": "abc"}], "model": "openai/gpt-image-2"}

        result = self._transform(payload)

        assert result._hidden_params["model"] == "openai/gpt-image-2"

    def test_response_without_data_raises(self):
        with pytest.raises(OpenRouterException) as exc_info:
            self._transform({"created": 0})

        assert "Error parsing OpenRouter image response" in str(exc_info.value)

    def test_unparseable_response_raises(self):
        raw_response = httpx.Response(
            status_code=502,
            text="<html>bad gateway</html>",
            request=httpx.Request(method="POST", url="https://openrouter.ai/api/v1/images"),
        )

        with pytest.raises(OpenRouterException) as exc_info:
            self.config.transform_image_edit_response(
                model=self.model,
                raw_response=raw_response,
                logging_obj=None,
            )

        assert exc_info.value.status_code == 502

    def test_get_error_class(self):
        error = self.config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, OpenRouterException)
        assert error.status_code == 400


class TestOpenRouterImageEditEnvironment:
    def setup_method(self):
        self.config = OpenRouterImageEditConfig()
        self.model = "openai/gpt-image-2"

    def test_validate_environment_prefers_explicit_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env_key")

        headers = self.config.validate_environment(headers={}, model=self.model, api_key="test_api_key")

        assert headers["Authorization"] == "Bearer test_api_key"

    def test_validate_environment_falls_back_to_env_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "secret_api_key")

        headers = self.config.validate_environment(headers={}, model=self.model, api_key=None)

        assert headers["Authorization"] == "Bearer secret_api_key"

    def test_validate_environment_without_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", None)

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
            self.config.validate_environment(headers={}, model=self.model, api_key=None)


class TestOpenRouterImageEditEndToEnd:
    """Covers param mapping + URL + body together, the way the proxy drives it."""

    def test_openai_style_edit_reaches_the_openrouter_images_endpoint(self):
        recorder = RequestRecorder(OPENROUTER_IMAGES_RESPONSE)

        response = litellm.image_edit(
            model="openrouter/openai/gpt-image-2",
            image=PNG_BYTES,
            prompt="Add a sunset behind the mountain",
            n=1,
            size="1024x1024",
            response_format="b64_json",
            api_key="sk-test",
            api_base="https://openrouter.ai/api/v1",
            client=make_client(recorder),
        )

        assert recorder.request is not None
        assert str(recorder.request.url) == "https://openrouter.ai/api/v1/images"
        assert recorder.request.headers["Authorization"] == "Bearer sk-test"
        assert recorder.body() == {
            "model": "openai/gpt-image-2",
            "prompt": "Add a sunset behind the mountain",
            "input_references": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"},
                }
            ],
            "n": 1,
            "aspect_ratio": "1:1",
        }
        assert response.data[0].b64_json == "iVBORw0KGgoAAAANS"

    def test_image_only_model_no_longer_asks_for_text_modality(self):
        """`modalities: ["image", "text"]` is what made OpenRouter answer 404 for these models."""
        recorder = RequestRecorder(OPENROUTER_IMAGES_RESPONSE)

        litellm.image_edit(
            model="openrouter/openai/gpt-image-2",
            image=PNG_BYTES,
            prompt="Add a sunset",
            api_key="sk-test",
            api_base="https://openrouter.ai/api/v1",
            client=make_client(recorder),
        )

        assert "chat/completions" not in str(recorder.request.url)
        assert "modalities" not in recorder.body()

    def test_hybrid_image_model_uses_the_same_endpoint(self):
        recorder = RequestRecorder(OPENROUTER_IMAGES_RESPONSE)

        litellm.image_edit(
            model="openrouter/google/gemini-2.5-flash-image",
            image=PNG_BYTES,
            prompt="Add a sunset",
            api_key="sk-test",
            api_base="https://openrouter.ai/api/v1",
            client=make_client(recorder),
        )

        assert str(recorder.request.url) == "https://openrouter.ai/api/v1/images"
        assert recorder.body()["model"] == "google/gemini-2.5-flash-image"

    def test_openrouter_error_is_surfaced(self):
        recorder = RequestRecorder(
            {"error": {"message": "No endpoints found that support image edit", "code": 404}},
            status_code=404,
        )

        with pytest.raises(litellm.NotFoundError):
            litellm.image_edit(
                model="openrouter/openai/gpt-image-2",
                image=PNG_BYTES,
                prompt="Add a sunset",
                api_key="sk-test",
                api_base="https://openrouter.ai/api/v1",
                client=make_client(recorder),
            )
