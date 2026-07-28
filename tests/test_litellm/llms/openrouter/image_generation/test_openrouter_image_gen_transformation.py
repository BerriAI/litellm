import json
import os
import sys
from typing import Optional

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.image_generation.transformation import (
    OpenRouterImageGenerationConfig,
)
from litellm.types.utils import ImageResponse

# Verbatim shape of a `POST https://openrouter.ai/api/v1/images` response.
OPENROUTER_IMAGES_RESPONSE = {
    "created": 0,
    "data": [{"b64_json": "iVBORw0KGgoAAAANS", "media_type": "image/png"}],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 4175,
        "total_tokens": 4175,
        "cost": 0.06,
        "is_byok": False,
        "prompt_tokens_details": {"cached_tokens": 0},
        "cost_details": {
            "upstream_inference_cost": 0.06,
            "upstream_inference_prompt_cost": 0.0,
            "upstream_inference_completions_cost": 0.06,
        },
        "completion_tokens_details": {"reasoning_tokens": 0, "image_tokens": 4175},
    },
}


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


class TestOpenRouterImageGenerationEndpoint:
    """OpenRouter serves image generation from /api/v1/images, never /chat/completions."""

    def setup_method(self):
        self.config = OpenRouterImageGenerationConfig()
        self.model = "krea/krea-2-large"

    def test_get_complete_url_appends_images_path(self):
        result = self.config.get_complete_url(
            api_base="https://openrouter.ai/api/v1",
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://openrouter.ai/api/v1/images"

    def test_get_complete_url_defaults_to_openrouter(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)

        result = self.config.get_complete_url(
            api_base=None,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://openrouter.ai/api/v1/images"

    def test_get_complete_url_reads_api_base_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_BASE", "https://gateway.internal/openrouter/v1/")

        result = self.config.get_complete_url(
            api_base=None,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://gateway.internal/openrouter/v1/images"

    def test_get_complete_url_does_not_duplicate_images_path(self):
        result = self.config.get_complete_url(
            api_base="https://openrouter.ai/api/v1/images",
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://openrouter.ai/api/v1/images"


class TestOpenRouterImageGenerationParams:
    def setup_method(self):
        self.config = OpenRouterImageGenerationConfig()
        self.model = "krea/krea-2-large"

    def test_get_supported_openai_params(self):
        assert set(self.config.get_supported_openai_params(self.model)) == {
            "n",
            "quality",
            "response_format",
            "size",
        }

    @pytest.mark.parametrize(
        "size, expected_aspect_ratio",
        [
            ("256x256", "1:1"),
            ("512x512", "1:1"),
            ("1024x1024", "1:1"),
            ("1536x1024", "3:2"),
            ("1792x1024", "16:9"),
            ("1024x1536", "2:3"),
            ("1024x1792", "9:16"),
            ("1184x864", "4:3"),
            ("896x1152", "4:5"),
            ("1536x672", "21:9"),
        ],
    )
    def test_size_maps_to_nearest_supported_aspect_ratio(self, size, expected_aspect_ratio):
        result = self.config.map_openai_params(
            non_default_params={"size": size},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"aspect_ratio": expected_aspect_ratio}

    @pytest.mark.parametrize("size", ["auto", "", "large", "1024", "0x0"])
    def test_size_without_a_pixel_value_is_omitted(self, size):
        """No aspect_ratio is better than guessing 1:1 and silently squaring the output."""
        result = self.config.map_openai_params(
            non_default_params={"size": size},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {}

    @pytest.mark.parametrize("quality", ["auto", "low", "medium", "high"])
    def test_quality_passes_through_untranslated(self, quality):
        """OpenRouter reuses OpenAI's quality enum; translating it to `resolution` silently
        drops the caller's intent on models that take `quality` (openai/gpt-image-*)."""
        result = self.config.map_openai_params(
            non_default_params={"quality": quality},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"quality": quality}

    def test_resolution_is_not_synthesized_from_quality(self):
        result = self.config.map_openai_params(
            non_default_params={"quality": "high"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert "resolution" not in result

    def test_n_is_passed_through(self):
        result = self.config.map_openai_params(
            non_default_params={"n": 2, "size": "1024x1024"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"n": 2, "aspect_ratio": "1:1"}

    def test_b64_json_response_format_is_accepted_and_not_sent(self):
        """Open WebUI always sends response_format=b64_json; it must not need additional_drop_params."""
        result = self.config.map_openai_params(
            non_default_params={"response_format": "b64_json"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {}

    def test_url_response_format_raises(self):
        with pytest.raises(litellm.UnsupportedParamsError) as exc_info:
            self.config.map_openai_params(
                non_default_params={"response_format": "url"},
                optional_params={},
                model=self.model,
                drop_params=False,
            )

        assert "response_format='url' is not supported" in str(exc_info.value)

    def test_url_response_format_is_dropped_when_drop_params_is_set(self):
        result = self.config.map_openai_params(
            non_default_params={"response_format": "url", "size": "1024x1024"},
            optional_params={},
            model=self.model,
            drop_params=True,
        )

        assert result == {"aspect_ratio": "1:1"}

    def test_map_openai_params_does_not_mutate_its_input(self):
        optional_params = {}

        self.config.map_openai_params(
            non_default_params={"size": "1024x1024"},
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert optional_params == {}


class TestOpenRouterImageGenerationRequest:
    def setup_method(self):
        self.config = OpenRouterImageGenerationConfig()
        self.model = "krea/krea-2-large"

    def test_request_body_is_prompt_based_not_chat_based(self):
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt="a red panda astronaut",
            optional_params={"aspect_ratio": "16:9", "resolution": "4K", "n": 1},
            litellm_params={},
            headers={},
        )

        assert result == {
            "model": self.model,
            "prompt": "a red panda astronaut",
            "aspect_ratio": "16:9",
            "resolution": "4K",
            "n": 1,
        }
        assert "messages" not in result
        assert "modalities" not in result

    def test_extra_headers_are_not_leaked_into_the_request_body(self):
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt="a red panda astronaut",
            optional_params={"extra_headers": {"X-Title": "litellm"}, "seed": 7},
            litellm_params={},
            headers={},
        )

        assert result == {"model": self.model, "prompt": "a red panda astronaut", "seed": 7}

    def test_provider_native_params_pass_through(self):
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt="a red panda astronaut",
            optional_params={"output_format": "webp", "input_references": ["https://example.com/a.png"]},
            litellm_params={},
            headers={},
        )

        assert result["output_format"] == "webp"
        assert result["input_references"] == ["https://example.com/a.png"]


class TestOpenRouterImageGenerationResponse:
    def setup_method(self):
        self.config = OpenRouterImageGenerationConfig()
        self.model = "krea/krea-2-large"
        self.logging_obj = None

    def _transform(self, payload: object, status_code: int = 200) -> ImageResponse:
        return self.config.transform_image_generation_response(
            model=self.model,
            raw_response=make_httpx_response(payload, status_code=status_code),
            model_response=ImageResponse(),
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    def test_extracts_base64_image(self):
        result = self._transform(OPENROUTER_IMAGES_RESPONSE)

        assert len(result.data) == 1
        assert result.data[0].b64_json == "iVBORw0KGgoAAAANS"
        assert result.data[0].url is None

    def test_extracts_multiple_images(self):
        payload = {
            "created": 0,
            "data": [
                {"b64_json": "image1data", "media_type": "image/png"},
                {"b64_json": "image2data", "media_type": "image/png"},
            ],
        }

        result = self._transform(payload)

        assert [image.b64_json for image in result.data] == ["image1data", "image2data"]

    def test_extracts_url_image(self):
        payload = {"created": 0, "data": [{"url": "https://example.com/image.png"}]}

        result = self._transform(payload)

        assert result.data[0].url == "https://example.com/image.png"
        assert result.data[0].b64_json is None

    def test_maps_usage_and_cost(self):
        result = self._transform(OPENROUTER_IMAGES_RESPONSE)

        assert result.usage is not None
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 4175
        assert result.usage.total_tokens == 4175
        assert result.usage.input_tokens_details.image_tokens == 0
        assert result.usage.input_tokens_details.text_tokens == 0
        assert result._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.06
        assert result._hidden_params["response_cost_details"]["upstream_inference_cost"] == 0.06

    def test_falls_back_to_completion_tokens_when_image_tokens_absent(self):
        payload = {
            "data": [{"b64_json": "abc"}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 1299, "total_tokens": 1305},
        }

        result = self._transform(payload)

        assert result.usage.output_tokens == 1299
        assert result.usage.input_tokens == 6

    def test_zero_created_does_not_overwrite_litellm_timestamp(self):
        """OpenRouter answers `created: 0`; clobbering the response timestamp with it breaks clients."""
        result = self._transform(OPENROUTER_IMAGES_RESPONSE)

        assert result.created > 0

    def test_created_is_used_when_openrouter_sends_one(self):
        payload = {"created": 1785224458, "data": [{"b64_json": "abc"}]}

        result = self._transform(payload)

        assert result.created == 1785224458

    def test_reports_response_model(self):
        payload = {"data": [{"b64_json": "abc"}], "model": "krea/krea-2-large"}

        result = self._transform(payload)

        assert result._hidden_params["model"] == "krea/krea-2-large"

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
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=raw_response,
                model_response=ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        assert exc_info.value.status_code == 502

    def test_get_error_class(self):
        error = self.config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, OpenRouterException)
        assert "Test error" in str(error)
        assert error.status_code == 400


class TestOpenRouterImageGenerationEnvironment:
    def setup_method(self):
        self.config = OpenRouterImageGenerationConfig()
        self.model = "krea/krea-2-large"

    def test_validate_environment_prefers_explicit_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env_key")

        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test_api_key",
        )

        assert headers["Authorization"] == "Bearer test_api_key"

    def test_validate_environment_falls_back_to_env_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "secret_api_key")

        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )

        assert headers["Authorization"] == "Bearer secret_api_key"


class TestOpenRouterImageGenerationEndToEnd:
    """Covers param mapping + URL + body together, the way the proxy drives it."""

    def test_openai_style_request_reaches_the_openrouter_images_endpoint(self):
        recorder = RequestRecorder(OPENROUTER_IMAGES_RESPONSE)

        response = litellm.image_generation(
            model="openrouter/krea/krea-2-large",
            prompt="a red panda astronaut",
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
            "model": "krea/krea-2-large",
            "prompt": "a red panda astronaut",
            "n": 1,
            "aspect_ratio": "1:1",
        }
        assert response.data[0].b64_json == "iVBORw0KGgoAAAANS"

    def test_hybrid_image_model_uses_the_same_endpoint(self):
        recorder = RequestRecorder(OPENROUTER_IMAGES_RESPONSE)

        litellm.image_generation(
            model="openrouter/google/gemini-2.5-flash-image",
            prompt="a red panda astronaut",
            api_key="sk-test",
            api_base="https://openrouter.ai/api/v1",
            client=make_client(recorder),
        )

        assert str(recorder.request.url) == "https://openrouter.ai/api/v1/images"
        assert recorder.body() == {
            "model": "google/gemini-2.5-flash-image",
            "prompt": "a red panda astronaut",
        }

    def test_openrouter_error_is_surfaced(self):
        recorder = RequestRecorder(
            {"error": {"message": "No endpoints found matching your data policy", "code": 404}},
            status_code=404,
        )

        with pytest.raises(litellm.NotFoundError):
            litellm.image_generation(
                model="openrouter/krea/krea-2-large",
                prompt="a red panda astronaut",
                api_key="sk-test",
                api_base="https://openrouter.ai/api/v1",
                client=make_client(recorder),
            )
