import json
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.image_generation.transformation import (
    OpenRouterImageGenerationConfig,
)
from litellm.types.utils import ImageResponse, ImageUsage, ImageUsageInputTokensDetails

CONFIG: Final = OpenRouterImageGenerationConfig()
IMAGE_ONLY_MODEL: Final = "openai/gpt-image-1-mini"
HYBRID_MODEL: Final = "google/gemini-2.5-flash-image"
PROMPT: Final = "a small red apple on a white table, simple flat illustration"
IMAGES_URL: Final = "https://openrouter.ai/api/v1/images"

# usage object returned by a real POST https://openrouter.ai/api/v1/images call for openai/gpt-image-1-mini
# (quality low, 1024x1024) on 2026-10-03
OPENROUTER_IMAGES_USAGE: Final = {
    "prompt_tokens": 18,
    "completion_tokens": 272,
    "total_tokens": 290,
    "cost": 0.002212,
    "is_byok": False,
    "prompt_tokens_details": {"cached_tokens": 0},
    "cost_details": {
        "upstream_inference_cost": 0.002212,
        "upstream_inference_prompt_cost": 3.6e-05,
        "upstream_inference_completions_cost": 0.002176,
    },
    "completion_tokens_details": {"reasoning_tokens": 0, "image_tokens": 272},
}


def _images_response(*b64_images: str, created: int = 1790994427) -> dict[str, object]:
    return {
        "created": created,
        "data": [{"b64_json": image, "media_type": "image/png"} for image in b64_images],
        "usage": OPENROUTER_IMAGES_USAGE,
    }


def _transform_response(raw_response: httpx.Response) -> ImageResponse:
    return CONFIG.transform_image_generation_response(
        model=IMAGE_ONLY_MODEL,
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


class RequestRecorder:
    """httpx.MockTransport handler that keeps every request it was called with"""

    def __init__(self, response_payload: object, status_code: int = 200) -> None:
        self.response_payload = response_payload
        self.status_code = status_code
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(status_code=self.status_code, json=self.response_payload)


def _client(recorder: RequestRecorder) -> HTTPHandler:
    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(recorder)))


def test_get_supported_openai_params():
    assert CONFIG.get_supported_openai_params(IMAGE_ONLY_MODEL) == ["size", "quality", "n"]


@pytest.mark.parametrize(
    ("api_base", "expected_url"),
    [
        (None, IMAGES_URL),
        ("https://openrouter.ai/api/v1", IMAGES_URL),
        ("https://openrouter.ai/api/v1/", IMAGES_URL),
        ("https://openrouter.ai/api/v1/chat/completions", IMAGES_URL),
        ("https://gateway.example.com/openrouter/v1", "https://gateway.example.com/openrouter/v1/images"),
        ("https://gateway.example.com/api/v1/images", "https://gateway.example.com/api/v1/images"),
    ],
)
def test_get_complete_url_points_at_the_images_endpoint(api_base: str | None, expected_url: str):
    url = CONFIG.get_complete_url(
        api_base=api_base,
        api_key="sk-test",
        model=IMAGE_ONLY_MODEL,
        optional_params={},
        litellm_params={},
    )

    assert url == expected_url


@pytest.mark.parametrize(
    ("non_default_params", "expected_params"),
    [
        ({"size": "1536x1024"}, {"size": "1536x1024"}),
        ({"size": "auto"}, {}),
        ({"quality": "low"}, {"quality": "low"}),
        ({"quality": "medium"}, {"quality": "medium"}),
        ({"quality": "high"}, {"quality": "high"}),
        ({"quality": "auto"}, {"quality": "auto"}),
        ({"quality": "standard"}, {"quality": "low"}),
        ({"quality": "hd"}, {"quality": "high"}),
        ({"n": 2}, {"n": 2}),
    ],
)
def test_map_openai_params_sends_size_quality_and_n_as_images_fields(
    non_default_params: dict[str, object], expected_params: dict[str, object]
):
    mapped = CONFIG.map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model=IMAGE_ONLY_MODEL,
        drop_params=False,
    )

    assert mapped == expected_params


@pytest.mark.parametrize(
    ("drop_params", "expected_params"),
    [
        (False, {"size": "1024x1024", "unsupported_param": "value"}),
        (True, {"size": "1024x1024"}),
    ],
)
def test_map_openai_params_unsupported_param_follows_drop_params(drop_params: bool, expected_params: dict[str, object]):
    mapped = CONFIG.map_openai_params(
        non_default_params={"size": "1024x1024", "unsupported_param": "value"},
        optional_params={},
        model=IMAGE_ONLY_MODEL,
        drop_params=drop_params,
    )

    assert mapped == expected_params


def test_map_openai_params_keeps_params_already_in_optional_params():
    mapped = CONFIG.map_openai_params(
        non_default_params={"n": 1},
        optional_params={"resolution": "2K"},
        model=HYBRID_MODEL,
        drop_params=False,
    )

    assert mapped == {"resolution": "2K", "n": 1}


@patch("litellm.llms.openrouter.image_generation.transformation.get_secret_str")
def test_validate_environment_with_api_key(mock_get_secret: MagicMock):
    result = CONFIG.validate_environment(
        headers={},
        model=HYBRID_MODEL,
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test_api_key",
    )

    assert result["Authorization"] == "Bearer test_api_key"
    mock_get_secret.assert_not_called()


@patch("litellm.llms.openrouter.image_generation.transformation.get_secret_str")
def test_validate_environment_with_secret_key(mock_get_secret: MagicMock):
    mock_get_secret.return_value = "secret_api_key"

    result = CONFIG.validate_environment(
        headers={},
        model=HYBRID_MODEL,
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
    )

    assert result["Authorization"] == "Bearer secret_api_key"
    mock_get_secret.assert_called_once_with("OPENROUTER_API_KEY")


def test_transform_request_body_holds_only_images_fields():
    body = CONFIG.transform_image_generation_request(
        model=IMAGE_ONLY_MODEL,
        prompt=PROMPT,
        optional_params={
            "size": "1024x1024",
            "quality": "low",
            "n": 1,
            "modalities": ["image", "text"],
            "stream": True,
            "extra_headers": {"Authorization": "Bearer sk-test"},
        },
        litellm_params={},
        headers={},
    )

    assert body == {"model": IMAGE_ONLY_MODEL, "prompt": PROMPT, "size": "1024x1024", "quality": "low", "n": 1}


@pytest.mark.parametrize(
    ("optional_params", "expected_fields"),
    [
        (
            {"image_config": {"aspect_ratio": "16:9", "image_size": "4K"}},
            {"aspect_ratio": "16:9", "resolution": "4K"},
        ),
        (
            {"image_config": {"aspect_ratio": "16:9", "image_size": "4K"}, "aspect_ratio": "1:1", "resolution": "2K"},
            {"aspect_ratio": "1:1", "resolution": "2K"},
        ),
    ],
)
def test_transform_request_maps_legacy_image_config_and_explicit_fields_win(
    optional_params: dict[str, object], expected_fields: dict[str, object]
):
    body = CONFIG.transform_image_generation_request(
        model=HYBRID_MODEL,
        prompt=PROMPT,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert body == {"model": HYBRID_MODEL, "prompt": PROMPT, **expected_fields}


# On 2026-10-03 POST https://openrouter.ai/api/v1/images returned 400 for size "1024x1024" with aspect_ratio
# "3:2" (openai/gpt-image-1-mini) and with resolution "2K" (google/gemini-2.5-flash-image). The size field in
# https://openrouter.ai/openapi.json says a tier size such as "2K" combines with aspect_ratio
@pytest.mark.parametrize(
    ("optional_params", "expected_fields"),
    [
        ({"size": "1024x1024", "image_config": {"aspect_ratio": "16:9"}}, {"aspect_ratio": "16:9"}),
        ({"size": "1024x1024", "resolution": "4K"}, {"resolution": "4K"}),
        ({"size": "1024x1024", "aspect_ratio": "1:1"}, {"aspect_ratio": "1:1"}),
        ({"size": "2K", "aspect_ratio": "16:9"}, {"size": "2K", "aspect_ratio": "16:9"}),
        ({"size": "1024x1024"}, {"size": "1024x1024"}),
    ],
)
def test_transform_request_lets_aspect_ratio_or_resolution_win_over_a_pixel_size(
    optional_params: dict[str, object], expected_fields: dict[str, object]
):
    body = CONFIG.transform_image_generation_request(
        model=HYBRID_MODEL,
        prompt=PROMPT,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert body == {"model": HYBRID_MODEL, "prompt": PROMPT, **expected_fields}


def test_transform_response_returns_every_image_in_order():
    response = _transform_response(httpx.Response(200, json=_images_response("aW1hZ2Ux", "aW1hZ2Uy")))

    assert [(image.b64_json, image.url) for image in response.data] == [("aW1hZ2Ux", None), ("aW1hZ2Uy", None)]


def test_transform_response_copies_the_openrouter_created_timestamp():
    response = _transform_response(httpx.Response(200, json=_images_response("aW1hZ2Ux", created=1790994427)))

    assert response.created == 1790994427


def test_transform_response_with_zero_created_keeps_a_real_timestamp():
    response = _transform_response(httpx.Response(200, json=_images_response("aW1hZ2Ux", created=0)))

    assert response.created > 0


def test_transform_response_reports_openrouter_usage_and_cost():
    response = _transform_response(httpx.Response(200, json=_images_response("aW1hZ2Ux")))

    assert response.usage == ImageUsage(
        input_tokens=18,
        input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=18),
        output_tokens=272,
        total_tokens=290,
    )
    assert response._hidden_params["additional_headers"] == {
        "llm_provider-x-litellm-response-cost": OPENROUTER_IMAGES_USAGE["cost"]
    }
    assert response._hidden_params["response_cost_details"] == OPENROUTER_IMAGES_USAGE["cost_details"]
    assert response._hidden_params["model"] == IMAGE_ONLY_MODEL


# The ImageGenerationUsage schema in https://openrouter.ai/openapi.json (2026-10-03) requires only
# prompt_tokens, completion_tokens and total_tokens, allows null for completion_tokens_details and
# image_tokens, and its example for a per-image priced model has no completion_tokens_details
PER_IMAGE_USAGE: Final = {"prompt_tokens": 0, "completion_tokens": 4175, "total_tokens": 4175, "cost": 0.04}


@pytest.mark.parametrize(
    "usage",
    [
        PER_IMAGE_USAGE,
        {**PER_IMAGE_USAGE, "completion_tokens_details": None},
        {**PER_IMAGE_USAGE, "completion_tokens_details": {"image_tokens": None}},
    ],
    ids=["no-details", "null-details", "null-image-tokens"],
)
def test_transform_response_without_image_tokens_reports_completion_tokens_and_cost(usage: dict[str, object]):
    response = _transform_response(
        httpx.Response(200, json={"created": 1790994427, "data": [{"b64_json": "aW1hZ2Ux"}], "usage": usage})
    )

    assert response.usage == ImageUsage(
        input_tokens=0,
        input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=0),
        output_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
    )
    assert response._hidden_params["additional_headers"] == {"llm_provider-x-litellm-response-cost": usage["cost"]}


def test_transform_response_with_non_json_body_raises_openrouter_exception():
    with pytest.raises(OpenRouterException, match="Error parsing OpenRouter response") as exc_info:
        _transform_response(httpx.Response(502, content=b"<html>bad gateway</html>"))

    assert exc_info.value.status_code == 502
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_get_error_class():
    error = CONFIG.get_error_class(
        error_message="Test error",
        status_code=400,
        headers={"Content-Type": "application/json"},
    )

    assert isinstance(error, OpenRouterException)
    assert "Test error" in str(error)
    assert error.status_code == 400


def test_image_only_model_is_sent_to_the_images_endpoint_and_charged_the_openrouter_cost():
    recorder = RequestRecorder(_images_response("aW1hZ2Ux"))

    response = litellm.image_generation(
        model=f"openrouter/{IMAGE_ONLY_MODEL}",
        prompt=PROMPT,
        size="1024x1024",
        quality="low",
        n=1,
        api_key="sk-test",
        client=_client(recorder),
    )

    (request,) = recorder.requests
    assert str(request.url) == IMAGES_URL
    assert request.headers["Authorization"] == "Bearer sk-test"
    assert json.loads(request.content) == {
        "model": IMAGE_ONLY_MODEL,
        "prompt": PROMPT,
        "size": "1024x1024",
        "quality": "low",
        "n": 1,
    }
    assert [image.b64_json for image in response.data] == ["aW1hZ2Ux"]
    assert response._hidden_params["response_cost"] == OPENROUTER_IMAGES_USAGE["cost"]


def test_hybrid_image_text_model_uses_the_same_images_endpoint():
    recorder = RequestRecorder(_images_response("aW1hZ2Ux"))

    litellm.image_generation(
        model=f"openrouter/{HYBRID_MODEL}",
        prompt=PROMPT,
        api_key="sk-test",
        client=_client(recorder),
    )

    (request,) = recorder.requests
    assert str(request.url) == IMAGES_URL
    assert json.loads(request.content) == {"model": HYBRID_MODEL, "prompt": PROMPT}


def test_legacy_image_config_with_an_openai_pixel_size_sends_only_the_aspect_ratio():
    recorder = RequestRecorder(_images_response("aW1hZ2Ux"))

    litellm.image_generation(
        model=f"openrouter/{HYBRID_MODEL}",
        prompt=PROMPT,
        size="1024x1024",
        image_config={"aspect_ratio": "16:9"},
        api_key="sk-test",
        client=_client(recorder),
    )

    (request,) = recorder.requests
    assert json.loads(request.content) == {"model": HYBRID_MODEL, "prompt": PROMPT, "aspect_ratio": "16:9"}


def test_legacy_chat_completions_api_base_still_reaches_the_images_endpoint():
    recorder = RequestRecorder(_images_response("aW1hZ2Ux"))

    litellm.image_generation(
        model=f"openrouter/{IMAGE_ONLY_MODEL}",
        prompt=PROMPT,
        api_key="sk-test",
        api_base="https://openrouter.ai/api/v1/chat/completions",
        client=_client(recorder),
    )

    (request,) = recorder.requests
    assert str(request.url) == IMAGES_URL


def test_openrouter_error_response_surfaces_as_not_found_error():
    recorder = RequestRecorder({"error": {"code": 404, "message": "Resource not found"}}, status_code=404)

    with pytest.raises(litellm.NotFoundError, match="Resource not found"):
        litellm.image_generation(
            model=f"openrouter/{IMAGE_ONLY_MODEL}",
            prompt=PROMPT,
            api_key="sk-test",
            client=_client(recorder),
        )
