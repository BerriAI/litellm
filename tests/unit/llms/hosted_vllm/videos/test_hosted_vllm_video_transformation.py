"""Tests for hosted_vllm video generation (vLLM-Omni /v1/videos)."""

import json
from io import BytesIO

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.url_utils import SSRFError
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.hosted_vllm.videos import get_hosted_vllm_video_config
from litellm.llms.hosted_vllm.videos.transformation import (
    HostedVLLMVideoConfig,
    _serialize_form_value,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import VideoObject
from litellm.utils import ProviderConfigManager


def test_provider_config_registration():
    config = ProviderConfigManager.get_provider_video_config(
        model="hosted_vllm/MiniMax-H3",
        provider=LlmProviders.HOSTED_VLLM,
    )

    assert config is not None
    assert isinstance(config, HostedVLLMVideoConfig)
    assert isinstance(get_hosted_vllm_video_config("MiniMax-H3"), HostedVLLMVideoConfig)


def test_get_complete_url_appends_videos():
    config = HostedVLLMVideoConfig()

    assert (
        config.get_complete_url(model="MiniMax-H3", api_base="http://localhost:8091", litellm_params={})
        == "http://localhost:8091/v1/videos"
    )
    assert (
        config.get_complete_url(model="MiniMax-H3", api_base="http://localhost:8091/v1", litellm_params={})
        == "http://localhost:8091/v1/videos"
    )
    assert (
        config.get_complete_url(model="MiniMax-H3", api_base="http://localhost:8091/v1/", litellm_params={})
        == "http://localhost:8091/v1/videos"
    )


def test_get_complete_url_requires_api_base():
    config = HostedVLLMVideoConfig()

    with pytest.raises(ValueError, match="api_base not set"):
        config.get_complete_url(model="MiniMax-H3", api_base=None, litellm_params={})


def test_validate_environment_defaults_to_fake_api_key():
    config = HostedVLLMVideoConfig()

    headers = config.validate_environment(
        headers={},
        model="MiniMax-H3",
        litellm_params=GenericLiteLLMParams(),
    )

    assert headers.get("Authorization") == "Bearer fake-api-key"


def test_validate_environment_uses_provided_api_key():
    config = HostedVLLMVideoConfig()

    headers = config.validate_environment(
        headers={"X-Test": "1"},
        model="MiniMax-H3",
        litellm_params=GenericLiteLLMParams(api_key="my-custom-key"),
    )

    assert headers.get("Authorization") == "Bearer my-custom-key"
    assert headers.get("X-Test") == "1"


def test_transform_video_create_request_uses_multipart_form_fields():
    """vLLM-Omni rejects JSON create bodies. Extra Omni fields must be form parts."""
    config = HostedVLLMVideoConfig()
    extra_params = {"task": "t2va", "duration": 10.0, "audio_flow_shift": 3.0}

    data, files, url = config.transform_video_create_request(
        model="MiniMax-H3",
        prompt="three cats march into a bedroom playing tiny brass instruments",
        api_base="http://localhost:8091/v1/videos",
        video_create_optional_request_params={
            "width": 1280,
            "height": 720,
            "fps": 24,
            "num_inference_steps": 20,
            "flow_shift": 12,
            "seed": 1101,
            "aspect_ratio": "16:9",
            "extra_params": extra_params,
            "extra_headers": {"X-Ignored": "yes"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert url == "http://localhost:8091/v1/videos"
    assert files == ()
    assert data["model"] == "MiniMax-H3"
    assert data["prompt"] == "three cats march into a bedroom playing tiny brass instruments"
    assert data["width"] == "1280"
    assert data["height"] == "720"
    assert data["fps"] == "24"
    assert data["num_inference_steps"] == "20"
    assert data["flow_shift"] == "12"
    assert data["seed"] == "1101"
    assert data["aspect_ratio"] == "16:9"
    assert json.loads(data["extra_params"]) == extra_params
    assert "extra_headers" not in data


def test_transform_video_create_request_keeps_openai_size_and_seconds():
    config = HostedVLLMVideoConfig()

    data, files, _ = config.transform_video_create_request(
        model="Wan2.2",
        prompt="a mountain lake at sunrise",
        api_base="http://localhost:8091/v1/videos",
        video_create_optional_request_params={"seconds": "8", "size": "1280x720"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert files == ()
    assert data["seconds"] == "8"
    assert data["size"] == "1280x720"


def test_transform_video_create_request_attaches_input_reference_file():
    config = HostedVLLMVideoConfig()
    reference = BytesIO(b"fake-png")
    reference.name = "input.png"

    data, files, _ = config.transform_video_create_request(
        model="Wan2.2",
        prompt="animate this image",
        api_base="http://localhost:8091/v1/videos",
        video_create_optional_request_params={"input_reference": reference, "width": 832},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert data["width"] == "832"
    assert "input_reference" not in data
    reference_parts = [value for name, value in files if name == "input_reference"]
    assert len(reference_parts) == 1
    filename, content, content_type = reference_parts[0]
    assert filename == "input_reference.png"
    assert content is reference
    assert content_type == "image/png"


def test_serialize_form_value_does_not_quote_plain_strings():
    assert _serialize_form_value("16:9") == "16:9"
    assert _serialize_form_value(True) == "true"
    assert _serialize_form_value({"task": "t2va"}) == json.dumps({"task": "t2va"})


def test_map_openai_params_passes_through_omni_fields():
    config = HostedVLLMVideoConfig()

    mapped = config.map_openai_params(
        video_create_optional_params={
            "width": 1280,
            "extra_params": {"task": "t2va"},
            "aspect_ratio": "16:9",
            "extra_body": None,
        },
        model="MiniMax-H3",
        drop_params=False,
    )

    assert mapped["width"] == 1280
    assert mapped["extra_params"] == {"task": "t2va"}
    assert mapped["aspect_ratio"] == "16:9"
    assert "extra_body" not in mapped


def test_get_supported_openai_params_includes_omni_extensions():
    config = HostedVLLMVideoConfig()
    supported = config.get_supported_openai_params("MiniMax-H3")

    assert "prompt" in supported
    assert "input_reference" in supported
    assert "width" in supported
    assert "extra_params" in supported
    assert "aspect_ratio" in supported
    assert "image_reference" in supported
    assert "audio_reference" in supported


def _http_handler_for(handler) -> HTTPHandler:
    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_video_generation_posts_multipart_not_json():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "video-123",
                "object": "video",
                "status": "queued",
                "created_at": 1701234567,
            },
        )

    response = litellm.video_generation(
        model="hosted_vllm/MiniMax-H3",
        prompt="three cats march into a bedroom playing tiny brass instruments",
        api_base="http://localhost:8091",
        api_key="test-key",
        client=_http_handler_for(handler),
        extra_body={
            "width": 1280,
            "height": 720,
            "fps": 24,
            "extra_params": {"task": "t2va", "duration": 10.0},
        },
    )

    assert isinstance(response, VideoObject)
    assert response.status == "queued"
    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == "http://localhost:8091/v1/videos"
    assert request.headers["authorization"] == "Bearer test-key"
    body = request.content
    assert b'name="prompt"' in body
    assert b"three cats march into a bedroom playing tiny brass instruments" in body
    assert b'name="width"' in body
    assert b"1280" in body
    assert b'name="extra_params"' in body
    assert b"t2va" in body
    assert request.headers.get("content-type", "").startswith("multipart/form-data")


def test_http_image_reference_is_forwarded_not_downloaded():
    config = HostedVLLMVideoConfig()
    data, files, _ = config.transform_video_create_request(
        model="MiniMax-H3",
        prompt="a person singing",
        api_base="http://localhost:8091/v1/videos",
        video_create_optional_request_params={
            "image_reference": {"image_url": "http://1.1.1.1/face.png"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert files == ()
    payload = json.loads(data["image_reference"])
    assert payload["image_url"] == "http://1.1.1.1/face.png"


def test_data_url_image_reference_is_forwarded():
    data_url = "data:image/png;base64,AAAA"
    config = HostedVLLMVideoConfig()
    data, files, _ = config.transform_video_create_request(
        model="MiniMax-H3",
        prompt="a person singing",
        api_base="http://localhost:8091/v1/videos",
        video_create_optional_request_params={"image_reference": {"image_url": data_url}},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert files == ()
    assert json.loads(data["image_reference"])["image_url"] == data_url


def test_metadata_url_in_image_reference_is_rejected():
    config = HostedVLLMVideoConfig()
    with pytest.raises(SSRFError, match="blocked address"):
        config.transform_video_create_request(
            model="MiniMax-H3",
            prompt="a person singing",
            api_base="http://localhost:8091/v1/videos",
            video_create_optional_request_params={
                "image_reference": {"image_url": "http://169.254.169.254/latest/meta-data/"},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


def test_file_scheme_media_reference_is_rejected():
    config = HostedVLLMVideoConfig()
    with pytest.raises(SSRFError, match="scheme"):
        config.transform_video_create_request(
            model="MiniMax-H3",
            prompt="a person singing",
            api_base="http://localhost:8091/v1/videos",
            video_create_optional_request_params={
                "video_reference": {"video_url": "file:///etc/passwd"},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
