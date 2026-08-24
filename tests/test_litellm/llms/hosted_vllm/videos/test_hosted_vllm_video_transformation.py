"""Tests for hosted_vllm video generation (vLLM-Omni /v1/videos)."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.hosted_vllm.videos.transformation import (
    HostedVLLMVideoConfig,
    _serialize_form_value,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import VideoObject
from litellm.utils import ProviderConfigManager


def _form_fields(files: list) -> dict[str, str]:
    return {name: value[1] for name, value in files if value[0] is None}


def test_provider_config_registration():
    config = ProviderConfigManager.get_provider_video_config(
        model="hosted_vllm/MiniMax-H3",
        provider=LlmProviders.HOSTED_VLLM,
    )

    assert config is not None
    assert isinstance(config, HostedVLLMVideoConfig)


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

    assert data == {}
    assert url == "http://localhost:8091/v1/videos"
    assert files
    fields = _form_fields(files)
    assert fields["model"] == "MiniMax-H3"
    assert fields["prompt"] == "three cats march into a bedroom playing tiny brass instruments"
    assert fields["width"] == "1280"
    assert fields["height"] == "720"
    assert fields["fps"] == "24"
    assert fields["num_inference_steps"] == "20"
    assert fields["flow_shift"] == "12"
    assert fields["seed"] == "1101"
    assert fields["aspect_ratio"] == "16:9"
    assert json.loads(fields["extra_params"]) == extra_params
    assert "extra_headers" not in fields


def test_transform_video_create_request_keeps_openai_size_and_seconds():
    config = HostedVLLMVideoConfig()

    _, files, _ = config.transform_video_create_request(
        model="Wan2.2",
        prompt="a mountain lake at sunrise",
        api_base="http://localhost:8091/v1/videos",
        video_create_optional_request_params={"seconds": "8", "size": "1280x720"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    fields = _form_fields(files)
    assert fields["seconds"] == "8"
    assert fields["size"] == "1280x720"


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

    assert data == {}
    fields = _form_fields(files)
    assert fields["width"] == "832"
    assert "input_reference" not in fields
    reference_parts = [value for name, value in files if name == "input_reference"]
    assert len(reference_parts) == 1
    filename, content, content_type = reference_parts[0]
    assert filename == "input.png"
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


def _mock_http_client(response_body: dict) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = response_body
    mock_response.text = json.dumps(response_body)
    mock_client.post.return_value = mock_response
    return mock_client


def test_video_generation_posts_multipart_not_json():
    mock_client = _mock_http_client(
        {
            "id": "video-123",
            "object": "video",
            "status": "queued",
            "created_at": 1701234567,
        }
    )

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=mock_client,
    ):
        response = litellm.video_generation(
            model="hosted_vllm/MiniMax-H3",
            prompt="three cats march into a bedroom playing tiny brass instruments",
            api_base="http://localhost:8091",
            api_key="test-key",
            extra_body={
                "width": 1280,
                "height": 720,
                "fps": 24,
                "extra_params": {"task": "t2va", "duration": 10.0},
            },
        )

    assert isinstance(response, VideoObject)
    assert response.status == "queued"
    mock_client.post.assert_called_once()
    post_kwargs = mock_client.post.call_args.kwargs
    assert post_kwargs["url"] == "http://localhost:8091/v1/videos"
    assert post_kwargs.get("json") is None
    assert post_kwargs["files"]
    fields = _form_fields(post_kwargs["files"])
    assert fields["prompt"] == "three cats march into a bedroom playing tiny brass instruments"
    assert fields["width"] == "1280"
    assert json.loads(fields["extra_params"]) == {"task": "t2va", "duration": 10.0}
    assert post_kwargs["headers"]["Authorization"] == "Bearer test-key"
