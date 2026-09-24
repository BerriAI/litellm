"""Unit tests for Bedrock Amazon Nova Reel video generation (issue #39552)."""

import base64
import io
import json
from typing import cast
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm
from litellm.llms.bedrock.videos.handler import BedrockVideoGeneration
from litellm.llms.bedrock.videos.transformation import BedrockNovaReelVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams
from litellm.types.videos.utils import decode_video_id_with_provider

TEST_MODEL = "amazon.nova-reel-v1:0"
TEST_ARN = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123-def456"


def _make_config() -> BedrockNovaReelVideoConfig:
    return BedrockNovaReelVideoConfig()


def _create_request(
    optional_params: dict | None = None,
    prompt: str = "A drone shot over the ocean",
    model: str = TEST_MODEL,
) -> dict:
    config = _make_config()
    params: dict = (
        optional_params if optional_params is not None else {"output_s3_uri": "s3://bucket/out/"}
    )
    body, files, method = config.transform_video_create_request(
        model=model,
        prompt=prompt,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/async-invoke",
        video_create_optional_request_params=cast(VideoCreateOptionalRequestParams, params),
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert files is None
    assert method == "POST"
    return body


#################################################
# Request transform
#################################################


def test_transform_create_request_defaults():
    body = _create_request()
    assert body["modelId"] == TEST_MODEL
    assert body["outputDataConfig"] == {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}
    model_input = body["modelInput"]
    assert model_input["taskType"] == "TEXT_VIDEO"
    assert model_input["textToVideoParams"]["text"] == "A drone shot over the ocean"
    assert "images" not in model_input["textToVideoParams"]
    cfg = model_input["videoGenerationConfig"]
    assert cfg["durationSeconds"] == 6
    assert cfg["fps"] == 24
    assert cfg["dimension"] == "1280x720"


def test_transform_create_request_seconds_and_size_map_to_generation_config():
    body = _create_request(
        {"output_s3_uri": "s3://bucket/out/", "seconds": "10", "size": "720x1280", "seed": 5}
    )
    cfg = body["modelInput"]["videoGenerationConfig"]
    assert cfg["durationSeconds"] == 10
    assert cfg["dimension"] == "720x1280"
    assert cfg["seed"] == 5


def test_transform_create_request_input_reference_becomes_images():
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 8
    body = _create_request(
        {"output_s3_uri": "s3://bucket/out/", "input_reference": io.BytesIO(png_bytes)}
    )
    images = body["modelInput"]["textToVideoParams"]["images"]
    assert len(images) == 1
    assert images[0]["format"] == "png"
    assert images[0]["source"]["bytes"] == base64.b64encode(png_bytes).decode("utf-8")


def test_transform_create_request_jpeg_detected():
    jpeg_bytes = b"\xff\xd8\xff\xe0" + b"0" * 8
    body = _create_request(
        {"output_s3_uri": "s3://bucket/out/", "input_reference": jpeg_bytes}
    )
    assert body["modelInput"]["textToVideoParams"]["images"][0]["format"] == "jpeg"


def test_transform_create_request_requires_output_s3_uri():
    with pytest.raises(ValueError, match="output_s3_uri"):
        _create_request({})


def test_transform_create_request_task_type_passthrough():
    body = _create_request(
        {"output_s3_uri": "s3://bucket/out/", "taskType": "MULTI_SHOT_AUTOMATED"}
    )
    assert body["modelInput"]["taskType"] == "MULTI_SHOT_AUTOMATED"


#################################################
# Create + status response transforms
#################################################


def test_transform_create_response_maps_invocation_arn():
    config = _make_config()
    resp = httpx.Response(200, json={"invocationArn": TEST_ARN})
    video = config.transform_video_create_response(
        model=TEST_MODEL,
        raw_response=resp,
        logging_obj=None,
        request_data={
            "modelInput": {"videoGenerationConfig": {"durationSeconds": 6}}
        },
    )
    assert video.status == "processing"
    assert video.model == TEST_MODEL
    assert video.usage is not None
    assert video.usage["duration_seconds"] == 6.0
    decoded = decode_video_id_with_provider(video.id)
    assert decoded["custom_llm_provider"] == "bedrock"
    assert decoded["model_id"] == TEST_MODEL
    assert decoded["video_id"] == TEST_ARN


def test_transform_create_response_missing_arn_raises():
    config = _make_config()
    resp = httpx.Response(200, json={})
    with pytest.raises(ValueError, match="invocationArn"):
        config.transform_video_create_response(
            model=TEST_MODEL, raw_response=resp, logging_obj=None
        )


@pytest.mark.parametrize(
    "raw_status,expected",
    [
        ("InProgress", "processing"),
        ("Completed", "completed"),
        ("Failed", "failed"),
    ],
)
def test_transform_status_response_maps_aws_enum(raw_status, expected):
    config = _make_config()
    resp = httpx.Response(
        200,
        json={
            "invocationArn": TEST_ARN,
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-reel-v1:0",
            "status": raw_status,
            "submitTime": 1758000000.0,
            "lastModifiedTime": 1758000060.0,
            "endTime": 1758000060.0 if raw_status != "InProgress" else None,
            "failureMessage": "blocked by content filters" if raw_status == "Failed" else None,
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}},
        },
    )
    video = config.transform_video_status_retrieve_response(
        raw_response=resp, logging_obj=None, model=TEST_MODEL
    )
    assert video.status == expected
    if raw_status == "Failed":
        assert video.error == {"message": "blocked by content filters"}
    if raw_status == "Completed":
        assert video.completed_at == 1758000060
        assert video.usage is not None
        assert video.usage["output_s3_uri"] == "s3://bucket/out/"
    decoded = decode_video_id_with_provider(video.id)
    assert decoded["video_id"] == TEST_ARN


#################################################
# Video id encoding round-trip
#################################################


def test_video_id_encoding_round_trip():
    from litellm.types.videos.utils import encode_video_id_with_provider

    encoded = encode_video_id_with_provider(TEST_ARN, "bedrock", TEST_MODEL)
    decoded = decode_video_id_with_provider(encoded)
    assert decoded["custom_llm_provider"] == "bedrock"
    assert decoded["model_id"] == TEST_MODEL
    assert decoded["video_id"] == TEST_ARN
    assert _make_config().extract_invocation_arn(encoded) == TEST_ARN


#################################################
# Provider config dispatch
#################################################


def test_provider_config_manager_returns_nova_reel_config():
    from litellm.utils import ProviderConfigManager

    cfg = ProviderConfigManager.get_provider_video_config(
        TEST_MODEL, litellm.LlmProviders.BEDROCK
    )
    assert isinstance(cfg, BedrockNovaReelVideoConfig)


def test_provider_config_manager_us_cross_region_variant():
    from litellm.utils import ProviderConfigManager

    cfg = ProviderConfigManager.get_provider_video_config(
        "us.amazon.nova-reel-v1:0", litellm.LlmProviders.BEDROCK
    )
    assert isinstance(cfg, BedrockNovaReelVideoConfig)


def test_provider_config_manager_non_reel_bedrock_returns_none():
    from litellm.utils import ProviderConfigManager

    cfg = ProviderConfigManager.get_provider_video_config(
        "amazon.titan-text-express-v1", litellm.LlmProviders.BEDROCK
    )
    assert cfg is None


def test_provider_config_manager_model_none_returns_config():
    """Status/content routes pass model=None; the model lives in the video id."""
    from litellm.utils import ProviderConfigManager

    cfg = ProviderConfigManager.get_provider_video_config(None, litellm.LlmProviders.BEDROCK)
    assert isinstance(cfg, BedrockNovaReelVideoConfig)


#################################################
# Handler request construction (no network)
#################################################


class _FakeCredentialsInfo:
    def __init__(self):
        from botocore.credentials import Credentials

        self.credentials = Credentials("AKIA-test", "secret-test")
        self.aws_region_name = "us-east-1"
        self.aws_bedrock_runtime_endpoint = None


def test_handler_builds_async_invoke_request(monkeypatch):
    handler = BedrockVideoGeneration()
    monkeypatch.setattr(
        BedrockVideoGeneration,
        "_get_boto_credentials_from_optional_params",
        lambda self, params, model=None, bearer_token=None: _FakeCredentialsInfo(),
    )
    endpoint_url, prepped, body, data = handler._prepare_async_invoke_request(
        model="bedrock/amazon.nova-reel-v1:0",
        prompt="waves at sunset",
        optional_params={"output_s3_uri": "s3://bucket/out/"},
        api_base=None,
        extra_headers=None,
        logging_obj=None,
    )
    assert endpoint_url == "https://bedrock-runtime.us-east-1.amazonaws.com/async-invoke"
    assert prepped.url == endpoint_url
    assert "Authorization" in prepped.headers
    parsed = json.loads(body)
    assert parsed["modelId"] == "amazon.nova-reel-v1:0"  # bedrock/ prefix stripped
    assert parsed["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"] == "s3://bucket/out/"
    assert parsed["modelInput"]["taskType"] == "TEXT_VIDEO"


def test_handler_builds_status_get_url(monkeypatch):
    handler = BedrockVideoGeneration()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-test")
    from litellm.types.videos.utils import encode_video_id_with_provider

    video_id = encode_video_id_with_provider(TEST_ARN, "bedrock", TEST_MODEL)
    status_url, prepped = handler._status_request_parts(
        TEST_ARN, {"aws_region_name": "us-east-1"}, None
    )
    from urllib.parse import quote

    expected = (
        "https://bedrock-runtime.us-east-1.amazonaws.com/async-invoke/"
        + quote(TEST_ARN, safe="")
    )
    assert status_url == expected
    assert prepped.url == expected


def test_handler_video_status_maps_response(monkeypatch):
    handler = BedrockVideoGeneration()
    from litellm.types.videos.utils import encode_video_id_with_provider

    video_id = encode_video_id_with_provider(TEST_ARN, "bedrock", TEST_MODEL)
    monkeypatch.setattr(
        handler,
        "_status_request_parts",
        lambda arn, params, api_base, api_key=None: (
            "https://example.com/async-invoke/arn",
            Mock(url="https://example.com/async-invoke/arn", headers={}),
        ),
    )
    resp = httpx.Response(
        200,
        json={
            "invocationArn": TEST_ARN,
            "status": "InProgress",
            "submitTime": 1758000000.0,
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}},
        },
    )
    monkeypatch.setattr(handler, "_sync_get", lambda prepped: resp)
    video = handler.video_status(video_id=video_id, litellm_params={"aws_region_name": "us-east-1"})
    assert video.status == "processing"
    assert decode_video_id_with_provider(video.id)["video_id"] == TEST_ARN


def test_handler_video_content_requires_completed(monkeypatch):
    handler = BedrockVideoGeneration()
    from litellm.types.videos.utils import encode_video_id_with_provider

    video_id = encode_video_id_with_provider(TEST_ARN, "bedrock", TEST_MODEL)
    monkeypatch.setattr(
        handler,
        "_status_request_parts",
        lambda arn, params, api_base, api_key=None: (
            "https://example.com/async-invoke/arn",
            Mock(url="https://example.com/async-invoke/arn", headers={}),
        ),
    )
    resp = httpx.Response(
        200,
        json={
            "invocationArn": TEST_ARN,
            "status": "InProgress",
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}},
        },
    )
    monkeypatch.setattr(handler, "_sync_get", lambda prepped: resp)
    with pytest.raises(ValueError, match="not complete"):
        handler.video_content(video_id=video_id, litellm_params={})


def test_handler_video_content_downloads_from_s3(monkeypatch):
    handler = BedrockVideoGeneration()
    from litellm.types.videos.utils import encode_video_id_with_provider

    video_id = encode_video_id_with_provider(TEST_ARN, "bedrock", TEST_MODEL)
    monkeypatch.setattr(
        handler,
        "_status_request_parts",
        lambda arn, params, api_base, api_key=None: (
            "https://example.com/async-invoke/arn",
            Mock(url="https://example.com/async-invoke/arn", headers={}),
        ),
    )
    resp = httpx.Response(
        200,
        json={
            "invocationArn": TEST_ARN,
            "status": "Completed",
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}},
        },
    )
    monkeypatch.setattr(handler, "_sync_get", lambda prepped: resp)

    downloaded_keys: list[str] = []

    def fake_download(bucket, key_candidates, litellm_params, raw):
        downloaded_keys.extend(key_candidates)
        return b"mp4-bytes"

    monkeypatch.setattr(handler, "_download_s3_object", fake_download)
    content = handler.video_content(video_id=video_id, litellm_params={})
    assert content == b"mp4-bytes"
    # v1:1 per-invocation folder first, then the older v1:0 flat layout.
    assert downloaded_keys[0] == "out/abc123-def456/output.mp4"
    assert downloaded_keys[1] == "out/output.mp4"


def test_get_supported_openai_params_includes_video_params():
    supported = _make_config().get_supported_openai_params(TEST_MODEL)
    assert "seconds" in supported
    assert "size" in supported
    assert "output_s3_uri" in supported
