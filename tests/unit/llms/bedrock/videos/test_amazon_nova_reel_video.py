"""Unit tests for Bedrock Amazon Nova Reel video generation (issue #39552)."""

import asyncio
import base64
import io
import json
from typing import cast
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.videos.handler import BedrockVideoGeneration
from litellm.llms.bedrock.videos.transformation import BedrockNovaReelVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams
from litellm.types.videos.utils import decode_video_id_with_provider

TEST_MODEL = "amazon.nova-reel-v1:0"
TEST_ARN = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123-def456"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 8
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"0" * 8


def _make_config() -> BedrockNovaReelVideoConfig:
    return BedrockNovaReelVideoConfig()


def _create_request(
    optional_params: dict | None = None,
    prompt: str = "A drone shot over the ocean",
    model: str = TEST_MODEL,
    litellm_params: GenericLiteLLMParams | None = None,
) -> dict:
    config = _make_config()
    params: dict = optional_params if optional_params is not None else {"output_s3_uri": "s3://bucket/out/"}
    body, files, method = config.transform_video_create_request(
        model=model,
        prompt=prompt,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/async-invoke",
        video_create_optional_request_params=cast(VideoCreateOptionalRequestParams, params),
        litellm_params=litellm_params or GenericLiteLLMParams(),
        headers={},
    )
    assert files == []
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
    body = _create_request({"output_s3_uri": "s3://bucket/out/", "seconds": "10", "size": "720x1280", "seed": 5})
    cfg = body["modelInput"]["videoGenerationConfig"]
    assert cfg["durationSeconds"] == 10
    assert cfg["dimension"] == "720x1280"
    assert cfg["seed"] == 5


def test_transform_create_request_input_reference_becomes_images():
    body = _create_request({"output_s3_uri": "s3://bucket/out/", "input_reference": io.BytesIO(PNG_BYTES)})
    images = body["modelInput"]["textToVideoParams"]["images"]
    assert len(images) == 1
    assert images[0]["format"] == "png"
    assert images[0]["source"]["bytes"] == base64.b64encode(PNG_BYTES).decode("utf-8")


def test_transform_create_request_jpeg_detected():
    body = _create_request({"output_s3_uri": "s3://bucket/out/", "input_reference": JPEG_BYTES})
    assert body["modelInput"]["textToVideoParams"]["images"][0]["format"] == "jpeg"


def test_transform_create_request_both_reference_keys_pop_both():
    """input_reference wins and no stray `image` key leaks into modelInput (F1)."""
    body = _create_request(
        {
            "output_s3_uri": "s3://bucket/out/",
            "input_reference": PNG_BYTES,
            "image": JPEG_BYTES,
        }
    )
    model_input = body["modelInput"]
    assert "image" not in model_input
    assert "input_reference" not in model_input
    images = model_input["textToVideoParams"]["images"]
    assert images[0]["format"] == "png"


def test_transform_create_request_file_like_image_not_leaked_with_input_reference():
    """A file-like `image` must be popped (not serialized) when input_reference is set (F1)."""
    body = _create_request(
        {
            "output_s3_uri": "s3://bucket/out/",
            "input_reference": PNG_BYTES,
            "image": io.BytesIO(PNG_BYTES),
        }
    )
    assert "image" not in body["modelInput"]
    assert body["modelInput"]["textToVideoParams"]["images"][0]["format"] == "png"


def test_input_reference_data_url_round_trips():
    """data:image/png;base64,<payload> decodes only the payload and sniffs the format (F2)."""
    data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("utf-8")
    body = _create_request({"output_s3_uri": "s3://bucket/out/", "input_reference": data_url})
    image = body["modelInput"]["textToVideoParams"]["images"][0]
    assert image["format"] == "png"
    assert image["source"]["bytes"] == base64.b64encode(PNG_BYTES).decode("utf-8")


def test_input_reference_undecodable_string_raises():
    with pytest.raises(ValueError, match="base64"):
        _create_request({"output_s3_uri": "s3://bucket/out/", "input_reference": "definitely!!not!!base64"})


def test_input_reference_unrecognized_magic_raises():
    mystery_b64 = base64.b64encode(b"neither a png nor a jpeg header").decode("utf-8")
    with pytest.raises(ValueError, match="PNG or JPEG"):
        _create_request({"output_s3_uri": "s3://bucket/out/", "input_reference": mystery_b64})


def test_input_reference_https_url_raises():
    with pytest.raises(ValueError, match="base64"):
        _create_request({"output_s3_uri": "s3://bucket/out/", "input_reference": "https://example.com/img.png"})


def test_transform_create_request_drops_non_aws_client_params():
    """Known non-AWS video-client params are dropped; provider keys still pass through (F3)."""
    body = _create_request(
        {
            "output_s3_uri": "s3://bucket/out/",
            "parameters": {"foo": 1},
            "resolution": "1080p",
            "characters": [],
            "user": "someone",
            "multiShotManualParams": {"shot": []},
        }
    )
    model_input = body["modelInput"]
    for leaked in ("parameters", "resolution", "characters", "user"):
        assert leaked not in model_input
    assert model_input["multiShotManualParams"] == {"shot": []}


def test_client_request_token_passthrough_populates_envelope():
    body = _create_request({"output_s3_uri": "s3://bucket/out/", "client_request_token": "my-token-123"})
    assert body["clientRequestToken"] == "my-token-123"
    assert "client_request_token" not in body["modelInput"]


def test_client_request_token_absent_omits_envelope_key():
    body = _create_request()
    assert "clientRequestToken" not in body


def test_client_request_token_falls_back_to_litellm_request_id():
    litellm_params = GenericLiteLLMParams()
    litellm_params.metadata = {"request_id": "req/abc_123"}  # extra field allowed on GenericLiteLLMParams
    body = _create_request(litellm_params=litellm_params)
    assert body["clientRequestToken"] == "req-abc-123"


def test_client_request_token_sanitized_and_truncated():
    litellm_params = GenericLiteLLMParams()
    litellm_params.metadata = {"request_id": "req/abc_123:" + "x" * 100}  # extra field allowed on GenericLiteLLMParams
    body = _create_request(litellm_params=litellm_params)
    token = body["clientRequestToken"]
    assert len(token) == 64
    assert "/" not in token and ":" not in token and "_" not in token


def test_fps_seed_string_coercion():
    body = _create_request({"output_s3_uri": "s3://bucket/out/", "fps": "24", "seed": "42"})
    cfg = body["modelInput"]["videoGenerationConfig"]
    assert cfg["fps"] == 24
    assert cfg["seed"] == 42


def test_fps_non_numeric_raises():
    with pytest.raises(ValueError, match="fps"):
        _create_request({"output_s3_uri": "s3://bucket/out/", "fps": "abc"})


def test_seed_non_numeric_raises():
    with pytest.raises(ValueError, match="seed"):
        _create_request({"output_s3_uri": "s3://bucket/out/", "seed": "abc"})


def test_transform_create_request_requires_output_s3_uri():
    with pytest.raises(ValueError, match="output_s3_uri"):
        _create_request({})


def test_transform_create_request_task_type_passthrough():
    body = _create_request({"output_s3_uri": "s3://bucket/out/", "taskType": "MULTI_SHOT_AUTOMATED"})
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
        request_data={"modelInput": {"videoGenerationConfig": {"durationSeconds": 6}}},
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
        config.transform_video_create_response(model=TEST_MODEL, raw_response=resp, logging_obj=None)


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
    video = config.transform_video_status_retrieve_response(raw_response=resp, logging_obj=None, model=TEST_MODEL)
    assert video.status == expected
    if raw_status == "Failed":
        assert video.error == {"message": "blocked by content filters"}
    if raw_status == "Completed":
        assert video.completed_at == 1758000060
        assert video.usage is not None
        assert video.usage["output_s3_uri"] == "s3://bucket/out/"
    decoded = decode_video_id_with_provider(video.id)
    assert decoded["video_id"] == TEST_ARN


def test_transform_status_response_missing_status_raises():
    """A get-async-invoke body without a status must fail loudly, not report InProgress (F6)."""
    config = _make_config()
    resp = httpx.Response(200, json={"invocationArn": TEST_ARN, "submitTime": 1758000000.0})
    with pytest.raises(BedrockError, match="unexpected shape") as excinfo:
        config.transform_video_status_retrieve_response(raw_response=resp, logging_obj=None, model=TEST_MODEL)
    message = str(excinfo.value.message)
    assert "invocationArn" in message  # observed keys are named
    assert "submitTime" in message


def test_transform_status_response_empty_status_raises():
    config = _make_config()
    resp = httpx.Response(200, json={"invocationArn": TEST_ARN, "status": ""})
    with pytest.raises(BedrockError, match="unexpected shape"):
        config.transform_video_status_retrieve_response(raw_response=resp, logging_obj=None, model=TEST_MODEL)


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

    cfg = ProviderConfigManager.get_provider_video_config(TEST_MODEL, litellm.LlmProviders.BEDROCK)
    assert isinstance(cfg, BedrockNovaReelVideoConfig)


def test_provider_config_manager_us_cross_region_variant():
    from litellm.utils import ProviderConfigManager

    cfg = ProviderConfigManager.get_provider_video_config("us.amazon.nova-reel-v1:0", litellm.LlmProviders.BEDROCK)
    assert isinstance(cfg, BedrockNovaReelVideoConfig)


def test_provider_config_manager_non_reel_bedrock_returns_none():
    from litellm.utils import ProviderConfigManager

    cfg = ProviderConfigManager.get_provider_video_config("amazon.titan-text-express-v1", litellm.LlmProviders.BEDROCK)
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
    status_url, prepped, region = handler._status_request_parts(TEST_ARN, {"aws_region_name": "us-east-1"}, None)
    from urllib.parse import quote

    expected = "https://bedrock-runtime.us-east-1.amazonaws.com/async-invoke/" + quote(TEST_ARN, safe="")
    assert status_url == expected
    assert prepped.url == expected
    assert region == "us-east-1"


def test_handler_status_region_defaults_to_arn_region(monkeypatch):
    """No explicit aws_region_name: the ARN region must be resolved and returned (F8)."""
    handler = BedrockVideoGeneration()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-test")
    _, _, region = handler._status_request_parts(TEST_ARN, {}, None)
    assert region == "us-east-1"


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
            "us-east-1",
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
    monkeypatch.setattr(handler, "_sync_get", lambda prepped, timeout=None: resp)
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
            "us-east-1",
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
    monkeypatch.setattr(handler, "_sync_get", lambda prepped, timeout=None: resp)
    with pytest.raises(ValueError, match="not complete"):
        handler.video_content(video_id=video_id, litellm_params={})


def test_handler_video_content_failed_status_raises_with_failure_message(monkeypatch):
    """A Failed invocation surfaces failureMessage as a 502, not a not-complete-yet error (F7)."""
    handler = BedrockVideoGeneration()
    from litellm.types.videos.utils import encode_video_id_with_provider

    video_id = encode_video_id_with_provider(TEST_ARN, "bedrock", TEST_MODEL)
    monkeypatch.setattr(
        handler,
        "_status_request_parts",
        lambda arn, params, api_base, api_key=None: (
            "https://example.com/async-invoke/arn",
            Mock(url="https://example.com/async-invoke/arn", headers={}),
            "us-east-1",
        ),
    )
    resp = httpx.Response(
        200,
        json={
            "invocationArn": TEST_ARN,
            "status": "Failed",
            "failureMessage": "Content filtration invoked",
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}},
        },
    )
    monkeypatch.setattr(handler, "_sync_get", lambda prepped, timeout=None: resp)
    with pytest.raises(BedrockError, match="Nova Reel invocation failed: Content filtration invoked") as excinfo:
        handler.video_content(video_id=video_id, litellm_params={})
    assert excinfo.value.status_code == 502


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
            "us-east-1",
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
    monkeypatch.setattr(handler, "_sync_get", lambda prepped, timeout=None: resp)

    downloaded_keys: list[str] = []

    def fake_download(bucket, key_candidates, litellm_params, raw, region_default=None):
        downloaded_keys.extend(key_candidates)
        return b"mp4-bytes"

    monkeypatch.setattr(handler, "_download_s3_object", fake_download)
    content = handler.video_content(video_id=video_id, litellm_params={})
    assert content == b"mp4-bytes"
    # v1:1 per-invocation folder first, then the older v1:0 flat layout.
    assert downloaded_keys[0] == "out/abc123-def456/output.mp4"
    assert downloaded_keys[1] == "out/output.mp4"


def test_handler_sync_create_passes_timeout(monkeypatch):
    """The sync create path must forward its timeout to the POST (F9)."""
    handler = BedrockVideoGeneration()
    monkeypatch.setattr(
        BedrockVideoGeneration,
        "_get_boto_credentials_from_optional_params",
        lambda self, params, model=None, bearer_token=None: _FakeCredentialsInfo(),
    )
    seen: dict[str, object] = {}

    class _RecordingClient:
        def post(self, **kwargs):
            seen.update(kwargs)
            return httpx.Response(
                200,
                json={"invocationArn": TEST_ARN},
                request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/async-invoke"),
            )

    monkeypatch.setattr("litellm.llms.custom_httpx.http_handler._get_httpx_client", lambda: _RecordingClient())
    video = handler.video_generation(
        model="bedrock/amazon.nova-reel-v1:0",
        prompt="waves at sunset",
        optional_params={"output_s3_uri": "s3://bucket/out/"},
        logging_obj=None,
        timeout=12.5,
        avideo_generation=False,
    )
    assert seen["timeout"] == 12.5
    assert video.status == "processing"


def test_handler_sync_get_passes_timeout(monkeypatch):
    """video_status must thread its timeout into the GET request (F9)."""
    handler = BedrockVideoGeneration()
    from litellm.types.videos.utils import encode_video_id_with_provider

    video_id = encode_video_id_with_provider(TEST_ARN, "bedrock", TEST_MODEL)
    monkeypatch.setattr(
        handler,
        "_status_request_parts",
        lambda arn, params, api_base, api_key=None: (
            "https://example.com/async-invoke/arn",
            Mock(url="https://example.com/async-invoke/arn", headers={}),
            "us-east-1",
        ),
    )
    seen: dict[str, object] = {}

    class _RecordingClient:
        def get(self, **kwargs):
            seen.update(kwargs)
            return httpx.Response(
                200,
                json={"invocationArn": TEST_ARN, "status": "InProgress", "submitTime": 1758000000.0},
            )

    monkeypatch.setattr("litellm.llms.custom_httpx.http_handler._get_httpx_client", lambda: _RecordingClient())
    video = handler.video_status(video_id=video_id, litellm_params={}, timeout=7.5)
    assert seen["timeout"] == 7.5
    assert video.status == "processing"


def test_handler_async_get_passes_timeout(monkeypatch):
    """_async_get must thread its timeout into the GET request (F9)."""
    handler = BedrockVideoGeneration()
    seen: dict[str, object] = {}

    class _RecordingAsyncClient:
        async def get(self, **kwargs):
            seen.update(kwargs)
            return httpx.Response(200, json={"invocationArn": TEST_ARN, "status": "InProgress"})

    monkeypatch.setattr(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        lambda llm_provider=None, params=None: _RecordingAsyncClient(),
    )
    prepped = Mock(url="https://example.com/async-invoke/arn", headers={})
    response = asyncio.run(handler._async_get(prepped, timeout=9.0))
    assert seen["timeout"] == 9.0
    assert response.status_code == 200


def test_sign_get_request_without_credentials_or_bearer_raises(monkeypatch):
    """No SigV4 credentials and no bearer token must fail fast (F11), like the shared POST signer."""
    from botocore.exceptions import NoCredentialsError

    from litellm.llms.bedrock.videos.handler import _sign_get_request

    with pytest.raises(NoCredentialsError):
        _sign_get_request(
            credentials=None,
            url="https://bedrock-runtime.us-east-1.amazonaws.com/async-invoke/arn",
            headers={},
            aws_region_name="us-east-1",
        )


def _patch_s3_download(monkeypatch, handler: BedrockVideoGeneration, get_object_side_effect) -> list[dict]:
    """Mock boto3 + credentials for _download_s3_object; returns the Session kwargs it saw."""
    sessions: list[dict] = []

    class _FakeS3Client:
        def get_object(self, Bucket, Key):
            return get_object_side_effect(Bucket, Key)

    class _FakeSession:
        def __init__(self, **kwargs):
            sessions.append(kwargs)

        def client(self, service_name):
            return _FakeS3Client()

    monkeypatch.setattr("boto3.Session", _FakeSession)
    monkeypatch.setattr(
        handler,
        "_load_credentials",
        lambda optional_params, aws_region_name=None, bearer_token=None: (None, aws_region_name or "us-east-1"),
    )
    return sessions


def test_download_s3_object_uses_status_region_by_default(monkeypatch):
    """region_default (ARN-derived) beats env/default; explicit litellm_params region still wins (F8)."""
    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}
    sessions = _patch_s3_download(
        monkeypatch,
        handler,
        lambda bucket, key: {"Body": io.BytesIO(b"mp4-bytes")},
    )
    content = handler._download_s3_object("bucket", ["out/output.mp4"], {}, raw, region_default="eu-central-1")
    assert content == b"mp4-bytes"
    assert sessions[0]["region_name"] == "eu-central-1"

    handler._download_s3_object(
        "bucket",
        ["out/output.mp4"],
        {"aws_region_name": "ap-south-1"},
        raw,
        region_default="eu-central-1",
    )
    assert sessions[1]["region_name"] == "ap-south-1"


def test_download_s3_object_error_message_redacts_invocation(monkeypatch):
    """The 404 names the s3Uri and tried keys, never the raw invocation (ARN/account id) (F12)."""
    from botocore.exceptions import ClientError

    def _raise(bucket, key):
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
            "GetObject",
        )

    handler = BedrockVideoGeneration()
    raw: dict = {
        "invocationArn": TEST_ARN,
        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}},
    }
    _patch_s3_download(monkeypatch, handler, _raise)
    with pytest.raises(BedrockError, match="not found in the S3 output location") as excinfo:
        handler._download_s3_object("bucket", ["out/output.mp4"], {}, raw, region_default="us-east-1")
    message = str(excinfo.value.message)
    assert "s3://bucket/out/" in message
    assert "out/output.mp4" in message
    assert "NoSuchKey" in message
    assert "123456789012" not in message
    assert "invocationArn" not in message


def test_transform_create_response_accepts_202(monkeypatch):
    """Any 2xx is a success once raise_for_status ran; 202 must reach the transform (F10)."""
    handler = BedrockVideoGeneration()
    resp = httpx.Response(202, json={"invocationArn": TEST_ARN})
    video = handler._transform_create_response(TEST_MODEL, resp, {}, None)
    assert video.status == "processing"


def test_get_supported_openai_params_includes_video_params():
    supported = _make_config().get_supported_openai_params(TEST_MODEL)
    assert "seconds" in supported
    assert "size" in supported
    assert "output_s3_uri" in supported
    assert "parameters" not in supported


#################################################
# litellm_params threading into the create request
#################################################


def test_handler_create_threads_litellm_params_request_id_into_token(monkeypatch):
    """video_generation must feed litellm_params into the transform so
    metadata.request_id reaches the signed POST body as clientRequestToken."""
    handler = BedrockVideoGeneration()
    monkeypatch.setattr(
        BedrockVideoGeneration,
        "_get_boto_credentials_from_optional_params",
        lambda self, params, model=None, bearer_token=None: _FakeCredentialsInfo(),
    )
    bodies: list[bytes] = []

    class _RecordingClient:
        def post(self, **kwargs):
            bodies.append(kwargs["content"])
            return httpx.Response(
                200,
                json={"invocationArn": TEST_ARN},
                request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/async-invoke"),
            )

    monkeypatch.setattr("litellm.llms.custom_httpx.http_handler._get_httpx_client", lambda: _RecordingClient())
    litellm_params = GenericLiteLLMParams()
    litellm_params.metadata = {"request_id": "req/abc_123"}  # extra field allowed on GenericLiteLLMParams
    video = handler.video_generation(
        model="bedrock/amazon.nova-reel-v1:0",
        prompt="waves at sunset",
        optional_params={"output_s3_uri": "s3://bucket/out/"},
        logging_obj=None,
        timeout=5.0,
        avideo_generation=False,
        litellm_params=litellm_params,
    )
    assert video.status == "processing"
    parsed = json.loads(bodies[0])
    assert parsed["clientRequestToken"] == "req-abc-123"
