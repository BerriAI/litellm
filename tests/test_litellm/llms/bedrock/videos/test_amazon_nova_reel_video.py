"""Unit tests for Bedrock Amazon Nova Reel video generation (issue #39552)."""

import asyncio
import base64
import io
import json
from datetime import datetime, timezone
from typing import Final, cast
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
    """input_reference wins and no stray `image` key leaks into modelInput."""
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
    """A file-like `image` must be popped (not serialized) when input_reference is set."""
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
    """data:image/png;base64,<payload> decodes only the payload and sniffs the format."""
    data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("utf-8")
    body = _create_request({"output_s3_uri": "s3://bucket/out/", "input_reference": data_url})
    image = body["modelInput"]["textToVideoParams"]["images"][0]
    assert image["format"] == "png"
    assert image["source"]["bytes"] == base64.b64encode(PNG_BYTES).decode("utf-8")


def test_input_reference_undecodable_string_raises():
    with pytest.raises(BedrockError) as excinfo:
        _create_request({"output_s3_uri": "s3://bucket/out/", "input_reference": "definitely!!not!!base64"})
    assert excinfo.value.status_code == 400
    assert "base64" in str(excinfo.value.message)


def test_input_reference_unrecognized_magic_raises():
    mystery_b64 = base64.b64encode(b"neither a png nor a jpeg header").decode("utf-8")
    with pytest.raises(BedrockError) as excinfo:
        _create_request({"output_s3_uri": "s3://bucket/out/", "input_reference": mystery_b64})
    assert excinfo.value.status_code == 400
    assert "PNG or JPEG" in str(excinfo.value.message)


def test_input_reference_https_url_raises():
    with pytest.raises(BedrockError) as excinfo:
        _create_request({"output_s3_uri": "s3://bucket/out/", "input_reference": "https://example.com/img.png"})
    assert excinfo.value.status_code == 400
    assert "base64" in str(excinfo.value.message)


def test_transform_create_request_drops_non_aws_client_params():
    """Known non-AWS video-client params are dropped; provider keys still pass through."""
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


def test_seconds_string_coerces():
    body = _create_request({"output_s3_uri": "s3://bucket/out/", "seconds": "8"})
    assert body["modelInput"]["videoGenerationConfig"]["durationSeconds"] == 8


def test_seconds_non_numeric_raises():
    """Non-numeric seconds must raise like fps/seed, not silently keep the default 6."""
    with pytest.raises(BedrockError) as excinfo:
        _create_request({"output_s3_uri": "s3://bucket/out/", "seconds": "abc"})
    assert excinfo.value.status_code == 400
    assert "seconds" in str(excinfo.value.message)


def test_fps_non_numeric_raises():
    with pytest.raises(BedrockError) as excinfo:
        _create_request({"output_s3_uri": "s3://bucket/out/", "fps": "abc"})
    assert excinfo.value.status_code == 400
    assert "fps" in str(excinfo.value.message)


def test_seed_non_numeric_raises():
    with pytest.raises(BedrockError) as excinfo:
        _create_request({"output_s3_uri": "s3://bucket/out/", "seed": "abc"})
    assert excinfo.value.status_code == 400
    assert "seed" in str(excinfo.value.message)


def test_transform_create_request_requires_output_s3_uri():
    with pytest.raises(BedrockError) as excinfo:
        _create_request({})
    assert excinfo.value.status_code == 400
    assert "output_s3_uri" in str(excinfo.value.message)


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
        # output_s3_uri rides on _hidden_params (provider detail), not usage.
        assert video.usage is None
        assert video._hidden_params["output_s3_uri"] == "s3://bucket/out/"
    decoded = decode_video_id_with_provider(video.id)
    assert decoded["video_id"] == TEST_ARN


def test_transform_status_response_missing_status_raises():
    """A get-async-invoke body without a status must fail loudly, not report InProgress."""
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


def test_transform_status_response_unknown_status_warns_and_maps_processing(monkeypatch):
    """An unmapped AWS invocationStatus must log a warning while still reporting processing."""
    config = _make_config()
    logger = Mock()
    monkeypatch.setattr("litellm.llms.bedrock.videos.transformation.verbose_logger", logger)
    resp = httpx.Response(
        200,
        json={
            "invocationArn": TEST_ARN,
            "status": "Throttled",
            "submitTime": 1758000000.0,
        },
    )
    video = config.transform_video_status_retrieve_response(raw_response=resp, logging_obj=None, model=TEST_MODEL)
    assert video.status == "processing"
    logger.warning.assert_called_once()
    assert "Throttled" in str(logger.warning.call_args)


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


#################################################
# logging headers redaction (pre_call additional_args)
#################################################


def test_redact_bedrock_headers_for_logging_masks_signed_headers():
    """SigV4 signature material must be replaced with [REDACTED]; safe headers survive."""
    from litellm.llms.bedrock.common_utils import redact_bedrock_headers_for_logging

    signed: Final[dict[str, str]] = {
        "Content-Type": "application/json",
        "Host": "bedrock-runtime.us-east-1.amazonaws.com",
        "Authorization": (
            "AWS4-HMAC-SHA256 Credential=AKIA-test/20260115/us-east-1/bedrock/aws4_request, "
            "SignedHeaders=host;x-amz-date, Signature=deadbeefsecret"
        ),
        "X-Amz-Date": "20260115T103000Z",
        "X-Amz-Security-Token": "session-token-secret",
        "X-Amz-Content-Sha256": "sensitive-payload-hash",
    }
    redacted = redact_bedrock_headers_for_logging(signed)
    assert redacted["Content-Type"] == "application/json"
    assert redacted["Host"] == "bedrock-runtime.us-east-1.amazonaws.com"
    assert redacted["Authorization"] == "[REDACTED]"
    assert redacted["X-Amz-Date"] == "[REDACTED]"
    assert redacted["X-Amz-Security-Token"] == "[REDACTED]"
    assert redacted["X-Amz-Content-Sha256"] == "[REDACTED]"
    # Every key stays present (log consumers see the full header shape).
    assert set(redacted.keys()) == set(signed.keys())
    # The input mapping is untouched: redaction never mutates the sent headers.
    assert signed["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert signed["X-Amz-Security-Token"] == "session-token-secret"


def _capturing_logging_obj(captured: dict):
    """A real Logging object whose logger_fn captures model_call_details."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    def _capture(model_call_details):
        captured.update(model_call_details)

    logging_obj = Logging(
        model="bedrock/amazon.nova-reel-v1:0",
        messages=[],
        stream=False,
        call_type="avideo_generation",
        start_time=datetime.now(),
        litellm_call_id="test-call-id",
        function_id="test-function",
        kwargs={"logger_fn": _capture},
    )
    logging_obj.update_environment_variables(
        litellm_params={"logger_fn": _capture},
        optional_params={},
    )
    return logging_obj


def test_prepare_request_logging_headers_redacted(monkeypatch):
    """pre_call additional_args must carry the redacted copy; the sent request keeps
    the real signed Authorization header."""
    handler = BedrockVideoGeneration()
    monkeypatch.setattr(
        BedrockVideoGeneration,
        "_get_boto_credentials_from_optional_params",
        lambda self, params, model=None, bearer_token=None: _FakeCredentialsInfo(),
    )
    captured: dict = {}
    _, prepped, _, _ = handler._prepare_async_invoke_request(
        model="bedrock/amazon.nova-reel-v1:0",
        prompt="waves at sunset",
        optional_params={"output_s3_uri": "s3://bucket/out/"},
        api_base=None,
        extra_headers=None,
        logging_obj=_capturing_logging_obj(captured),
    )
    logged_headers: Final = captured["additional_args"]["headers"]
    assert logged_headers["Authorization"] == "[REDACTED]"
    assert logged_headers["Content-Type"] == "application/json"
    assert "deadbeefsecret" not in str(captured)
    # The sent request still carries the real SigV4 Authorization header.
    assert prepped.headers["Authorization"].startswith("AWS4-HMAC-SHA256")


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
    """No explicit aws_region_name: the ARN region must be resolved and returned."""
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
    with pytest.raises(BedrockError) as excinfo:
        handler.video_content(video_id=video_id, litellm_params={})
    assert excinfo.value.status_code == 400
    assert "not complete" in str(excinfo.value.message)


def test_handler_video_content_failed_status_raises_with_failure_message(monkeypatch):
    """A Failed invocation surfaces failureMessage as a 502, not a not-complete-yet error."""
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

    # Pin a v1:0 model explicitly: the flat fallback key is v1:0-only.
    assert TEST_MODEL == "amazon.nova-reel-v1:0"
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

    def fake_download(bucket, key_candidates, litellm_params, raw, region_default=None, api_key=None, timeout=None):
        downloaded_keys.extend(key_candidates)
        return b"mp4-bytes"

    monkeypatch.setattr(handler, "_download_s3_object", fake_download)
    content = handler.video_content(video_id=video_id, litellm_params={})
    assert content == b"mp4-bytes"
    # v1:1 per-invocation folder first, then the older v1:0 flat layout.
    assert downloaded_keys[0] == "out/abc123-def456/output.mp4"
    assert downloaded_keys[1] == "out/output.mp4"


def _completed_content_setup(
    monkeypatch,
    handler: BedrockVideoGeneration,
    encoded_model: str,
    extra_status: dict,
) -> tuple[str, list[str]]:
    """Patch status+download for a Completed video_content run; returns (video_id, keys)."""
    from litellm.types.videos.utils import encode_video_id_with_provider

    video_id = encode_video_id_with_provider(TEST_ARN, "bedrock", encoded_model)
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
            **extra_status,
        },
    )
    monkeypatch.setattr(handler, "_sync_get", lambda prepped, timeout=None: resp)

    downloaded_keys: list[str] = []

    def fake_download(bucket, key_candidates, litellm_params, raw, region_default=None, api_key=None, timeout=None):
        downloaded_keys.extend(key_candidates)
        return b"mp4-bytes"

    monkeypatch.setattr(handler, "_download_s3_object", fake_download)
    return video_id, downloaded_keys


def test_video_content_v1_1_does_not_fall_back_to_flat_key(monkeypatch):
    """v1:1 writes per-invocation folders only; the shared-prefix flat key must not
    be tried (it can hold a foreign or stale object)."""
    handler = BedrockVideoGeneration()
    video_id, keys = _completed_content_setup(
        monkeypatch,
        handler,
        "amazon.nova-reel-v1:1",
        {"modelArn": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-reel-v1:1"},
    )
    assert handler.video_content(video_id=video_id, litellm_params={}) == b"mp4-bytes"
    assert keys == ["out/abc123-def456/output.mp4"]


def test_video_content_v1_0_keeps_flat_fallback(monkeypatch):
    """A v1:0 invocation (cross-region id + foundation-model arn) keeps both candidate
    keys, per-invocation folder first."""
    handler = BedrockVideoGeneration()
    video_id, keys = _completed_content_setup(
        monkeypatch,
        handler,
        "us.amazon.nova-reel-v1:0",
        {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-reel-v1:0"},
    )
    assert handler.video_content(video_id=video_id, litellm_params={}) == b"mp4-bytes"
    assert keys == ["out/abc123-def456/output.mp4", "out/output.mp4"]


def test_video_content_model_arn_overrides_encoded_model(monkeypatch):
    """modelArn on the status response is authoritative: a v1:0-encoded id whose
    invocation actually ran v1:1 (per the arn) drops the flat fallback."""
    handler = BedrockVideoGeneration()
    video_id, keys = _completed_content_setup(
        monkeypatch,
        handler,
        "amazon.nova-reel-v1:0",
        {"modelArn": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-reel-v1:1"},
    )
    assert handler.video_content(video_id=video_id, litellm_params={}) == b"mp4-bytes"
    assert keys == ["out/abc123-def456/output.mp4"]


def test_handler_sync_create_passes_timeout(monkeypatch):
    """The sync create path must forward its timeout to the POST."""
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
    """video_status must thread its timeout into the GET request."""
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
    """_async_get must thread its timeout into the GET request."""
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
    """No SigV4 credentials and no bearer token must fail fast, like the shared POST signer."""
    from botocore.exceptions import NoCredentialsError

    from litellm.llms.bedrock.videos.handler import _sign_get_request

    with pytest.raises(NoCredentialsError):
        _sign_get_request(
            credentials=None,
            url="https://bedrock-runtime.us-east-1.amazonaws.com/async-invoke/arn",
            headers={},
            aws_region_name="us-east-1",
        )


def _patch_s3_download(
    monkeypatch, handler: BedrockVideoGeneration, get_object_side_effect
) -> tuple[list, list, list[str]]:
    """Mock boto3 + credentials for _download_s3_object.

    Returns (session_kwargs, s3_clients, attempted_keys); each fake client records
    the botocore Config it was built with and whether close() ran.
    """
    sessions: list[dict] = []
    clients: list = []
    attempted: list[str] = []

    class _FakeS3Client:
        def __init__(self):
            self.config = None
            self.closed = False

        def get_object(self, Bucket, Key):
            attempted.append(Key)
            return get_object_side_effect(Bucket, Key)

        def close(self):
            self.closed = True

    class _FakeSession:
        def __init__(self, **kwargs):
            sessions.append(kwargs)

        def client(self, service_name, config=None):
            client: Final = _FakeS3Client()
            client.config = config
            clients.append(client)
            return client

    monkeypatch.setattr("boto3.Session", _FakeSession)
    monkeypatch.setattr(
        handler,
        "_load_credentials",
        lambda optional_params, aws_region_name=None, bearer_token=None: (None, aws_region_name or "us-east-1"),
    )
    return sessions, clients, attempted


class _TrackingBody(io.BytesIO):
    """BytesIO that counts close() calls so the download's cleanup is observable."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        super().close()


def test_download_s3_object_uses_status_region_by_default(monkeypatch):
    """region_default (ARN-derived) beats env/default; explicit litellm_params region still wins."""
    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}
    sessions, _, _ = _patch_s3_download(
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
    """The 404 names the s3Uri and tried keys, never the raw invocation (ARN/account id)."""
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
    message: Final = str(excinfo.value.message)
    assert "s3://bucket/out/" in message
    assert "out/output.mp4" in message
    assert "NoSuchKey" in message
    assert "123456789012" not in message
    assert "invocationArn" not in message


def test_transform_create_response_accepts_202(monkeypatch):
    """Any 2xx is a success once raise_for_status ran; 202 must reach the transform."""
    handler = BedrockVideoGeneration()
    resp = httpx.Response(202, json={"invocationArn": TEST_ARN})
    video = handler._transform_create_response(TEST_MODEL, resp, {}, None)
    assert video.status == "processing"


def test_get_supported_openai_params_includes_video_params():
    supported = _make_config().get_supported_openai_params(TEST_MODEL)
    assert "seconds" in supported
    assert "size" in supported
    assert "output_s3_uri" in supported
    assert "kmsKeyId" in supported
    assert "bucketOwner" in supported
    assert "parameters" not in supported


#################################################
# litellm_params threading + timeout/error mapping on the HTTP paths
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


def test_handler_async_injected_client_receives_timeout_on_post(monkeypatch):
    """An injected async client must receive the timeout kwarg on post (the shared
    factory path applied it; the injected path dropped it)."""
    handler = BedrockVideoGeneration()
    monkeypatch.setattr(
        BedrockVideoGeneration,
        "_get_boto_credentials_from_optional_params",
        lambda self, params, model=None, bearer_token=None: _FakeCredentialsInfo(),
    )
    seen: dict[str, object] = {}

    class _RecordingAsyncClient(httpx.AsyncClient):
        async def post(self, **kwargs):
            seen.update(kwargs)
            return httpx.Response(
                200,
                json={"invocationArn": TEST_ARN},
                request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/async-invoke"),
            )

    video = asyncio.run(
        handler.async_video_generation(
            model="bedrock/amazon.nova-reel-v1:0",
            prompt="waves at sunset",
            optional_params={"output_s3_uri": "s3://bucket/out/"},
            logging_obj=None,
            timeout=4.5,
            client=_RecordingAsyncClient(),
        )
    )
    assert seen["timeout"] == 4.5
    assert video.status == "processing"


def test_handler_sync_get_timeout_maps_to_bedrock_408(monkeypatch):
    """A status GET timeout must surface as BedrockError 408 like the create path."""
    handler = BedrockVideoGeneration()

    class _TimingOutClient:
        def get(self, **kwargs):
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr("litellm.llms.custom_httpx.http_handler._get_httpx_client", lambda: _TimingOutClient())
    with pytest.raises(BedrockError) as excinfo:
        handler._sync_get(Mock(url="https://example.com/async-invoke/arn", headers={}), timeout=1.0)
    assert excinfo.value.status_code == 408


def test_handler_async_get_timeout_maps_to_bedrock_408(monkeypatch):
    handler = BedrockVideoGeneration()

    class _TimingOutAsyncClient:
        async def get(self, **kwargs):
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        lambda llm_provider=None, params=None: _TimingOutAsyncClient(),
    )
    with pytest.raises(BedrockError) as excinfo:
        asyncio.run(handler._async_get(Mock(url="https://example.com/async-invoke/arn", headers={}), timeout=1.0))
    assert excinfo.value.status_code == 408


def test_handler_video_content_missing_s3_uri_redacts_invocation(monkeypatch):
    """The no-S3-output ValueError must not leak invocationArn (account id); the raw
    payload goes to debug logs and the raise names only the observed key names."""
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
            "outputDataConfig": {"s3OutputDataConfig": {}},
        },
    )
    monkeypatch.setattr(handler, "_sync_get", lambda prepped, timeout=None: resp)
    with pytest.raises(ValueError, match="No S3 output location") as excinfo:
        handler.video_content(video_id=video_id, litellm_params={})
    message = str(excinfo.value)
    assert "123456789012" not in message
    assert "invocationArn" not in message
    assert "s3OutputDataConfig" in message  # observed key names are still reported


def test_download_s3_object_no_credentials_maps_to_502(monkeypatch):
    """NoCredentialsError (a BotoCoreError) must map to BedrockError 502, not escape raw."""
    from botocore.exceptions import NoCredentialsError

    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}

    def _raise(bucket, key):
        raise NoCredentialsError()

    _patch_s3_download(monkeypatch, handler, _raise)
    with pytest.raises(BedrockError) as excinfo:
        handler._download_s3_object("bucket", ["out/output.mp4"], {}, raw, region_default="us-east-1")
    assert excinfo.value.status_code == 502
    assert "Failed to download Nova Reel output from S3" in str(excinfo.value.message)
    assert "NoCredentialsError" in str(excinfo.value.message)


def test_download_s3_object_endpoint_connection_error_maps_to_502(monkeypatch):
    from botocore.exceptions import EndpointConnectionError

    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}

    def _raise(bucket, key):
        raise EndpointConnectionError(endpoint_url="https://s3.us-east-1.amazonaws.com/bucket/out/output.mp4")

    _patch_s3_download(monkeypatch, handler, _raise)
    with pytest.raises(BedrockError) as excinfo:
        handler._download_s3_object("bucket", ["out/output.mp4"], {}, raw, region_default="us-east-1")
    assert excinfo.value.status_code == 502
    assert "EndpointConnectionError" in str(excinfo.value.message)


def test_download_s3_object_falls_back_to_second_candidate_key(monkeypatch):
    """A ClientError on the first candidate key falls through to the second."""
    from botocore.exceptions import ClientError

    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}
    attempted: list[str] = []

    def _first_key_missing(bucket, key):
        if key == "out/abc123-def456/output.mp4":
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
                "GetObject",
            )
        return {"Body": io.BytesIO(b"mp4-from-second-key")}

    _, clients, attempted = _patch_s3_download(monkeypatch, handler, _first_key_missing)
    content = handler._download_s3_object(
        "bucket",
        ["out/abc123-def456/output.mp4", "out/output.mp4"],
        {},
        raw,
        region_default="us-east-1",
    )
    assert content == b"mp4-from-second-key"
    assert attempted == ["out/abc123-def456/output.mp4", "out/output.mp4"]


def test_main_layer_bedrock_status_and_content_pass_default_timeout(monkeypatch):
    """The bedrock branches of litellm.video_status / litellm.video_content must thread
    a real timeout into the handler (never None), like the create branch already does."""
    from litellm.llms.bedrock.videos.handler import BedrockVideoGeneration as _Handler
    from litellm.types.videos.utils import encode_video_id_with_provider
    from litellm.videos import main as videos_main

    status_kwargs: dict = {}
    content_kwargs: dict = {}

    def fake_video_status(self, **kwargs):
        status_kwargs.update(kwargs)
        return Mock()

    def fake_video_content(self, **kwargs):
        content_kwargs.update(kwargs)
        return b"mp4-bytes"

    monkeypatch.setattr(_Handler, "video_status", fake_video_status)
    monkeypatch.setattr(_Handler, "video_content", fake_video_content)

    video_id = encode_video_id_with_provider(TEST_ARN, "bedrock", TEST_MODEL)
    status_result = videos_main.video_status(video_id=video_id, custom_llm_provider="bedrock")
    content_result = videos_main.video_content(video_id=video_id, custom_llm_provider="bedrock")

    assert status_result is not None
    assert content_result == b"mp4-bytes"
    # video_status signature default (600s) flows through as-is; video_content's
    # None default is replaced by the layer DEFAULT_REQUEST_TIMEOUT. Never None.
    assert status_kwargs["timeout"] == 600
    assert content_kwargs["timeout"] == videos_main.DEFAULT_REQUEST_TIMEOUT
    assert status_kwargs["timeout"] is not None
    assert content_kwargs["timeout"] is not None

    # Explicit timeout threads through both branches unchanged.
    videos_main.video_status(video_id=video_id, custom_llm_provider="bedrock", timeout=33.5)
    videos_main.video_content(video_id=video_id, custom_llm_provider="bedrock", timeout=44.5)
    assert status_kwargs["timeout"] == 33.5
    assert content_kwargs["timeout"] == 44.5


def test_main_layer_bedrock_create_dispatch_threads_params(monkeypatch):
    """The create branch must merge the aws_* auth params riding on litellm_params into
    the handler optional params and thread timeout/api_key/avideo_generation through."""
    from litellm.llms.bedrock.videos.handler import BedrockVideoGeneration as _Handler
    from litellm.videos import main as videos_main

    seen: dict = {}

    def fake_generation(self, **kwargs):
        seen.update(kwargs)
        return Mock()

    monkeypatch.setattr(_Handler, "video_generation", fake_generation)
    result = videos_main.video_generation(
        prompt="waves at sunset",
        model="bedrock/amazon.nova-reel-v1:0",
        output_s3_uri="s3://bucket/out/",
        aws_region_name="eu-central-1",
        timeout=11.5,
        api_key="sigv4-key",
    )
    assert result is not None
    assert seen["timeout"] == 11.5
    assert seen["api_key"] == "sigv4-key"
    assert seen["avideo_generation"] is False
    optional_params: Final[dict] = seen["optional_params"]
    assert optional_params["output_s3_uri"] == "s3://bucket/out/"
    # aws_* auth params ride on litellm_params and are merged in by the dispatch.
    assert optional_params["aws_region_name"] == "eu-central-1"
    # The real litellm_params object (not a copy) reaches the handler.
    assert isinstance(seen["litellm_params"], videos_main.GenericLiteLLMParams)


def test_dispatch_functions_forward_kwargs_verbatim(monkeypatch):
    """The three dispatch shims forward their arguments verbatim to the handler."""
    from litellm.llms.bedrock.videos.dispatch import (
        dispatch_bedrock_video_content,
        dispatch_bedrock_video_generation,
        dispatch_bedrock_video_status,
    )
    from litellm.llms.bedrock.videos.handler import BedrockVideoGeneration as _Handler

    seen: dict[str, dict] = {}

    def fake_generation(self, **kwargs):
        seen["generation"] = kwargs
        return Mock()

    def fake_status(self, **kwargs):
        seen["status"] = kwargs
        return Mock()

    def fake_content(self, **kwargs):
        seen["content"] = kwargs
        return b"mp4-bytes"

    monkeypatch.setattr(_Handler, "video_generation", fake_generation)
    monkeypatch.setattr(_Handler, "video_status", fake_status)
    monkeypatch.setattr(_Handler, "video_content", fake_content)

    litellm_params: Final = GenericLiteLLMParams(api_base="https://examplebedrock", api_key="k1")

    dispatch_bedrock_video_generation(
        model="amazon.nova-reel-v1:0",
        prompt="waves",
        video_generation_request_params={"output_s3_uri": "s3://bucket/out/"},
        litellm_params=litellm_params,
        logging_obj=None,
        timeout=30.0,
        is_async=True,
        client="fake-client",
        extra_headers={"X-Test": "1"},
        api_key="sigv4-key",
    )
    generation_kwargs: Final[dict] = seen["generation"]
    assert generation_kwargs["model"] == "amazon.nova-reel-v1:0"
    assert generation_kwargs["prompt"] == "waves"
    assert generation_kwargs["optional_params"] == {"output_s3_uri": "s3://bucket/out/"}
    assert generation_kwargs["timeout"] == 30.0
    assert generation_kwargs["avideo_generation"] is True
    assert generation_kwargs["client"] == "fake-client"
    assert generation_kwargs["extra_headers"] == {"X-Test": "1"}
    assert generation_kwargs["api_key"] == "sigv4-key"
    assert generation_kwargs["api_base"] == "https://examplebedrock"
    assert generation_kwargs["litellm_params"] is litellm_params

    dispatch_bedrock_video_status(
        video_id="vid-1",
        litellm_params=litellm_params,
        logging_obj=None,
        api_base="https://examplebedrock",
        api_key="sigv4-key",
        astatus=False,
        timeout=600,
    )
    status_kwargs: Final[dict] = seen["status"]
    assert status_kwargs == {
        "video_id": "vid-1",
        "litellm_params": litellm_params,
        "logging_obj": None,
        "api_base": "https://examplebedrock",
        "api_key": "sigv4-key",
        "astatus": False,
        "timeout": 600,
    }

    content_result = dispatch_bedrock_video_content(
        video_id="vid-1",
        litellm_params=litellm_params,
        logging_obj=None,
        api_base="https://examplebedrock",
        api_key="sigv4-key",
        timeout=60.0,
    )
    assert content_result == b"mp4-bytes"
    assert seen["content"] == {
        "video_id": "vid-1",
        "litellm_params": litellm_params,
        "logging_obj": None,
        "api_base": "https://examplebedrock",
        "api_key": "sigv4-key",
        "timeout": 60.0,
    }


#################################################
# _to_epoch iso8601 timestamps
#################################################


def test_to_epoch_iso8601_with_z_suffix():
    """Real GetAsyncInvoke payloads carry iso8601 submitTime/endTime (Smithy timestampFormat)."""
    from litellm.llms.bedrock.videos.transformation import _to_epoch

    expected: Final = int(datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp())
    assert _to_epoch("2026-01-15T10:30:00Z") == expected


def test_to_epoch_numeric_epoch_still_works():
    from litellm.llms.bedrock.videos.transformation import _to_epoch

    assert _to_epoch(1758000000.75) == 1758000000
    assert _to_epoch("1758000000.75") == 1758000000


def test_to_epoch_garbage_warns_and_returns_none(monkeypatch):
    from litellm.llms.bedrock.videos.transformation import _to_epoch

    logger = Mock()
    monkeypatch.setattr("litellm.llms.bedrock.videos.transformation.verbose_logger", logger)
    assert _to_epoch("not-a-timestamp") is None
    logger.warning.assert_called_once()


def test_transform_status_response_iso8601_times_become_epochs():
    config = _make_config()
    resp = httpx.Response(
        200,
        json={
            "invocationArn": TEST_ARN,
            "status": "Completed",
            "submitTime": "2026-01-15T10:30:00Z",
            "endTime": "2026-01-15T10:31:00Z",
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}},
        },
    )
    video = config.transform_video_status_retrieve_response(raw_response=resp, logging_obj=None, model=TEST_MODEL)
    submit: Final = int(datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp())
    end: Final = int(datetime(2026, 1, 15, 10, 31, 0, tzinfo=timezone.utc).timestamp())
    assert video.created_at == submit
    assert video.completed_at == end


#################################################
# empty prompt guard
#################################################


def test_transform_create_request_blank_prompt_raises():
    with pytest.raises(BedrockError) as excinfo:
        _create_request(prompt="   ")
    assert excinfo.value.status_code == 400
    assert "prompt is required" in str(excinfo.value.message)


#################################################
# s3:// scheme validation
#################################################


def test_transform_create_request_non_s3_output_uri_raises():
    with pytest.raises(BedrockError) as excinfo:
        _create_request({"output_s3_uri": "https://bucket/out/"})
    assert excinfo.value.status_code == 400
    assert "s3://" in str(excinfo.value.message)


def test_parse_s3_uri_rejects_non_s3_scheme():
    from litellm.llms.bedrock.videos.handler import _parse_s3_uri

    with pytest.raises(BedrockError) as excinfo:
        _parse_s3_uri("https://example.com/bucket/key")
    assert excinfo.value.status_code == 400
    assert "s3://" in str(excinfo.value.message)


#################################################
# MULTI_SHOT body construction
#################################################


def test_transform_multi_shot_automated_uses_automated_params():
    body = _create_request({"output_s3_uri": "s3://bucket/out/", "taskType": "MULTI_SHOT_AUTOMATED"})
    model_input = body["modelInput"]
    assert model_input["taskType"] == "MULTI_SHOT_AUTOMATED"
    assert model_input["multiShotAutomatedParams"] == {"text": "A drone shot over the ocean"}
    assert "textToVideoParams" not in model_input
    # durationSeconds stays on videoGenerationConfig for automated multi-shot.
    assert model_input["videoGenerationConfig"]["durationSeconds"] == 6


def test_transform_multi_shot_automated_preserves_explicit_params():
    body = _create_request(
        {
            "output_s3_uri": "s3://bucket/out/",
            "taskType": "MULTI_SHOT_AUTOMATED",
            "multiShotAutomatedParams": {"text": "custom shot plan"},
        }
    )
    assert body["modelInput"]["multiShotAutomatedParams"] == {"text": "custom shot plan"}


def test_transform_multi_shot_automated_with_image_raises():
    with pytest.raises(BedrockError) as excinfo:
        _create_request({"output_s3_uri": "s3://bucket/out/", "taskType": "MULTI_SHOT_AUTOMATED", "image": PNG_BYTES})
    assert excinfo.value.status_code == 400
    assert "does not accept input images" in str(excinfo.value.message)


def test_transform_multi_shot_manual_without_params_raises():
    with pytest.raises(BedrockError) as excinfo:
        _create_request({"output_s3_uri": "s3://bucket/out/", "taskType": "MULTI_SHOT_MANUAL"})
    assert excinfo.value.status_code == 400
    assert "multiShotManualParams" in str(excinfo.value.message)


def test_transform_multi_shot_manual_body_omits_duration_seconds():
    body = _create_request(
        {
            "output_s3_uri": "s3://bucket/out/",
            "taskType": "MULTI_SHOT_MANUAL",
            "multiShotManualParams": {"shots": [{"text": "shot one", "durationSeconds": 6}]},
        }
    )
    model_input = body["modelInput"]
    assert model_input["multiShotManualParams"] == {"shots": [{"text": "shot one", "durationSeconds": 6}]}
    assert "textToVideoParams" not in model_input
    # Durations live per shot for MANUAL; no top-level durationSeconds.
    assert "durationSeconds" not in model_input["videoGenerationConfig"]


#################################################
# kmsKeyId/bucketOwner plumbing
#################################################


def test_kms_and_bucket_owner_forwarded_to_s3_output_config():
    body = _create_request(
        {
            "output_s3_uri": "s3://bucket/out/",
            "kmsKeyId": "arn:aws:kms:us-east-1:111122223333:key/k",
            "bucketOwner": "111122223333",
        }
    )
    s3_config: Final = body["outputDataConfig"]["s3OutputDataConfig"]
    assert s3_config["s3Uri"] == "s3://bucket/out/"
    assert s3_config["kmsKeyId"] == "arn:aws:kms:us-east-1:111122223333:key/k"
    assert s3_config["bucketOwner"] == "111122223333"
    assert "kmsKeyId" not in body["modelInput"]
    assert "bucketOwner" not in body["modelInput"]


def test_kms_and_bucket_owner_snake_case_aliases_accepted():
    body = _create_request(
        {
            "output_s3_uri": "s3://bucket/out/",
            "output_s3_kms_key_id": "key-id-1",
            "output_s3_bucket_owner": "222233334444",
        }
    )
    s3_config: Final = body["outputDataConfig"]["s3OutputDataConfig"]
    assert s3_config["kmsKeyId"] == "key-id-1"
    assert s3_config["bucketOwner"] == "222233334444"
    assert "output_s3_kms_key_id" not in body["modelInput"]
    assert "output_s3_bucket_owner" not in body["modelInput"]


#################################################
# non-JSON 2xx responses
#################################################


def test_transform_create_response_non_json_maps_to_bedrock_502():
    config = _make_config()
    resp = httpx.Response(200, content=b"<html>gateway error</html>")
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_create_response(model=TEST_MODEL, raw_response=resp, logging_obj=None)
    assert excinfo.value.status_code == 502
    assert "non-JSON" in str(excinfo.value.message)


def test_transform_status_response_non_json_maps_to_bedrock_502():
    config = _make_config()
    resp = httpx.Response(200, content=b"not json at all")
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_status_retrieve_response(raw_response=resp, logging_obj=None, model=TEST_MODEL)
    assert excinfo.value.status_code == 502
    assert "non-JSON" in str(excinfo.value.message)


#################################################
# unsupported operations raise a 400-class error
#################################################


def test_unsupported_list_operation_raises_400_class_bedrock_error():
    """litellm.exception_type maps BedrockError(status_code=400) to BadRequestError;
    a plain NotImplementedError would surface as APIConnectionError 500."""
    config = _make_config()
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_list_request(
            api_base="",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "not supported" in str(excinfo.value.message)


def test_unsupported_remix_operation_raises_400_class_bedrock_error():
    config = _make_config()
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_remix_request(
            video_id="vid",
            prompt="p",
            api_base="",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
    assert excinfo.value.status_code == 400


#################################################
# unsupported character/edit/extension operations raise 400-class errors
#################################################


def test_unsupported_create_character_operation_raises_400_class_bedrock_error():
    config = _make_config()
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_create_character_request(
            name="hero",
            video=b"video-bytes",
            api_base="",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "create character" in str(excinfo.value.message)


def test_unsupported_create_character_response_raises_400_class_bedrock_error():
    config = _make_config()
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_create_character_response(raw_response=Mock(), logging_obj=None)
    assert excinfo.value.status_code == 400


def test_unsupported_get_character_operation_raises_400_class_bedrock_error():
    config = _make_config()
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_get_character_request(
            character_id="char-1",
            api_base="",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "get character" in str(excinfo.value.message)


def test_unsupported_get_character_response_raises_400_class_bedrock_error():
    config = _make_config()
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_get_character_response(raw_response=Mock(), logging_obj=None)
    assert excinfo.value.status_code == 400


def test_unsupported_edit_operation_raises_400_class_bedrock_error():
    config = _make_config()
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_edit_request(
            prompt="brighter",
            video_id="vid",
            api_base="",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "video edit" in str(excinfo.value.message)


def test_unsupported_edit_response_raises_400_class_bedrock_error():
    config = _make_config()
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_edit_response(raw_response=Mock(), logging_obj=None)
    assert excinfo.value.status_code == 400


def test_unsupported_extension_operation_raises_400_class_bedrock_error():
    config = _make_config()
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_extension_request(
            prompt="longer",
            video_id="vid",
            seconds="6",
            api_base="",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "video extension" in str(excinfo.value.message)


def test_unsupported_extension_response_raises_400_class_bedrock_error():
    config = _make_config()
    with pytest.raises(BedrockError) as excinfo:
        config.transform_video_extension_response(raw_response=Mock(), logging_obj=None)
    assert excinfo.value.status_code == 400


def test_avideo_edit_bedrock_maps_unsupported_operation_to_bad_request(monkeypatch):
    """Through the litellm video layer, litellm.avideo_edit on bedrock must surface
    BadRequestError (400-class), never a NotImplementedError-driven 500. get_complete_url
    is stubbed because it is documented as unused for this config (URLs are built in
    the handler); the unsupported-operation error comes from the transform."""
    monkeypatch.setattr(
        BedrockNovaReelVideoConfig,
        "get_complete_url",
        lambda self, model, api_base, litellm_params: "https://example.com/videos/edits",
    )
    with pytest.raises(litellm.BadRequestError) as excinfo:
        asyncio.run(
            litellm.avideo_edit(
                video_id="some-video-id",
                prompt="brighter",
                custom_llm_provider="bedrock",
            )
        )
    # The 400-class mapping is the contract; the generic edit error wrapper reads
    # .text off the exception (empty for a synthesized BedrockError), so only the
    # status survives here. The message is asserted in the config-level tests.
    assert excinfo.value.status_code == 400


def test_avideo_create_character_bedrock_maps_unsupported_operation_to_bad_request(monkeypatch):
    """Through the litellm video layer, litellm.avideo_create_character on bedrock must
    surface BadRequestError (400-class). get_complete_url is stubbed as above."""
    monkeypatch.setattr(
        BedrockNovaReelVideoConfig,
        "get_complete_url",
        lambda self, model, api_base, litellm_params: "https://example.com/characters",
    )
    with pytest.raises(litellm.BadRequestError) as excinfo:
        asyncio.run(
            litellm.avideo_create_character(
                name="hero",
                video=b"video-bytes",
                custom_llm_provider="bedrock",
            )
        )
    assert excinfo.value.status_code == 400
    assert "video create character is not supported" in str(excinfo.value)


#################################################
# text-mode file guard
#################################################


def test_input_reference_text_mode_file_raises_value_error():
    with pytest.raises(BedrockError) as excinfo:
        _create_request({"output_s3_uri": "s3://bucket/out/", "input_reference": io.StringIO("not binary")})
    assert excinfo.value.status_code == 400
    assert "binary mode" in str(excinfo.value.message)


#################################################
# clientRequestToken falls back to litellm_call_id
#################################################


def test_client_request_token_falls_back_to_litellm_call_id():
    litellm_params = GenericLiteLLMParams()
    litellm_params.litellm_call_id = "call/abc_123"  # extra field set by the @client decorator
    body = _create_request(litellm_params=litellm_params)
    assert body["clientRequestToken"] == "call-abc-123"


def test_handler_create_threads_litellm_call_id_from_mapping_into_token(monkeypatch):
    """A Mapping litellm_params carrying only litellm_call_id (no metadata) must reach
    the signed POST body through _as_generic_litellm_params."""
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
    video = handler.video_generation(
        model="bedrock/amazon.nova-reel-v1:0",
        prompt="waves at sunset",
        optional_params={"output_s3_uri": "s3://bucket/out/"},
        logging_obj=None,
        timeout=5.0,
        avideo_generation=False,
        litellm_params={"litellm_call_id": "call/xyz_789"},
    )
    assert video.status == "processing"
    parsed: Final = json.loads(bodies[0])
    assert parsed["clientRequestToken"] == "call-xyz-789"


#################################################
# S3 download error mapping per AWS error code
#################################################


def test_download_s3_object_access_denied_maps_to_403_without_second_get(monkeypatch):
    from botocore.exceptions import ClientError

    def _deny(bucket, key):
        raise ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "GetObject",
        )

    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}
    _, clients, attempted = _patch_s3_download(monkeypatch, handler, _deny)
    with pytest.raises(BedrockError) as excinfo:
        handler._download_s3_object(
            "bucket",
            ["out/abc/output.mp4", "out/output.mp4"],
            {},
            raw,
            region_default="us-east-1",
        )
    assert excinfo.value.status_code == 403
    # Aborts immediately: no second candidate get_object.
    assert attempted == ["out/abc/output.mp4"]
    assert "s3://bucket/out/" in str(excinfo.value.message)
    assert "AccessDenied" in str(excinfo.value.message)
    assert clients[0].closed is True


def test_download_s3_object_500_class_client_error_maps_to_502(monkeypatch):
    from botocore.exceptions import ClientError

    def _internal_error(bucket, key):
        raise ClientError(
            {
                "Error": {"Code": "InternalError", "Message": "We encountered an internal error."},
                "ResponseMetadata": {"HTTPStatusCode": 500},
            },
            "GetObject",
        )

    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}
    _, _, attempted = _patch_s3_download(monkeypatch, handler, _internal_error)
    with pytest.raises(BedrockError) as excinfo:
        handler._download_s3_object("bucket", ["out/output.mp4"], {}, raw, region_default="us-east-1")
    assert excinfo.value.status_code == 502
    # The AWS error code is preserved in the message.
    assert "InternalError" in str(excinfo.value.message)
    assert attempted == ["out/output.mp4"]


def test_download_s3_object_http_404_client_error_falls_through(monkeypatch):
    """A 404 without a NoSuchKey code still falls through to the next candidate."""
    from botocore.exceptions import ClientError

    def _missing_via_http_status(bucket, key):
        if key.endswith("abc123-def456/output.mp4"):
            raise ClientError(
                {
                    "Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "GetObject",
            )
        return {"Body": io.BytesIO(b"mp4-after-404")}

    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}
    _, _, attempted = _patch_s3_download(monkeypatch, handler, _missing_via_http_status)
    content = handler._download_s3_object(
        "bucket",
        ["out/abc123-def456/output.mp4", "out/output.mp4"],
        {},
        raw,
        region_default="us-east-1",
    )
    assert content == b"mp4-after-404"
    assert len(attempted) == 2


#################################################
# bearer-only S3 path
#################################################


def test_download_s3_object_bearer_without_sigv4_credentials_maps_to_400(monkeypatch):
    """A bedrock bearer token covers only the async-invoke API; S3 needs SigV4."""
    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}
    _, clients, attempted = _patch_s3_download(monkeypatch, handler, lambda bucket, key: {"Body": io.BytesIO(b"x")})
    with pytest.raises(BedrockError) as excinfo:
        handler._download_s3_object(
            "bucket",
            ["out/output.mp4"],
            {},
            raw,
            region_default="us-east-1",
            api_key="some-bearer-token",
        )
    assert excinfo.value.status_code == 400
    assert "SigV4" in str(excinfo.value.message)
    assert "bearer" in str(excinfo.value.message)
    # Failed fast: no S3 client built, no get_object attempted.
    assert clients == []
    assert attempted == []


def test_download_s3_object_bearer_with_failing_credential_chain_maps_to_400(monkeypatch):
    """A bearer token plus an unresolvable SigV4 chain (NoCredentialsError from
    _load_credentials itself) must surface the bearer guidance 400, not a raw
    BotoCoreError/500."""
    from botocore.exceptions import NoCredentialsError

    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}

    def _raise_credentials(optional_params, aws_region_name=None, bearer_token=None):
        raise NoCredentialsError()

    monkeypatch.setattr(handler, "_load_credentials", _raise_credentials)
    with pytest.raises(BedrockError) as excinfo:
        handler._download_s3_object(
            "bucket",
            ["out/output.mp4"],
            {},
            raw,
            region_default="us-east-1",
            api_key="some-bearer-token",
        )
    assert excinfo.value.status_code == 400
    assert "SigV4" in str(excinfo.value.message)
    assert "bearer" in str(excinfo.value.message)


def test_download_s3_object_no_bearer_credential_chain_failure_propagates(monkeypatch):
    """Without a bearer token the NoCredentialsError propagation is unchanged."""
    from botocore.exceptions import NoCredentialsError

    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}

    def _raise_credentials(optional_params, aws_region_name=None, bearer_token=None):
        raise NoCredentialsError()

    monkeypatch.setattr(handler, "_load_credentials", _raise_credentials)
    with pytest.raises(NoCredentialsError):
        handler._download_s3_object(
            "bucket",
            ["out/output.mp4"],
            {},
            raw,
            region_default="us-east-1",
        )


#################################################
# non-JSON status responses through the handler
#################################################


def test_map_status_response_non_json_maps_to_bedrock_502():
    """A 200 with a non-JSON body must surface as BedrockError 502 from the handler's
    guarded parse (before the transform's own guard, which never runs)."""
    handler = BedrockVideoGeneration()
    resp = httpx.Response(200, content=b"<html>gateway error</html>")
    with pytest.raises(BedrockError) as excinfo:
        handler._map_status_response(resp, TEST_MODEL, "video-id", None)
    assert excinfo.value.status_code == 502
    assert "non-JSON response from Bedrock status endpoint" in str(excinfo.value.message)


def test_video_status_non_json_body_maps_to_bedrock_502(monkeypatch):
    """End-to-end through video_status: non-JSON 200 -> 502, never a raw 500."""
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
    resp = httpx.Response(200, content=b"not json at all")
    monkeypatch.setattr(handler, "_sync_get", lambda prepped, timeout=None: resp)
    with pytest.raises(BedrockError) as excinfo:
        handler.video_status(video_id=video_id, litellm_params={})
    assert excinfo.value.status_code == 502
    assert "non-JSON response from Bedrock status endpoint" in str(excinfo.value.message)


#################################################
# S3 client timeouts + resource closes
#################################################


def test_download_s3_object_builds_client_with_config_timeouts(monkeypatch):
    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}
    _, clients, _ = _patch_s3_download(monkeypatch, handler, lambda bucket, key: {"Body": io.BytesIO(b"x")})

    # Call timeout threads into read_timeout; connect stays at the 5s default.
    handler._download_s3_object("bucket", ["out/output.mp4"], {}, raw, region_default="us-east-1", timeout=12.5)
    assert clients[0].config.read_timeout == 12.5
    assert clients[0].config.connect_timeout == 5

    # No timeout: sane defaults.
    handler._download_s3_object("bucket", ["out/output.mp4"], {}, raw, region_default="us-east-1")
    assert clients[1].config.read_timeout == 60.0
    assert clients[1].config.connect_timeout == 5

    # A small call timeout caps the connect timeout too.
    handler._download_s3_object("bucket", ["out/output.mp4"], {}, raw, region_default="us-east-1", timeout=2.0)
    assert clients[2].config.read_timeout == 2.0
    assert clients[2].config.connect_timeout == 2.0


def test_download_s3_object_closes_body_and_client_on_success(monkeypatch):
    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}
    body = _TrackingBody(b"mp4-bytes")
    _, clients, _ = _patch_s3_download(monkeypatch, handler, lambda bucket, key: {"Body": body})
    content = handler._download_s3_object("bucket", ["out/output.mp4"], {}, raw, region_default="us-east-1")
    assert content == b"mp4-bytes"
    assert body.close_calls == 1
    assert clients[0].closed is True


def test_download_s3_object_closes_client_on_404(monkeypatch):
    from botocore.exceptions import ClientError

    def _raise(bucket, key):
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
            "GetObject",
        )

    handler = BedrockVideoGeneration()
    raw: dict = {"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/out/"}}}
    _, clients, _ = _patch_s3_download(monkeypatch, handler, _raise)
    with pytest.raises(BedrockError):
        handler._download_s3_object("bucket", ["out/output.mp4"], {}, raw, region_default="us-east-1")
    assert clients[0].closed is True


#################################################
# video id decoded once per status call
#################################################


def test_video_status_decodes_video_id_once(monkeypatch):
    import litellm.llms.bedrock.videos.handler as handler_module
    from litellm.types.videos.utils import encode_video_id_with_provider

    real_decode = handler_module.decode_video_id_with_provider
    decode_calls: list[str] = []

    def counting_decode(video_id):
        decode_calls.append(video_id)
        return real_decode(video_id)

    monkeypatch.setattr(handler_module, "decode_video_id_with_provider", counting_decode)

    handler = BedrockVideoGeneration()
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
    resp = httpx.Response(200, json={"invocationArn": TEST_ARN, "status": "InProgress"})
    monkeypatch.setattr(handler, "_sync_get", lambda prepped, timeout=None: resp)
    handler.video_status(video_id=video_id, litellm_params={})
    assert decode_calls == [video_id]


#################################################
# private async status arm
#################################################


def test_video_status_async_dispatch_uses_private_async_arm(monkeypatch):
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
    resp = httpx.Response(200, json={"invocationArn": TEST_ARN, "status": "InProgress", "submitTime": 1758000000.0})

    async def fake_async_get(prepped, timeout=None):
        return resp

    monkeypatch.setattr(handler, "_async_get", fake_async_get)
    assert not hasattr(BedrockVideoGeneration, "async_video_status")
    video = asyncio.run(handler.video_status(video_id=video_id, litellm_params={}, astatus=True))
    assert video.status == "processing"


#################################################
# api_key falls back to litellm_params on the bedrock branches
#################################################


def test_bedrock_branches_fall_back_to_litellm_params_api_key(monkeypatch):
    """kwargs without api_key but litellm_params.api_key set: the handler must receive it."""
    from litellm.llms.bedrock.videos.handler import BedrockVideoGeneration as _Handler
    from litellm.types.videos.utils import encode_video_id_with_provider
    from litellm.videos import main as videos_main

    real_params_cls = videos_main.GenericLiteLLMParams

    class _InjectsApiKey(real_params_cls):
        def __init__(self, **kw):
            kw.setdefault("api_key", "sigv4-key-from-litellm-params")
            super().__init__(**kw)

    seen: dict[str, dict] = {}

    def fake_generation(self, **kwargs):
        seen["create"] = kwargs
        return Mock()

    def fake_status(self, **kwargs):
        seen["status"] = kwargs
        return Mock()

    def fake_content(self, **kwargs):
        seen["content"] = kwargs
        return b"mp4-bytes"

    monkeypatch.setattr(videos_main, "GenericLiteLLMParams", _InjectsApiKey)
    monkeypatch.setattr(_Handler, "video_generation", fake_generation)
    monkeypatch.setattr(_Handler, "video_status", fake_status)
    monkeypatch.setattr(_Handler, "video_content", fake_content)

    videos_main.video_generation(prompt="waves", model="bedrock/amazon.nova-reel-v1:0")
    video_id = encode_video_id_with_provider(TEST_ARN, "bedrock", TEST_MODEL)
    videos_main.video_status(video_id=video_id, custom_llm_provider="bedrock")
    videos_main.video_content(video_id=video_id, custom_llm_provider="bedrock")

    assert seen["create"]["api_key"] == "sigv4-key-from-litellm-params"
    assert seen["status"]["api_key"] == "sigv4-key-from-litellm-params"
    assert seen["content"]["api_key"] == "sigv4-key-from-litellm-params"


#################################################
# proxy contract: user-input validation maps to 400-class errors
#################################################


def test_avideo_generation_missing_output_s3_uri_maps_to_bad_request(monkeypatch):
    """Through the litellm video layer, the missing-output_s3_uri validation error must
    surface as litellm.BadRequestError (400-class), never APIConnectionError/500."""
    monkeypatch.setattr(
        BedrockVideoGeneration,
        "_get_boto_credentials_from_optional_params",
        lambda self, params, model=None, bearer_token=None: _FakeCredentialsInfo(),
    )
    with pytest.raises(litellm.BadRequestError) as excinfo:
        asyncio.run(
            litellm.avideo_generation(
                prompt="waves at sunset",
                model="bedrock/amazon.nova-reel-v1:0",
            )
        )
    assert excinfo.value.status_code == 400
    assert "output_s3_uri" in str(excinfo.value)


def test_avideo_content_not_complete_yet_maps_to_bad_request(monkeypatch):
    """Through the litellm video layer, downloading an in-progress video must surface
    as litellm.BadRequestError (400-class), never APIConnectionError/500."""
    from litellm.types.videos.utils import encode_video_id_with_provider

    video_id = encode_video_id_with_provider(TEST_ARN, "bedrock", TEST_MODEL)
    monkeypatch.setattr(
        BedrockVideoGeneration,
        "_status_request_parts",
        lambda self, arn, params, api_base, api_key=None: (
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
    monkeypatch.setattr(BedrockVideoGeneration, "_sync_get", lambda self, prepped, timeout=None: resp)
    with pytest.raises(litellm.BadRequestError) as excinfo:
        asyncio.run(litellm.avideo_content(video_id=video_id, custom_llm_provider="bedrock"))
    assert excinfo.value.status_code == 400
    assert "not complete" in str(excinfo.value)


#################################################
# MULTI_SHOT_MANUAL spend: per-shot durations sum into usage
#################################################


def test_transform_multi_shot_manual_usage_sums_shot_durations():
    """3 shots x 6s must record usage.duration_seconds 18 (the billable total), not the
    omitted top-level durationSeconds; the video cost calculator then prices 18s."""
    from litellm.llms.openai.cost_calculation import video_generation_cost

    body = _create_request(
        {
            "output_s3_uri": "s3://bucket/out/",
            "taskType": "MULTI_SHOT_MANUAL",
            "multiShotManualParams": {
                "shots": [
                    {"text": "shot one", "durationSeconds": 6},
                    {"text": "shot two", "durationSeconds": 6},
                    {"text": "shot three", "durationSeconds": 6},
                ]
            },
        }
    )
    config = _make_config()
    resp = httpx.Response(200, json={"invocationArn": TEST_ARN})
    video = config.transform_video_create_response(
        model="amazon.nova-reel-v1:1",
        raw_response=resp,
        logging_obj=None,
        request_data=body,
    )
    assert video.usage is not None
    assert video.usage["duration_seconds"] == 18.0
    cost = video_generation_cost(
        model="amazon.nova-reel-v1:1",
        duration_seconds=video.usage["duration_seconds"],
        custom_llm_provider="bedrock",
    )
    assert cost == pytest.approx(18 * 0.08)


#################################################
# cross-region video ids resolve to the base-model deployment
#################################################


def test_cross_region_video_id_resolves_to_base_model_deployment():
    """An id encoded with us.amazon.nova-reel-v1:1 must resolve to a deployment whose
    litellm_params.model is the base bedrock/amazon.nova-reel-v1:1, so status/content
    dispatch through the router and pick up the deployment aws_* credentials."""
    from litellm.llms.bedrock.videos.transformation import BedrockNovaReelVideoConfig as _Config
    from litellm.proxy.video_endpoints.endpoints import _resolve_model_name_from_decoded_model_id
    from litellm.router import Router
    from litellm.types.videos.utils import encode_video_id_with_provider

    router = Router(
        model_list=[
            {
                "model_name": "amazon.nova-reel-v1:1",
                "litellm_params": {"model": "bedrock/amazon.nova-reel-v1:1"},
            }
        ]
    )
    invocation_arn = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/xyz"
    video_id = encode_video_id_with_provider(invocation_arn, "bedrock", "us.amazon.nova-reel-v1:1")
    decoded = decode_video_id_with_provider(video_id)
    assert decoded["model_id"] == "us.amazon.nova-reel-v1:1"
    # The pre-fix behavior: a bare resolve on the cross-region id finds nothing.
    assert router.resolve_model_name_from_model_id(decoded["model_id"]) is None
    resolved = _resolve_model_name_from_decoded_model_id(router, decoded["model_id"], "bedrock")
    assert resolved == "amazon.nova-reel-v1:1"
    # Scoped to bedrock: the same id under another provider stays unresolved.
    assert _resolve_model_name_from_decoded_model_id(router, decoded["model_id"], "vertex_ai") is None
    # Exact (non cross-region) ids keep resolving through the unchanged first lookup.
    assert _resolve_model_name_from_decoded_model_id(router, "amazon.nova-reel-v1:1", "bedrock") == (
        "amazon.nova-reel-v1:1"
    )
    # The status transform still encodes the cross-region id verbatim.
    assert _Config.extract_invocation_arn(video_id) == invocation_arn
