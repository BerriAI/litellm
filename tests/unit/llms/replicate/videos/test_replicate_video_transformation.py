from typing import Final
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.cost_calculator import default_video_cost_calculator
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.url_utils import SSRFError
from litellm.llms.replicate.common_utils import ReplicateError
from litellm.llms.replicate.videos.transformation import ReplicateVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.utils import decode_video_id_with_provider
from litellm.utils import ProviderConfigManager

API_BASE: Final = "https://api.replicate.com/v1"
MODEL: Final = "xai/grok-imagine-video"
MINIMAX_MODEL: Final = "minimax/h3-lite"
IMAGE_URL: Final = "https://example.com/frame.png"
VIDEO_URL: Final = "https://replicate.delivery/out/video.mp4"


@pytest.fixture(autouse=True)
def _skip_url_dns_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)


class _FakeClient:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self, url: str, headers: dict[str, str] | None = None, params: object = None, **kwargs: object
    ) -> httpx.Response:
        self.calls.append(url)
        return self.responses.pop(0)


class _FakeAsyncClient(_FakeClient):
    async def get(
        self, url: str, headers: dict[str, str] | None = None, params: object = None, **kwargs: object
    ) -> httpx.Response:  # pyright: ignore[reportIncompatibleMethodOverride]  # async twin of the sync fake
        return super().get(url, headers=headers, params=params, **kwargs)


def _prediction_response(body: dict[str, object], url: str = f"{API_BASE}/predictions/abc") -> httpx.Response:
    return httpx.Response(200, json=body, request=httpx.Request("GET", url, headers={"Authorization": "Bearer k"}))


def test_provider_config_is_registered():
    config = ProviderConfigManager.get_provider_video_config(model=MODEL, provider=LlmProviders.REPLICATE)
    assert isinstance(config, ReplicateVideoConfig)
    assert isinstance(config.get_error_class("bad key", 401, {}), ReplicateError)


def test_map_openai_params_generic_model():
    mapped = ReplicateVideoConfig().map_openai_params(
        {"seconds": "6", "size": "1280x720", "input_reference": IMAGE_URL, "user": "u1", "seed": 7},
        MODEL,
        False,
    )
    assert mapped == {"image": IMAGE_URL, "duration": 6, "resolution": "720p", "aspect_ratio": "16:9", "seed": 7}
    assert isinstance(mapped["duration"], int)


@pytest.mark.parametrize("model", [MINIMAX_MODEL, f"deployments/{MINIMAX_MODEL}", f"{MINIMAX_MODEL}:0123abcd"])
def test_map_openai_params_minimax_profile(model: str):
    mapped = ReplicateVideoConfig().map_openai_params(
        {"seconds": 5, "size": "1024x768", "input_reference": IMAGE_URL}, model, False
    )
    assert mapped == {"first_frame_image": IMAGE_URL, "duration": 5, "resolution": "768P", "ratio": "4:3"}


@pytest.mark.parametrize(
    ("size", "expected"),
    [("720p", {"resolution": "720p"}), ("1080P", {"resolution": "1080p"}), ("4k", {"resolution": "4k"})],
)
def test_map_openai_params_named_sizes(size: str, expected: dict[str, str]):
    assert ReplicateVideoConfig().map_openai_params({"size": size}, MODEL, False) == expected


def test_map_openai_params_fractional_seconds_and_bad_values():
    assert ReplicateVideoConfig().map_openai_params({"seconds": "2.5"}, MODEL, False) == {"duration": 2.5}
    with pytest.raises(ValueError, match="numeric"):
        ReplicateVideoConfig().map_openai_params({"seconds": "auto"}, MODEL, False)
    with pytest.raises(ValueError, match="public image URL"):
        ReplicateVideoConfig().map_openai_params({"input_reference": b"bytes"}, MODEL, False)


def test_map_openai_params_unsupported_native_fields():
    with pytest.raises(UnsupportedParamsError, match="frame_images"):
        ReplicateVideoConfig().map_openai_params({"frame_images": [{"url": IMAGE_URL}]}, MODEL, False)
    assert ReplicateVideoConfig().map_openai_params(
        {"frame_images": [{"url": IMAGE_URL}], "seconds": 5}, MODEL, True
    ) == {"duration": 5}


def test_validate_environment_requires_key_and_sends_bearer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("REPLICATE_API_KEY", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.setattr(litellm, "replicate_key", None)
    with pytest.raises(ReplicateError, match="REPLICATE_API_KEY"):
        ReplicateVideoConfig().validate_environment({}, MODEL)
    headers = ReplicateVideoConfig().validate_environment({"X-Trace": "1"}, MODEL, api_key="r8_key")
    assert headers["Authorization"] == "Bearer r8_key"
    assert headers["X-Trace"] == "1"
    via_params = ReplicateVideoConfig().validate_environment(
        {}, MODEL, litellm_params=GenericLiteLLMParams(api_key="r8_p")
    )
    assert via_params["Authorization"] == "Bearer r8_p"


@pytest.mark.parametrize(
    ("model", "url", "extra_body"),
    [
        (MODEL, f"{API_BASE}/models/{MODEL}/predictions", {}),
        ("owner/name:abc123", f"{API_BASE}/models/owner/name/versions/abc123/predictions", {}),
        ("deployments/acme/fast-video", f"{API_BASE}/deployments/acme/fast-video/predictions", {}),
    ],
)
def test_transform_video_create_request(model: str, url: str, extra_body: dict[str, str]):
    body, files, request_url = ReplicateVideoConfig().transform_video_create_request(
        model=model,
        prompt="a quiet ocean",
        api_base=API_BASE,
        video_create_optional_request_params={"duration": 5, "resolution": "720p", "model": model},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert request_url == url
    assert files == []
    assert body == {**extra_body, "input": {"prompt": "a quiet ocean", "duration": 5, "resolution": "720p"}}


def test_transform_video_create_request_keeps_the_inspected_prompt():
    body, _, _ = ReplicateVideoConfig().transform_video_create_request(
        model="minimax/h3-lite",
        prompt="a quiet ocean",
        api_base=API_BASE,
        video_create_optional_request_params={"duration": 5, "prompt": "unaudited"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert body["input"] == {"duration": 5, "prompt": "a quiet ocean"}


def test_get_complete_url_defaults_and_override():
    config = ReplicateVideoConfig()
    assert config.get_complete_url(MODEL, None, {}) == API_BASE
    assert config.get_complete_url(MODEL, "https://proxy.internal/", {}) == "https://proxy.internal/v1"
    assert config.get_complete_url(MODEL, "https://proxy.internal/v1", {}) == "https://proxy.internal/v1"


def test_transform_video_create_response_encodes_id_and_usage():
    response = _prediction_response(
        {"id": "abc", "status": "starting", "input": {"prompt": "x", "duration": 6, "resolution": "480p"}}
    )
    video = ReplicateVideoConfig().transform_video_create_response(MODEL, response, Mock(), "replicate")
    decoded = decode_video_id_with_provider(video.id)
    assert decoded["custom_llm_provider"] == "replicate"
    assert decoded["model_id"] == MODEL
    assert decoded["video_id"] == "abc"
    assert video.status == "queued"
    assert video.model == MODEL
    assert video.seconds == "6"
    assert video.size == "480p"
    assert video.usage == {"duration_seconds": 6.0, "video_resolution": "480p"}


def test_status_request_uses_prediction_id_from_encoded_video_id():
    create = _prediction_response({"id": "abc", "status": "starting"})
    video = ReplicateVideoConfig().transform_video_create_response(MODEL, create, Mock())
    url, data = ReplicateVideoConfig().transform_video_status_retrieve_request(
        video.id, API_BASE, GenericLiteLLMParams(), {}
    )
    assert url == f"{API_BASE}/predictions/abc#{MODEL}"
    assert httpx.URL(url).raw_path == b"/v1/predictions/abc"
    assert data == {}
    content_url, _ = ReplicateVideoConfig().transform_video_content_request(
        video.id, API_BASE, GenericLiteLLMParams(), {}
    )
    assert content_url == url


@pytest.mark.parametrize("model", [f"{MODEL}:0123abcd", "deployments/acme/fast-video"])
def test_status_response_keeps_the_requested_model_and_video_id(model: str):
    create = _prediction_response({"id": "abc", "status": "starting", "model": "xai/grok-imagine-video"})
    video = ReplicateVideoConfig().transform_video_create_response(model, create, Mock())
    url, _ = ReplicateVideoConfig().transform_video_status_retrieve_request(
        video.id, API_BASE, GenericLiteLLMParams(), {}
    )
    status = httpx.Response(
        200,
        json={"id": "abc", "status": "succeeded", "model": "xai/grok-imagine-video", "output": VIDEO_URL},
        request=httpx.Request("GET", url),
    )
    polled = ReplicateVideoConfig().transform_video_status_retrieve_response(status, Mock())
    assert polled.id == video.id
    assert polled.model == model


@pytest.mark.parametrize(
    ("replicate_status", "expected"),
    [("starting", "queued"), ("processing", "in_progress"), ("succeeded", "completed")],
)
def test_status_response_maps_statuses(replicate_status: str, expected: str):
    response = _prediction_response(
        {"id": "abc", "status": replicate_status, "model": MODEL, "input": {"duration": 5, "resolution": "720p"}}
    )
    video = ReplicateVideoConfig().transform_video_status_retrieve_response(response, Mock(), "replicate")
    assert video.status == expected
    assert video.error is None
    assert video.model == MODEL
    assert video.usage == {"duration_seconds": 5.0, "video_resolution": "720p"}
    assert decode_video_id_with_provider(video.id)["video_id"] == "abc"


@pytest.mark.parametrize("replicate_status", ["failed", "canceled"])
def test_status_response_surfaces_failures(replicate_status: str):
    response = _prediction_response({"id": "abc", "status": replicate_status, "error": "NSFW content detected"})
    video = ReplicateVideoConfig().transform_video_status_retrieve_response(response, Mock())
    assert video.status == "failed"
    assert video.error == {"code": "replicate_error", "message": "NSFW content detected"}


def test_status_response_without_duration_has_no_usage():
    response = _prediction_response({"id": "abc", "status": "succeeded", "input": {"prompt": "x"}})
    assert ReplicateVideoConfig().transform_video_status_retrieve_response(response, Mock()).usage == {}


@pytest.mark.parametrize("output", [VIDEO_URL, [VIDEO_URL], {"video": VIDEO_URL}])
def test_content_response_downloads_first_output(output: object):
    download = httpx.Response(200, content=b"mp4-bytes", request=httpx.Request("GET", VIDEO_URL))
    client = _FakeClient(download)
    config = ReplicateVideoConfig(sync_client_factory=lambda: client)  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    response = _prediction_response({"id": "abc", "status": "succeeded", "output": output})
    assert config.transform_video_content_response(response, Mock()) == b"mp4-bytes"
    assert client.calls == [VIDEO_URL]


@pytest.mark.asyncio
async def test_async_content_response_downloads_first_output():
    download = httpx.Response(200, content=b"mp4-bytes", request=httpx.Request("GET", VIDEO_URL))
    client = _FakeAsyncClient(download)
    config = ReplicateVideoConfig(async_client_factory=lambda: client)  # pyright: ignore[reportArgumentType]  # fake stands in for AsyncHTTPHandler
    response = _prediction_response({"id": "abc", "status": "succeeded", "output": [VIDEO_URL]})
    assert await config.async_transform_video_content_response(response, Mock()) == b"mp4-bytes"


def test_content_response_prefers_the_handler_client():
    download = httpx.Response(200, content=b"mp4-bytes", request=httpx.Request("GET", VIDEO_URL))
    handler_client = _FakeClient(download)
    config = ReplicateVideoConfig(sync_client_factory=lambda: _FakeClient())  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    response = _prediction_response({"id": "abc", "status": "succeeded", "output": VIDEO_URL})
    assert config.transform_video_content_response(response, Mock(), client=handler_client) == b"mp4-bytes"  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    assert handler_client.calls == [VIDEO_URL]


@pytest.mark.parametrize(
    ("requested_model", "prediction"),
    [
        (
            "acme/other-video",
            {"id": "abc", "status": "succeeded", "model": "xai/grok-imagine-video", "output": VIDEO_URL},
        ),
        (f"{MODEL}:v2", {"id": "abc", "status": "succeeded", "model": MODEL, "version": "v1", "output": VIDEO_URL}),
    ],
)
def test_retrieval_rejects_a_prediction_from_another_model(requested_model: str, prediction: dict[str, object]):
    create = _prediction_response({"id": "abc", "status": "starting"})
    video = ReplicateVideoConfig().transform_video_create_response(requested_model, create, Mock())
    config = ReplicateVideoConfig(sync_client_factory=lambda: _FakeClient())  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    status_url, _ = config.transform_video_status_retrieve_request(video.id, API_BASE, GenericLiteLLMParams(), {})
    foreign = httpx.Response(200, json=prediction, request=httpx.Request("GET", status_url))
    with pytest.raises(ReplicateError, match="does not belong to") as excinfo:
        config.transform_video_status_retrieve_response(foreign, Mock())
    assert excinfo.value.status_code == 404
    with pytest.raises(ReplicateError, match="does not belong to"):
        config.transform_video_content_response(foreign, Mock())


def test_retrieval_accepts_deployment_predictions_without_a_model_check():
    create = _prediction_response({"id": "abc", "status": "starting"})
    video = ReplicateVideoConfig().transform_video_create_response("deployments/acme/fast", create, Mock())
    status_url, _ = ReplicateVideoConfig().transform_video_status_retrieve_request(
        video.id, API_BASE, GenericLiteLLMParams(), {}
    )
    status = httpx.Response(
        200,
        json={"id": "abc", "status": "succeeded", "model": "xai/grok-imagine-video"},
        request=httpx.Request("GET", status_url),
    )
    assert (
        ReplicateVideoConfig().transform_video_status_retrieve_response(status, Mock()).model == "deployments/acme/fast"
    )


def test_content_download_blocks_internal_urls(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "user_url_validation", True)
    config = ReplicateVideoConfig(sync_client_factory=lambda: _FakeClient())  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    response = _prediction_response(
        {"id": "abc", "status": "succeeded", "output": "http://169.254.169.254/latest/meta-data"}
    )
    with pytest.raises(SSRFError):
        config.transform_video_content_response(response, Mock())


def test_content_response_errors_without_output():
    config = ReplicateVideoConfig(sync_client_factory=lambda: _FakeClient())  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    with pytest.raises(ReplicateError, match="no video output yet"):
        config.transform_video_content_response(_prediction_response({"id": "abc", "status": "processing"}), Mock())
    with pytest.raises(ReplicateError, match="boom"):
        config.transform_video_content_response(
            _prediction_response({"id": "abc", "status": "failed", "error": "boom"}), Mock()
        )


def test_completed_status_usage_prices_through_cost_map(local_model_cost_map: None):
    row = litellm.model_cost[f"replicate/{MINIMAX_MODEL}"]
    response = _prediction_response(
        {"id": "abc", "status": "succeeded", "model": MINIMAX_MODEL, "input": {"duration": 6, "resolution": "480P"}}
    )
    usage = ReplicateVideoConfig().transform_video_status_retrieve_response(response, Mock()).usage
    assert usage is not None
    cost = default_video_cost_calculator(
        f"replicate/{MINIMAX_MODEL}",
        usage["duration_seconds"],
        "replicate",
        video_resolution=usage["video_resolution"],
    )
    assert cost == pytest.approx(6 * row["output_cost_per_second_480p"])


def test_video_cost_uses_tiered_replicate_rows(local_model_cost_map: None):
    rows = {
        model: row
        for model, row in litellm.model_cost.items()
        if row.get("litellm_provider") == "replicate" and row.get("mode") == "video_generation"
    }
    assert rows
    for model, row in rows.items():
        for key, value in row.items():
            if key.startswith("output_cost_per_second_") and value is not None:
                tier = key.removeprefix("output_cost_per_second_")
                assert default_video_cost_calculator(model, 5, "replicate", video_resolution=tier) == pytest.approx(
                    5 * value
                )
