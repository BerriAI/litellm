import base64
import time
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPResponseLimitError
from litellm.constants import REPLICATE_MAX_OUTPUT_DOWNLOAD_BYTES
from litellm.litellm_core_utils.url_utils import SSRFError
from litellm.llms.replicate import common_utils
from litellm.llms.replicate.common_utils import ReplicateError
from litellm.llms.replicate.image_generation.transformation import ReplicateImageGenerationConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

API_BASE: Final = "https://api.replicate.com/v1"
MODEL: Final = "black-forest-labs/flux-schnell"
PREDICTIONS_URL: Final = f"{API_BASE}/models/{MODEL}/predictions"
IMAGE_URLS: Final = ("https://replicate.delivery/out/0.webp", "https://replicate.delivery/out/1.webp")


@pytest.fixture(autouse=True)
def _skip_url_dns_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)


class _FakeClient:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get(
        self, url: str, headers: dict[str, str] | None = None, params: object = None, **kwargs: object
    ) -> httpx.Response:
        self.calls.append((url, headers))
        return self.responses.pop(0)


class _FakeAsyncClient(_FakeClient):
    async def get(
        self, url: str, headers: dict[str, str] | None = None, params: object = None, **kwargs: object
    ) -> httpx.Response:  # pyright: ignore[reportIncompatibleMethodOverride]  # async twin of the sync fake
        return super().get(url, headers=headers, params=params, **kwargs)


def _prediction_response(body: dict[str, object], status_code: int = 201) -> httpx.Response:
    request = httpx.Request(
        "POST",
        PREDICTIONS_URL,
        headers={"Authorization": "Bearer k", "Prefer": "wait", "Content-Type": "application/json"},
    )
    return httpx.Response(status_code, json=body, request=request)


def _download(url: str, content: bytes) -> httpx.Response:
    return httpx.Response(200, content=content, request=httpx.Request("GET", url))


def _config(client: _FakeClient) -> ReplicateImageGenerationConfig:
    return ReplicateImageGenerationConfig(sync_client_factory=lambda _ssl_verify: client)  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler


def _transform(
    config: ReplicateImageGenerationConfig, response: httpx.Response, optional_params: dict[str, object] | None = None
):
    return config.transform_image_generation_response(
        model=MODEL,
        raw_response=response,
        model_response=litellm.ImageResponse(),
        logging_obj=Mock(),
        request_data={},
        optional_params=optional_params or {},
        litellm_params={},
        encoding=None,
    )


def test_provider_config_is_registered():
    config = ProviderConfigManager.get_provider_image_generation_config(model=MODEL, provider=LlmProviders.REPLICATE)
    assert isinstance(config, ReplicateImageGenerationConfig)


def test_map_openai_params_maps_n_and_size():
    mapped = ReplicateImageGenerationConfig().map_openai_params(
        {"n": 2, "size": "1024x1024", "response_format": "b64_json"}, {}, MODEL, False
    )
    assert mapped == {"num_outputs": 2, "aspect_ratio": "1:1", "response_format": "b64_json"}
    assert ReplicateImageGenerationConfig().map_openai_params({"size": "1792x1024"}, {}, MODEL, False) == {
        "aspect_ratio": "7:4"
    }


def test_map_openai_params_rejects_or_drops_unsupported():
    with pytest.raises(ValueError, match="quality"):
        ReplicateImageGenerationConfig().map_openai_params({"quality": "hd"}, {}, MODEL, False)
    assert ReplicateImageGenerationConfig().map_openai_params({"quality": "hd", "n": 1}, {}, MODEL, True) == {
        "num_outputs": 1
    }


def test_validate_environment_sets_bearer_and_prefer_wait():
    headers = ReplicateImageGenerationConfig().validate_environment({}, MODEL, [], {}, {}, api_key="r8_key")
    assert headers["Authorization"] == "Bearer r8_key"
    assert headers["Prefer"] == "wait"


def test_request_url_and_body():
    config = ReplicateImageGenerationConfig()
    assert config.get_complete_url(None, None, MODEL, {}, {}) == PREDICTIONS_URL
    assert config.get_complete_url("https://proxy.internal", None, "owner/name:v1", {}, {}) == (
        "https://proxy.internal/v1/models/owner/name/versions/v1/predictions"
    )
    body = config.transform_image_generation_request(
        MODEL, "a red fox", {"num_outputs": 2, "aspect_ratio": "1:1", "response_format": "url"}, {}, {}
    )
    assert body == {"input": {"prompt": "a red fox", "num_outputs": 2, "aspect_ratio": "1:1"}}
    versioned = config.transform_image_generation_request("owner/name:v1", "a red fox", {}, {}, {})
    assert versioned == {"input": {"prompt": "a red fox"}}


@pytest.mark.parametrize("output", [list(IMAGE_URLS), IMAGE_URLS[0]])
def test_finished_prediction_returns_urls_without_downloading(output: object):
    client = _FakeClient()
    result = _transform(_config(client), _prediction_response({"id": "p1", "status": "succeeded", "output": output}))
    assert [image.url for image in result.data] == (list(IMAGE_URLS) if isinstance(output, list) else [IMAGE_URLS[0]])
    assert all(image.b64_json is None for image in result.data)
    assert client.calls == []


def test_b64_json_downloads_each_output():
    client = _FakeClient(_download(IMAGE_URLS[0], b"img0"), _download(IMAGE_URLS[1], b"img1"))
    result = _transform(
        _config(client),
        _prediction_response({"id": "p1", "status": "succeeded", "output": list(IMAGE_URLS)}),
        {"response_format": "b64_json"},
    )
    assert [image.b64_json for image in result.data] == [
        base64.b64encode(b"img0").decode(),
        base64.b64encode(b"img1").decode(),
    ]
    assert all(image.url is None for image in result.data)
    assert [url for url, _ in client.calls] == list(IMAGE_URLS)


def test_unfinished_prediction_is_polled_until_output(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    poll_url = f"{API_BASE}/predictions/p1"
    client = _FakeClient(
        httpx.Response(
            200, json={"id": "p1", "status": "processing", "output": None}, request=httpx.Request("GET", poll_url)
        ),
        httpx.Response(
            200,
            json={"id": "p1", "status": "succeeded", "output": [IMAGE_URLS[0]]},
            request=httpx.Request("GET", poll_url),
        ),
    )
    pending = _prediction_response({"id": "p1", "status": "starting", "output": None, "urls": {"get": poll_url}})
    result = _transform(_config(client), pending)
    assert [image.url for image in result.data] == [IMAGE_URLS[0]]
    assert [url for url, _ in client.calls] == [poll_url, poll_url]
    poll_headers = client.calls[0][1]
    assert poll_headers is not None
    assert poll_headers["authorization"] == "Bearer k"
    assert "prefer" not in poll_headers


def test_partial_output_keeps_polling_until_terminal_status(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    poll_url = f"{API_BASE}/predictions/p1"
    client = _FakeClient(
        httpx.Response(
            200,
            json={"id": "p1", "status": "succeeded", "output": list(IMAGE_URLS)},
            request=httpx.Request("GET", poll_url),
        ),
    )
    partial = _prediction_response(
        {"id": "p1", "status": "processing", "output": [IMAGE_URLS[0]], "urls": {"get": poll_url}}
    )
    result = _transform(_config(client), partial)
    assert [image.url for image in result.data] == list(IMAGE_URLS)


def test_polling_gives_up_after_retry_budget(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(common_utils, "DEFAULT_REPLICATE_SYNC_POLLING_RETRIES", 2)
    poll_url = f"{API_BASE}/predictions/p1"
    still_running = {"id": "p1", "status": "processing", "output": None}
    client = _FakeClient(
        *(httpx.Response(200, json=still_running, request=httpx.Request("GET", poll_url)) for _ in range(2))
    )
    pending = _prediction_response({"id": "p1", "status": "starting", "output": None, "urls": {"get": poll_url}})
    with pytest.raises(ReplicateError, match="did not finish after 2 polls"):
        _transform(_config(client), pending)


def test_poll_http_error_raises_instead_of_retrying(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    poll_url = f"{API_BASE}/predictions/p1"
    client = _FakeClient(
        httpx.Response(401, json={"detail": "Unauthenticated"}, request=httpx.Request("GET", poll_url)),
        httpx.Response(200, json={"id": "p1", "status": "succeeded", "output": IMAGE_URLS[0]}),
    )
    pending = _prediction_response({"id": "p1", "status": "starting", "output": None, "urls": {"get": poll_url}})
    with pytest.raises(ReplicateError, match="Unauthenticated") as excinfo:
        _transform(_config(client), pending)
    assert excinfo.value.status_code == 401
    assert len(client.calls) == 1


def test_polling_client_gets_deployment_ssl_verify():
    seen: list[object] = []

    def factory(ssl_verify: object) -> _FakeClient:
        seen.append(ssl_verify)
        return _FakeClient()

    config = ReplicateImageGenerationConfig(sync_client_factory=factory)  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    config.transform_image_generation_response(
        model=MODEL,
        raw_response=_prediction_response({"id": "p1", "status": "succeeded", "output": [IMAGE_URLS[0]]}),
        model_response=litellm.ImageResponse(),
        logging_obj=Mock(),
        request_data={},
        optional_params={},
        litellm_params={"ssl_verify": "/etc/ssl/gateway.pem"},
        encoding=None,
    )
    assert seen == ["/etc/ssl/gateway.pem"]


def test_output_urls_keep_nested_order_without_recursion():
    nested = [[IMAGE_URLS[0]], "not a url", {"files": [IMAGE_URLS[1], {"deep": "https://cdn.example/3.png"}]}, 7]
    assert common_utils.replicate_output_urls(nested) == (IMAGE_URLS[0], IMAGE_URLS[1], "https://cdn.example/3.png")
    assert common_utils.replicate_output_urls("plain text") == ()


def test_request_flattens_extra_body_into_native_input():
    body = ReplicateImageGenerationConfig().transform_image_generation_request(
        MODEL,
        "a boat",
        {
            "num_outputs": 1,
            "extra_body": {"seed": 7, "num_outputs": 3, "prompt": "unaudited"},
            "response_format": "url",
        },
        {},
        {},
    )
    assert body == {"input": {"prompt": "a boat", "num_outputs": 3, "seed": 7}}


def test_poll_url_must_share_the_api_origin():
    pending = _prediction_response(
        {"id": "p1", "status": "starting", "output": None, "urls": {"get": "https://evil.example/predictions/p1"}}
    )
    with pytest.raises(SSRFError, match="Origin mismatch"):
        _transform(_config(_FakeClient()), pending)


def test_b64_download_blocks_internal_urls(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "user_url_validation", True)
    done = _prediction_response({"id": "p1", "status": "succeeded", "output": ["http://10.0.0.8/secret.png"]})
    with pytest.raises(SSRFError):
        _transform(_config(_FakeClient()), done, {"response_format": "b64_json"})


def test_failed_prediction_raises_with_replicate_error_text():
    with pytest.raises(ReplicateError, match="NSFW content detected"):
        _transform(
            _config(_FakeClient()),
            _prediction_response({"id": "p1", "status": "failed", "error": "NSFW content detected"}),
        )


@pytest.mark.asyncio
async def test_async_transform_downloads_b64_json_with_async_client():
    client = _FakeAsyncClient(_download(IMAGE_URLS[0], b"img0"))
    config = ReplicateImageGenerationConfig(async_client_factory=lambda _ssl_verify: client)  # pyright: ignore[reportArgumentType]  # fake stands in for AsyncHTTPHandler
    result = await config.async_transform_image_generation_response(
        model=MODEL,
        raw_response=_prediction_response({"id": "p1", "status": "succeeded", "output": [IMAGE_URLS[0]]}),
        model_response=litellm.ImageResponse(),
        logging_obj=Mock(),
        request_data={},
        optional_params={"response_format": "b64_json"},
        litellm_params={},
        encoding=None,
    )
    assert [image.b64_json for image in result.data] == [base64.b64encode(b"img0").decode()]


def test_native_width_and_height_become_the_response_size():
    result = _transform(
        _config(_FakeClient()),
        _prediction_response(
            {
                "id": "p1",
                "status": "succeeded",
                "output": IMAGE_URLS[0],
                "input": {"prompt": "x", "aspect_ratio": "custom", "width": 2048, "height": 1536},
            }
        ),
    )
    assert result.size == "2048x1536"


@pytest.mark.parametrize(
    "prediction_input",
    [
        {"prompt": "x"},
        {"prompt": "x", "width": 1024},
        {"prompt": "x", "width": "1024", "height": "768"},
        {"width": 0, "height": 768},
    ],
)
def test_response_size_stays_unset_without_both_native_dimensions(prediction_input: dict[str, object]):
    result = _transform(
        _config(_FakeClient()),
        _prediction_response({"id": "p1", "status": "succeeded", "output": IMAGE_URLS[0], "input": prediction_input}),
    )
    assert result.size is None


def test_b64_download_is_bounded_by_the_output_byte_limit():
    seen: list[object] = []

    class _LimitAwareClient(_FakeClient):
        def get(
            self, url: str, headers: dict[str, str] | None = None, params: object = None, **kwargs: object
        ) -> httpx.Response:
            seen.append(kwargs.get("max_response_bytes"))
            return super().get(url, headers=headers, params=params, **kwargs)

    client = _LimitAwareClient(_download(IMAGE_URLS[0], b"png"))
    result = _transform(
        _config(client),
        _prediction_response({"id": "p1", "status": "succeeded", "output": IMAGE_URLS[0]}),
        {"response_format": "b64_json"},
    )
    assert result.data[0].b64_json == base64.b64encode(b"png").decode()
    assert seen == [REPLICATE_MAX_OUTPUT_DOWNLOAD_BYTES]


def test_oversized_b64_download_is_a_replicate_error():
    class _OverflowingClient(_FakeClient):
        def get(
            self, url: str, headers: dict[str, str] | None = None, params: object = None, **kwargs: object
        ) -> httpx.Response:
            raise HTTPResponseLimitError("Response exceeds the configured size limit")

    with pytest.raises(ReplicateError, match="REPLICATE_MAX_OUTPUT_DOWNLOAD_BYTES") as refused:
        _transform(
            _config(_OverflowingClient()),
            _prediction_response({"id": "p1", "status": "succeeded", "output": IMAGE_URLS[0]}),
            {"response_format": "b64_json"},
        )
    assert refused.value.status_code == 502


@pytest.mark.parametrize(
    "model",
    [
        "black-forest-labs/../xai/grok-imagine-image",
        "deployments/acme/../../models/xai/grok-imagine-image",
        "black-forest-labs//flux-schnell",
        "./flux-schnell",
    ],
)
def test_model_paths_with_dot_or_empty_segments_are_rejected(model: str):
    with pytest.raises(ReplicateError) as refused:
        common_utils.replicate_prediction_request("https://api.replicate.com/v1", model, {})
    assert refused.value.status_code == 400


def test_model_path_segments_are_url_encoded():
    request = common_utils.replicate_prediction_request("https://api.replicate.com/v1", "acme/odd name?x", {})
    assert request.url == "https://api.replicate.com/v1/models/acme/odd%20name%3Fx/predictions"


def test_a_pinned_version_stays_under_the_authorized_model_path():
    request = common_utils.replicate_prediction_request("https://api.replicate.com/v1", "acme/fox:ab12", {})
    assert request.url == "https://api.replicate.com/v1/models/acme/fox/versions/ab12/predictions"
    assert "version" not in request.body
    with pytest.raises(ReplicateError) as refused:
        common_utils.replicate_prediction_request("https://api.replicate.com/v1", "acme/fox:", {})
    assert refused.value.status_code == 400
