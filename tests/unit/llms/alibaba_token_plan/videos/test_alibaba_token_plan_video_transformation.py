import base64
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import partial
from io import BytesIO
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.alibaba_token_plan.common_utils import DEFAULT_API_BASE, VIDEO_ENDPOINT
from litellm.llms.alibaba_token_plan.videos.transformation import AlibabaTokenPlanVideoConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx import http_handler as http_handler_module
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler, blocked_cookie_jar
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider

_BASE: Final = "https://gateway.example/token-plan"
_IMAGE: Final = b"\x89PNG\r\n\x1a\nimage"
_VIDEO: Final = b"\x00\x00\x00\x18ftypmp42-video"
_MODELS: Final = ("happyhorse-1.1-t2v", "happyhorse-1.1-i2v", "happyhorse-1.1-r2v")


class _InjectedAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client


def _set_sync_download_client(monkeypatch: pytest.MonkeyPatch, client: httpx.Client) -> None:
    handler: Final = HTTPHandler(client=client)
    monkeypatch.setattr(litellm, "module_level_client", handler)


def _set_async_download_client(monkeypatch: pytest.MonkeyPatch, client: httpx.AsyncClient) -> None:
    handler: Final = _InjectedAsyncHTTPHandler(client)
    monkeypatch.setattr(
        http_handler_module,
        "get_async_httpx_client",
        lambda llm_provider, params=None, shared_session=None: handler,
    )


def _logging_obj() -> Logging:
    return Logging(
        model=_MODELS[0],
        messages=[],
        stream=False,
        call_type="create_video",
        start_time=datetime(2026, 10, 2, tzinfo=timezone.utc),
        litellm_call_id="test-call",
        function_id="test-function",
    )


def _video_response(request: httpx.Request, model: str) -> httpx.Response:
    if request.headers.get("Host") == "video.example":
        assert request.method == "GET"
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers
        return httpx.Response(200, content=_VIDEO, headers={"Content-Type": "video/mp4"})
    assert request.headers["Authorization"] == "Bearer explicit-key"
    if request.method == "POST":
        assert request.headers["X-DashScope-Async"] == "enable"
        assert str(request.url) == f"{_BASE}/{VIDEO_ENDPOINT}"
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content) == {
            "model": model,
            "input": {
                "prompt": "A slowly moving cloud",
                **(
                    {
                        "media": [
                            {"type": "first_frame", "url": "data:image/png;base64," + base64.b64encode(_IMAGE).decode()}
                        ]
                    }
                    if model.endswith("-i2v")
                    else {}
                ),
                **(
                    {"media": [{"type": "reference_image", "url": "https://images.example/subject.png"}]}
                    if model.endswith("-r2v")
                    else {}
                ),
            },
            "parameters": {"duration": 3, "resolution": "480P", "watermark": False},
        }
        return httpx.Response(200, json={"output": {"task_id": "task-test", "task_status": "PENDING"}})
    assert request.method == "GET"
    assert "X-DashScope-Async" not in request.headers
    assert str(request.url) == f"{_BASE}/api/v1/tasks/task-test"
    return httpx.Response(
        200,
        json={
            "output": {
                "task_id": "task-test",
                "task_status": "SUCCEEDED",
                "video_url": "https://video.example/video.mp4?signature=secret",
                "submit_time": "2026-10-02 12:00:00.000",
                "end_time": "2026-10-02 12:00:30.000",
            },
            "usage": {"output_video_duration": 3, "SR": 480, "video_count": 1},
        },
    )


def _extra_body(model: str) -> dict[str, object]:  # mutable-ok: public API accepts a dictionary
    return {
        "parameters": {"watermark": False},
        **(
            {"input": {"media": [{"type": "reference_image", "url": "https://images.example/subject.png"}]}}
            if model.endswith("-r2v")
            else {}
        ),
    }


def _assert_completed(created: VideoObject, completed: VideoObject, content: bytes) -> None:
    assert created.status == "queued"
    assert decode_video_id_with_provider(created.id)["custom_llm_provider"] == "alibaba_token_plan"
    assert completed.id == created.id
    assert completed.status == "completed"
    assert completed.progress == 100
    assert completed.seconds == "3"
    assert completed.usage == {"duration_seconds": 3, "video_resolution": "480p", "video_count": 1}
    assert completed.created_at == int(datetime(2026, 10, 2, 4, 0, tzinfo=timezone.utc).timestamp())
    assert completed.completed_at == completed.created_at + 30
    assert completed.expires_at is None
    assert content == _VIDEO


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("path", ["", "/compatible-mode/v1", "/apps/anthropic", f"/{VIDEO_ENDPOINT}"])
def test_public_sync_create_poll_and_download(model: str, path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    transport: Final = httpx.MockTransport(partial(_video_response, model=model))
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        handler: Final = HTTPHandler(client=client)
        created: Final = litellm.video_generation(
            model=f"alibaba_token_plan/{model}",
            prompt="A slowly moving cloud",
            seconds="3",
            resolution="480P",
            input_reference=BytesIO(_IMAGE) if model.endswith("-i2v") else None,
            extra_body=_extra_body(model),
            api_key="explicit-key",
            api_base=f"{_BASE}{path}",
            client=handler,
        )
        completed: Final = litellm.video_status(
            created.id, api_key="explicit-key", api_base=f"{_BASE}{path}", client=handler
        )
        content: Final = litellm.video_content(
            created.id, api_key="explicit-key", api_base=f"{_BASE}{path}", client=handler
        )
    assert isinstance(content, bytes)
    _assert_completed(created, completed, content)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", _MODELS)
async def test_public_async_create_poll_and_download(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    transport: Final = httpx.MockTransport(partial(_video_response, model=model))
    async with (
        httpx.AsyncClient(transport=transport) as client,
        httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_async_download_client(monkeypatch, download_client)
        handler: Final = _InjectedAsyncHTTPHandler(client)
        created: Final = await litellm.avideo_generation(
            model=f"alibaba_token_plan/{model}",
            prompt="A slowly moving cloud",
            seconds="3",
            resolution="480P",
            input_reference=("first.png", _IMAGE, "image/png") if model.endswith("-i2v") else None,
            extra_body=_extra_body(model),
            api_key="explicit-key",
            api_base=_BASE,
            client=handler,
        )
        completed: Final = await litellm.avideo_status(
            created.id, api_key="explicit-key", api_base=_BASE, client=handler
        )
        content: Final = await litellm.avideo_content(
            created.id, api_key="explicit-key", api_base=_BASE, client=handler
        )
    _assert_completed(created, completed, content)


def _default_status_response(request: httpx.Request) -> httpx.Response:
    if request.headers.get("Host") == "video.example":
        return httpx.Response(200, content=_VIDEO, headers={"Content-Type": "video/mp4"})
    assert request.method == "GET"
    assert str(request.url) == f"{DEFAULT_API_BASE}/api/v1/tasks/task-test"
    assert request.headers["Authorization"] == "Bearer explicit-key"
    return httpx.Response(
        200,
        json={
            "output": {
                "task_id": "task-test",
                "task_status": "SUCCEEDED",
                "video_url": "https://video.example/video.mp4",
            }
        },
    )


def test_public_status_and_content_use_default_native_task_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    video_id: Final = encode_video_id_with_provider("task-test", "alibaba_token_plan", _MODELS[0])
    transport: Final = httpx.MockTransport(_default_status_response)
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        handler: Final = HTTPHandler(client=client)
        status: Final = litellm.video_status(video_id, api_key="explicit-key", client=handler)
        content: Final = litellm.video_content(video_id, api_key="explicit-key", client=handler)
    assert status.status == "completed"
    assert content == _VIDEO


@pytest.mark.asyncio
async def test_public_async_status_and_content_use_default_native_task_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id: Final = encode_video_id_with_provider("task-test", "alibaba_token_plan", _MODELS[0])
    transport: Final = httpx.MockTransport(_default_status_response)
    async with (
        httpx.AsyncClient(transport=transport) as client,
        httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_async_download_client(monkeypatch, download_client)
        handler: Final = _InjectedAsyncHTTPHandler(client)
        status: Final = await litellm.avideo_status(video_id, api_key="explicit-key", client=handler)
        content: Final = await litellm.avideo_content(video_id, api_key="explicit-key", client=handler)
    assert status.status == "completed"
    assert content == _VIDEO


@pytest.mark.parametrize(
    "status,expected",
    [
        ("PENDING", "queued"),
        ("RUNNING", "in_progress"),
        ("FAILED", "failed"),
        ("CANCELED", "failed"),
        ("UNKNOWN", "failed"),
    ],
)
def test_task_status_and_error_mapping(status: str, expected: str) -> None:
    result: Final = AlibabaTokenPlanVideoConfig().transform_video_status_retrieve_response(
        httpx.Response(
            200,
            json={
                "output": {
                    "task_id": "task-test",
                    "task_status": status,
                    "code": "ProviderError",
                    "message": "Task failed",
                }
            },
        ),
        _logging_obj(),
    )
    assert result.status == expected
    if expected == "failed":
        assert result.error == {"code": "ProviderError", "message": "Task failed"}
    else:
        assert result.error is None


def test_aware_video_timestamps_preserve_their_offset() -> None:
    result: Final = AlibabaTokenPlanVideoConfig().transform_video_status_retrieve_response(
        httpx.Response(
            200,
            json={
                "output": {
                    "task_id": "task-test",
                    "task_status": "SUCCEEDED",
                    "submit_time": "2026-10-02T04:00:00+00:00",
                    "end_time": "2026-10-02T04:00:30+00:00",
                }
            },
        ),
        _logging_obj(),
    )
    assert result.created_at == int(datetime(2026, 10, 2, 4, 0, tzinfo=timezone.utc).timestamp())
    assert result.completed_at == result.created_at + 30


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"code": "InvalidApiKey", "message": "bad key"}, 502),
        ({"output": {"task_status": "SUCCEEDED"}}, 502),
        ({"output": {"task_id": "task-test", "task_status": "RUNNING"}}, 409),
        ({"output": {"task_id": "task-test", "task_status": "SUCCEEDED"}}, 502),
    ],
)
def test_content_rejects_errors_missing_url_and_incomplete_tasks(payload: Mapping[str, object], status: int) -> None:
    with pytest.raises(BaseLLMException) as error:
        AlibabaTokenPlanVideoConfig().transform_video_content_response(
            httpx.Response(200, json=payload),
            _logging_obj(),
        )
    assert error.value.status_code == status


@pytest.mark.parametrize(
    "model,params,message",
    [
        (_MODELS[1], {}, "exactly one first_frame"),
        (_MODELS[2], {}, "1 to 9 reference_image"),
        (_MODELS[0], {"input_reference": "https://images.example/image.png"}, "does not support reference"),
        (
            _MODELS[1],
            {"input_reference": "https://images.example/image.png", "size": "1280x720"},
            "preserves the image aspect ratio",
        ),
        (_MODELS[0], {"parameters": {"duration": 2}}, "greater than or equal"),
        (_MODELS[0], {"parameters": {"resolution": "4K"}}, "Input should be"),
        (_MODELS[0], {"input_reference": b"not an image"}, "JPEG, PNG, or WebP"),
    ],
)
def test_invalid_video_requests_are_rejected(model: str, params: Mapping[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AlibabaTokenPlanVideoConfig().transform_video_create_request(
            model, "A cloud", _BASE, params, GenericLiteLLMParams(), {}
        )


def test_openai_size_and_seconds_map_without_changing_native_parameter_precedence() -> None:
    request, files, url = AlibabaTokenPlanVideoConfig().transform_video_create_request(
        _MODELS[0],
        "A cloud",
        _BASE,
        {"seconds": "6", "size": "1280x720", "parameters": {"duration": 7, "watermark": False}},
        GenericLiteLLMParams(),
        {},
    )
    assert request == {
        "model": _MODELS[0],
        "input": {"prompt": "A cloud"},
        "parameters": {"duration": 7, "resolution": "720P", "ratio": "16:9", "watermark": False},
    }
    assert files == {}
    assert url == f"{_BASE}/{VIDEO_ENDPOINT}"


def _credential_response(request: httpx.Request) -> httpx.Response:
    if request.headers.get("Host") == "gateway.example":
        assert request.headers["Authorization"]
        return httpx.Response(
            200,
            json={
                "output": {
                    "task_id": "task-test",
                    "task_status": "SUCCEEDED",
                    "video_url": "https://video.example/start?signature=signed",
                }
            },
        )
    assert "authorization" not in request.headers
    assert "x-api-key" not in request.headers
    assert "cookie" not in request.headers
    assert "api_key" not in request.url.params
    if request.url.path == "/start":
        assert request.url.params["signature"] == "signed"
        return httpx.Response(302, headers={"location": "https://cdn.example/final"})
    assert request.headers["Host"] == "cdn.example"
    assert request.url.path == "/final"
    return httpx.Response(200, content=_VIDEO)


def test_content_download_and_redirect_never_forward_client_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    transport: Final = httpx.MockTransport(_credential_response)
    with (
        httpx.Client(
            transport=transport,
            headers={"Authorization": "Bearer secret", "x-api-key": "secret", "Cookie": "secret=value"},
            auth=("user", "password"),
            cookies={"cookie": "secret"},
            params={"api_key": "secret"},
        ) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        result: Final = litellm.video_content(
            encode_video_id_with_provider("task-test", "alibaba_token_plan", _MODELS[0]),
            api_key="explicit-key",
            api_base=_BASE,
            client=HTTPHandler(client=client),
        )
    assert result == _VIDEO


@pytest.mark.asyncio
async def test_async_content_download_and_redirect_never_forward_client_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport: Final = httpx.MockTransport(_credential_response)
    async with (
        httpx.AsyncClient(
            transport=transport,
            headers={"Authorization": "Bearer secret", "x-api-key": "secret", "Cookie": "secret=value"},
            auth=("user", "password"),
            cookies={"cookie": "secret"},
            params={"api_key": "secret"},
        ) as client,
        httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_async_download_client(monkeypatch, download_client)
        result: Final = await litellm.avideo_content(
            encode_video_id_with_provider("task-test", "alibaba_token_plan", _MODELS[0]),
            api_key="explicit-key",
            api_base=_BASE,
            client=_InjectedAsyncHTTPHandler(client),
        )
    assert result == _VIDEO


def _download_error_response(request: httpx.Request) -> httpx.Response:
    if request.headers.get("Host") == "gateway.example":
        return httpx.Response(
            200,
            json={
                "output": {
                    "task_id": "task-test",
                    "task_status": "SUCCEEDED",
                    "video_url": "https://video.example/expired.mp4?X-Amz-Credential=temporary&X-Amz-Signature=secret-value",
                }
            },
        )
    return httpx.Response(403, content=b"Video download expired")


def test_malformed_video_url_is_not_exposed() -> None:
    secret: Final = "sensitive-presigned-value"
    with pytest.raises(BaseLLMException, match="Invalid Alibaba Token Plan video response") as error:
        AlibabaTokenPlanVideoConfig().transform_video_content_response(
            httpx.Response(
                200,
                json={
                    "output": {
                        "task_id": "task-test",
                        "task_status": "SUCCEEDED",
                        "video_url": f"not-a-url?Signature={secret}",
                    }
                },
            ),
            _logging_obj(),
        )
    assert secret not in str(error.value)


def test_public_video_content_propagates_expired_download_error(monkeypatch: pytest.MonkeyPatch) -> None:
    transport: Final = httpx.MockTransport(_download_error_response)
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        with pytest.raises(litellm.APIError, match="video download failed") as error:
            litellm.video_content(
                encode_video_id_with_provider("task-test", "alibaba_token_plan", _MODELS[0]),
                api_key="explicit-key",
                api_base=_BASE,
                client=HTTPHandler(client=client),
            )
    assert "secret-value" not in str(error.value)
    assert "X-Amz-Signature" not in str(error.value)


@pytest.mark.asyncio
async def test_public_async_video_content_propagates_expired_download_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport: Final = httpx.MockTransport(_download_error_response)
    async with (
        httpx.AsyncClient(transport=transport) as client,
        httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_async_download_client(monkeypatch, download_client)
        with pytest.raises(litellm.APIError, match="video download failed") as error:
            await litellm.avideo_content(
                encode_video_id_with_provider("task-test", "alibaba_token_plan", _MODELS[0]),
                api_key="explicit-key",
                api_base=_BASE,
                client=_InjectedAsyncHTTPHandler(client),
            )
    assert "secret-value" not in str(error.value)
    assert "X-Amz-Signature" not in str(error.value)


def _gateway_query_response(request: httpx.Request) -> httpx.Response:
    if request.headers.get("Host") == "video.example":
        assert request.url.params == httpx.QueryParams("signature=signed")
        return httpx.Response(200, content=_VIDEO)
    assert str(request.url) == f"{_BASE}/api/v1/tasks/task-test?tenant=test"
    return httpx.Response(
        200,
        json={
            "output": {
                "task_id": "task-test",
                "task_status": "SUCCEEDED",
                "video_url": "https://video.example/video.mp4?signature=signed",
            }
        },
    )


def test_public_status_and_content_preserve_gateway_query(monkeypatch: pytest.MonkeyPatch) -> None:
    video_id: Final = encode_video_id_with_provider("task-test", "alibaba_token_plan", _MODELS[0])
    transport: Final = httpx.MockTransport(_gateway_query_response)
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        handler: Final = HTTPHandler(client=client)
        status: Final = litellm.video_status(video_id, api_key="key", api_base=f"{_BASE}?tenant=test", client=handler)
        content: Final = litellm.video_content(video_id, api_key="key", api_base=f"{_BASE}?tenant=test", client=handler)
    assert status.id == video_id
    assert status.status == "completed"
    assert content == _VIDEO


@pytest.mark.asyncio
async def test_public_async_status_and_content_preserve_gateway_query(monkeypatch: pytest.MonkeyPatch) -> None:
    video_id: Final = encode_video_id_with_provider("task-test", "alibaba_token_plan", _MODELS[0])
    transport: Final = httpx.MockTransport(_gateway_query_response)
    async with (
        httpx.AsyncClient(transport=transport) as client,
        httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_async_download_client(monkeypatch, download_client)
        handler: Final = _InjectedAsyncHTTPHandler(client)
        status: Final = await litellm.avideo_status(
            video_id, api_key="key", api_base=f"{_BASE}?tenant=test", client=handler
        )
        content: Final = await litellm.avideo_content(
            video_id, api_key="key", api_base=f"{_BASE}?tenant=test", client=handler
        )
    assert status.id == video_id
    assert status.status == "completed"
    assert content == _VIDEO


@pytest.mark.parametrize("spooled", [False, True])
def test_video_reference_retries_read_full_file_and_preserve_cursor(spooled: bool) -> None:
    reference: Final = SpooledTemporaryFile(mode="w+b") if spooled else BytesIO()
    reference.write(_IMAGE)
    reference.seek(4)
    params: Final = {"input_reference": ("frame.png", reference, "image/png")}
    config: Final = AlibabaTokenPlanVideoConfig()
    first, _, _ = config.transform_video_create_request(
        _MODELS[1], "A cloud", _BASE, params, GenericLiteLLMParams(), {}
    )
    assert reference.tell() == 4
    second, _, _ = config.transform_video_create_request(
        _MODELS[1], "A cloud", _BASE, params, GenericLiteLLMParams(), {}
    )
    assert reference.tell() == 4
    assert (
        first
        == second
        == {
            "model": _MODELS[1],
            "input": {
                "prompt": "A cloud",
                "media": [{"type": "first_frame", "url": "data:image/png;base64," + base64.b64encode(_IMAGE).decode()}],
            },
            "parameters": {},
        }
    )
    reference.close()


def test_invalid_video_reference_preserves_cursor_on_error() -> None:
    reference: Final = BytesIO(b"not an image")
    reference.seek(3)
    with pytest.raises(ValueError, match="JPEG, PNG, or WebP"):
        AlibabaTokenPlanVideoConfig().transform_video_create_request(
            _MODELS[1], "A cloud", _BASE, {"input_reference": reference}, GenericLiteLLMParams(), {}
        )
    assert reference.tell() == 3


def test_video_reference_accepts_pathlike_without_treating_strings_as_paths(tmp_path: Path) -> None:
    reference: Final = tmp_path / "frame.png"
    reference.write_bytes(_IMAGE)
    config: Final = AlibabaTokenPlanVideoConfig()
    result, _, _ = config.transform_video_create_request(
        _MODELS[1], "A cloud", _BASE, {"input_reference": reference}, GenericLiteLLMParams(), {}
    )
    assert result["input"] == {
        "prompt": "A cloud",
        "media": [{"type": "first_frame", "url": "data:image/png;base64," + base64.b64encode(_IMAGE).decode()}],
    }
    with pytest.raises(ValueError, match="public HTTP"):
        config.transform_video_create_request(
            _MODELS[1], "A cloud", _BASE, {"input_reference": str(reference)}, GenericLiteLLMParams(), {}
        )


def _blocked_video_response(request: httpx.Request, redirect: bool) -> httpx.Response:
    if request.headers.get("Host") == "gateway.example":
        return httpx.Response(
            200,
            json={
                "output": {
                    "task_id": "task-test",
                    "task_status": "SUCCEEDED",
                    "video_url": "https://video.example/start" if redirect else "http://169.254.169.254/metadata",
                }
            },
        )
    assert redirect and request.url.host == "video.example", "Blocked destination reached the transport"
    return httpx.Response(302, headers={"Location": "http://169.254.169.254/metadata"})


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("redirect", [False, True])
async def test_video_download_blocks_private_destinations_before_transport(
    use_async: bool, redirect: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport: Final = httpx.MockTransport(partial(_blocked_video_response, redirect=redirect))
    video_id: Final = encode_video_id_with_provider("task-test", "alibaba_token_plan", _MODELS[0])
    if use_async:
        async with (
            httpx.AsyncClient(transport=transport) as async_client,
            httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
        ):
            _set_async_download_client(monkeypatch, download_client)
            with pytest.raises(litellm.InternalServerError, match="blocked address"):
                await litellm.avideo_content(
                    video_id, api_key="explicit-key", api_base=_BASE, client=_InjectedAsyncHTTPHandler(async_client)
                )
        return
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        _set_sync_download_client(monkeypatch, download_client)
        with pytest.raises(litellm.InternalServerError, match="blocked address"):
            litellm.video_content(video_id, api_key="explicit-key", api_base=_BASE, client=HTTPHandler(client=client))
