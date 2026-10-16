import base64
import json
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.alibaba_token_plan.common_utils import VIDEO_PATH
from litellm.llms.alibaba_token_plan.videos.transformation import AlibabaTokenPlanVideoConfig
from litellm.llms.custom_httpx import http_handler as http_handler_module
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler, blocked_cookie_jar
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider

GATEWAY: Final = "https://gateway.example/token-plan"
IMAGE: Final = b"\x89PNG\r\n\x1a\nimage"
VIDEO: Final = b"\x00\x00\x00\x18ftypmp42-video"
SIGNED_VIDEO_URL: Final = "https://video.example/video.mp4?OSSAccessKeyId=temporary&Signature=secret-value"


class _InjectedAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client


def _use_download_clients(
    monkeypatch: pytest.MonkeyPatch, sync_client: httpx.Client, async_client: httpx.AsyncClient
) -> None:
    monkeypatch.setattr(litellm, "module_level_client", HTTPHandler(client=sync_client))
    monkeypatch.setattr(
        http_handler_module,
        "get_async_httpx_client",
        lambda llm_provider, params=None, shared_session=None: _InjectedAsyncHTTPHandler(async_client),
    )


def _video_provider(model: str, download_status: int = 200) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host in ("video.example", "93.184.216.34"):
            assert "authorization" not in request.headers
            return httpx.Response(download_status, content=VIDEO, headers={"Content-Type": "video/mp4"})
        assert request.headers["Authorization"] == "Bearer explicit-key"
        if request.method == "POST":
            assert str(request.url) == f"{GATEWAY}/{VIDEO_PATH}"
            assert request.headers["X-DashScope-Async"] == "enable"
            media_type: Final = "first_frame" if model.endswith("-i2v") else "reference_image"
            assert json.loads(request.content) == {
                "model": model,
                "input": {
                    "prompt": "A slowly moving cloud",
                    "media": [{"type": media_type, "url": f"data:image/png;base64,{base64.b64encode(IMAGE).decode()}"}],
                },
                "parameters": {"resolution": "720P", "ratio": "16:9", "duration": 5, "watermark": False},
            }
            return httpx.Response(200, json={"output": {"task_id": "task-test", "task_status": "PENDING"}})
        assert str(request.url) == f"{GATEWAY}/api/v1/tasks/task-test"
        assert "X-DashScope-Async" not in request.headers
        return httpx.Response(
            200,
            json={
                "output": {"task_id": "task-test", "task_status": "SUCCEEDED", "video_url": SIGNED_VIDEO_URL},
                "usage": {"output_video_duration": 5, "SR": 720},
            },
        )

    return httpx.MockTransport(respond)


def _assert_completed(created: VideoObject, completed: VideoObject, content: bytes) -> None:
    assert created.status == "queued"
    assert decode_video_id_with_provider(created.id)["custom_llm_provider"] == "alibaba_token_plan"
    assert completed.id == created.id
    assert completed.status == "completed"
    assert completed.seconds == "5"
    assert completed.usage == {"duration_seconds": 5, "video_resolution": "720p"}
    assert content == VIDEO


@pytest.mark.parametrize("model", ["happyhorse-1.1-i2v", "happyhorse-1.1-r2v"])
def test_sync_video_create_poll_and_download(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    transport: Final = _video_provider(model)
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        monkeypatch.setattr(litellm, "module_level_client", HTTPHandler(client=download_client))
        handler: Final = HTTPHandler(client=client)
        base: Final = {"api_key": "explicit-key", "api_base": f"{GATEWAY}/compatible-mode/v1", "client": handler}
        created: Final = litellm.video_generation(
            model=f"alibaba_token_plan/{model}",
            prompt="A slowly moving cloud",
            seconds="5",
            size="1280x720",
            input_reference=("first.png", IMAGE, "image/png"),
            extra_body={"parameters": {"watermark": False}},
            **base,
        )
        completed: Final = litellm.video_status(created.id, **base)
        content: Final = litellm.video_content(created.id, **base)
    assert isinstance(content, bytes)
    _assert_completed(created, completed, content)


@pytest.mark.asyncio
async def test_async_video_create_poll_and_download(monkeypatch: pytest.MonkeyPatch) -> None:
    transport: Final = _video_provider("happyhorse-1.1-i2v")
    async with (
        httpx.AsyncClient(transport=transport) as client,
        httpx.AsyncClient(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        with httpx.Client(transport=transport) as unused_sync_client:
            _use_download_clients(monkeypatch, unused_sync_client, download_client)
        handler: Final = _InjectedAsyncHTTPHandler(client)
        base: Final = {"api_key": "explicit-key", "api_base": f"{GATEWAY}/compatible-mode/v1", "client": handler}
        created: Final = await litellm.avideo_generation(
            model="alibaba_token_plan/happyhorse-1.1-i2v",
            prompt="A slowly moving cloud",
            seconds="5",
            size="1280x720",
            input_reference=IMAGE,
            extra_body={"parameters": {"watermark": False}},
            **base,
        )
        completed: Final = await litellm.avideo_status(created.id, **base)
        content: Final = await litellm.avideo_content(created.id, **base)
    _assert_completed(created, completed, content)


@pytest.mark.parametrize(
    ("task_status", "expected"),
    [("PENDING", "queued"), ("RUNNING", "in_progress"), ("FAILED", "failed"), ("CANCELED", "failed")],
)
def test_task_status_and_error_mapping(task_status: str, expected: str) -> None:
    logging_obj: Final = Logging(
        model="happyhorse-1.1-t2v",
        messages=[],
        stream=False,
        call_type="create_video",
        start_time=datetime(2026, 10, 2, tzinfo=timezone.utc),
        litellm_call_id="test-call",
        function_id="test-function",
    )
    result: Final = AlibabaTokenPlanVideoConfig().transform_video_status_retrieve_response(
        httpx.Response(
            200,
            json={"output": {"task_id": "t", "task_status": task_status, "code": "Err", "message": "Task failed"}},
        ),
        logging_obj,
    )
    assert result.status == expected
    assert result.error == ({"code": "Err", "message": "Task failed"} if expected == "failed" else None)
    with pytest.raises(Exception, match="not available") as error:
        AlibabaTokenPlanVideoConfig().transform_video_content_response(
            httpx.Response(200, json={"output": {"task_id": "t", "task_status": task_status}}), logging_obj
        )
    assert getattr(error.value, "status_code", None) == 409


def test_video_download_failure_does_not_expose_the_signed_url(monkeypatch: pytest.MonkeyPatch) -> None:
    transport: Final = _video_provider("happyhorse-1.1-t2v", download_status=403)
    with (
        httpx.Client(transport=transport) as client,
        httpx.Client(transport=transport, cookies=blocked_cookie_jar()) as download_client,
    ):
        monkeypatch.setattr(litellm, "module_level_client", HTTPHandler(client=download_client))
        video_id: Final = encode_video_id_with_provider("task-test", "alibaba_token_plan", "happyhorse-1.1-t2v")
        with pytest.raises(Exception, match="video download failed") as error:
            litellm.video_content(
                video_id,
                api_key="explicit-key",
                api_base=f"{GATEWAY}/compatible-mode/v1",
                client=HTTPHandler(client=client),
            )
    assert "secret-value" not in str(error.value)
    assert "Signature" not in str(error.value)


def test_native_input_cannot_replace_the_guardrail_checked_prompt() -> None:
    request, _, _ = AlibabaTokenPlanVideoConfig().transform_video_create_request(
        model="happyhorse-1.1-t2v",
        prompt="checked prompt",
        api_base=GATEWAY,
        video_create_optional_request_params={"input": {"prompt": "unchecked prompt", "negative_prompt": "blur"}},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert request["input"] == {"prompt": "checked prompt", "negative_prompt": "blur"}
