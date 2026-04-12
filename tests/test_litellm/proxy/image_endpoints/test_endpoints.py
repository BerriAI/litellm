import asyncio
import copy
import io
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, call

import orjson
import pytest
from starlette.requests import Request
from starlette.responses import Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.image_endpoints import endpoints
from litellm.proxy.image_endpoints.endpoints import batch_to_bytesio, uploadfile_to_bytesio


@pytest.mark.asyncio
async def test_image_generation_prompt_rerouting(monkeypatch):
    """Ensure image prompts are exposed to guardrails and restored afterwards."""

    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_update_request_status(**_: Any) -> None:
        await asyncio.sleep(0)

    proxy_logger_calls: Dict[str, Any] = {}

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):  # type: ignore[override]
        proxy_logger_calls["pre_call_input"] = copy.deepcopy(data)
        modified = {
            **data,
            "messages": [
                {
                    "role": "user",
                    "content": "sanitized prompt",
                }
            ],
        }
        return modified

    async def fake_post_call_failure_hook(**_: Any) -> None:
        return None

    async def fake_post_call_success_hook(*, data, user_api_key_dict, response):
        return response

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        update_request_status=fake_update_request_status,
        post_call_failure_hook=fake_post_call_failure_hook,
        post_call_success_hook=fake_post_call_success_hook,
    )

    captured_route_request_data: Dict[str, Any] = {}

    async def fake_route_request(*, data, **kwargs):  # type: ignore[override]
        captured_route_request_data.update(data)

        async def _inner():
            class FakeResponse(dict):
                _hidden_params = {}

            return FakeResponse(result="ok")

        return _inner()

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/images/generations",
        "headers": [],
    }
    body = orjson.dumps({"prompt": "original prompt"})

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    response = Response()
    user_api_key = UserAPIKeyAuth()

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.add_litellm_data_to_request",
        fake_add_litellm_data_to_request,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.get_custom_headers",
        classmethod(lambda *args, **kwargs: {}),
    )
    monkeypatch.setattr(
        "litellm.proxy.image_endpoints.endpoints.route_request", fake_route_request
    )

    result = await endpoints.image_generation(
        request=request,
        fastapi_response=response,
        user_api_key_dict=user_api_key,
    )
    await asyncio.sleep(0)

    assert result == {"result": "ok"}
    pre_call_input = proxy_logger_calls["pre_call_input"]
    assert pre_call_input["messages"][0]["content"] == "original prompt"
    assert captured_route_request_data["prompt"] == "sanitized prompt"
    assert "messages" not in captured_route_request_data


# ---------------------------------------------------------------------------
# Tests for uploadfile_to_bytesio / batch_to_bytesio — memory-leak fixes
# ---------------------------------------------------------------------------

def _make_upload_file(data: bytes, filename: str = "test.png") -> MagicMock:
    """Return a mock UploadFile whose read/close methods are async."""
    upload = MagicMock()
    upload.read = AsyncMock(return_value=data)
    upload.close = AsyncMock()
    upload.filename = filename
    return upload


@pytest.mark.asyncio
async def test_uploadfile_to_bytesio_returns_correct_data():
    """BytesIO buffer must contain the exact bytes from the upload."""
    image_bytes = b"\x89PNG\r\n\x1a\nfakeimage"
    upload = _make_upload_file(image_bytes, filename="photo.png")

    buf = await uploadfile_to_bytesio(upload)

    assert isinstance(buf, io.BytesIO)
    assert buf.read() == image_bytes
    assert buf.name == "photo.png"


@pytest.mark.asyncio
async def test_uploadfile_to_bytesio_closes_upload_after_read():
    """UploadFile.close() must be called exactly once to release SpooledTemporaryFile."""
    upload = _make_upload_file(b"data")

    await uploadfile_to_bytesio(upload)

    upload.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_uploadfile_to_bytesio_closes_upload_on_read_error():
    """close() must still be called even when read() raises an exception."""
    upload = MagicMock()
    upload.read = AsyncMock(side_effect=RuntimeError("disk error"))
    upload.close = AsyncMock()
    upload.filename = "broken.png"

    with pytest.raises(RuntimeError, match="disk error"):
        await uploadfile_to_bytesio(upload)

    upload.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_batch_to_bytesio_none_returns_none():
    """None input should pass through without error."""
    assert await batch_to_bytesio(None) is None


@pytest.mark.asyncio
async def test_batch_to_bytesio_empty_list_returns_none():
    """Empty list is treated the same as None."""
    assert await batch_to_bytesio([]) is None


@pytest.mark.asyncio
async def test_batch_to_bytesio_closes_all_uploads():
    """Every UploadFile in the list must be closed after conversion."""
    uploads = [_make_upload_file(f"img{i}".encode()) for i in range(3)]

    buffers = await batch_to_bytesio(uploads)  # type: ignore[arg-type]

    assert buffers is not None
    assert len(buffers) == 3
    for upload in uploads:
        upload.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_batch_to_bytesio_preserves_filenames():
    """Each BytesIO must carry the filename from its source UploadFile."""
    names = ["a.png", "b.png", "c.png"]
    uploads = [_make_upload_file(b"data", filename=n) for n in names]

    buffers = await batch_to_bytesio(uploads)  # type: ignore[arg-type]

    assert buffers is not None
    assert [b.name for b in buffers] == names
