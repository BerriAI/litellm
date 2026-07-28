import asyncio
import copy
import io
import tempfile
from types import SimpleNamespace
from typing import Any, Dict

import orjson
import pytest
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.image_endpoints import endpoints


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

    async def fake_post_call_response_headers_hook(**kwargs):
        return {"x-callback-test": "value"}

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        update_request_status=fake_update_request_status,
        post_call_failure_hook=fake_post_call_failure_hook,
        post_call_success_hook=fake_post_call_success_hook,
        post_call_response_headers_hook=fake_post_call_response_headers_hook,
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
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger
    )
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
    assert response.headers.get("x-callback-test") == "value"


@pytest.mark.asyncio
async def test_image_edit_converts_style_image_upload_to_bytesio(monkeypatch):
    """Bedrock Stability style-transfer needs a second image (``style_image``).

    It must be converted from a Starlette ``UploadFile`` to a synchronously
    readable file-like object before routing; a raw ``UploadFile`` has an async
    ``.read()`` that the downstream transform reads synchronously, stringifying
    the resulting coroutine into the base64 payload sent to Bedrock.
    """
    style_bytes = b"\x89PNG\r\n\x1a\nstyle-reference-bytes"
    spool = tempfile.SpooledTemporaryFile()
    spool.write(style_bytes)
    spool.seek(0)
    style_upload = UploadFile(file=spool, filename="style.png")

    captured: Dict[str, Any] = {}

    async def fake_base_process(self, **kwargs):
        captured["data"] = self.data
        return {"result": "ok"}

    monkeypatch.setattr(
        ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        fake_base_process,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/images/edits",
        "headers": [(b"content-type", b"application/json")],
    }
    body = orjson.dumps(
        {"prompt": "an apple", "model": "stability.stable-style-transfer"}
    )

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)

    result = await endpoints.image_edit_api(
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=UserAPIKeyAuth(),
        image=None,
        image_array=None,
        mask=None,
        mask_array=None,
        style_image=[style_upload],
        model="stability.stable-style-transfer",
    )

    assert result == {"result": "ok"}
    routed_style_image = captured["data"]["style_image"]
    assert isinstance(routed_style_image, list)
    buffer = routed_style_image[0]
    assert isinstance(buffer, io.BytesIO)
    assert buffer.read() == style_bytes
