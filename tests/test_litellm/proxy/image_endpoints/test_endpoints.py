import asyncio
import copy
import logging
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any, Dict

import orjson
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.image_endpoints import endpoints
from litellm.proxy.route_llm_request import ProxyMissingRequiredParamError


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
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.get_custom_headers",
        classmethod(lambda *args, **kwargs: {}),
    )
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.route_request", fake_route_request)

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
async def test_image_generation__missing_required_param_is_400(monkeypatch):
    """image_generation()'s except block only special-cased HTTPException, so a
    ProxyMissingRequiredParamError (a ProxyException with code=400) fell into the
    `else` branch's `getattr(e, "status_code", 500)` and surfaced as a 500."""

    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):  # type: ignore[override]
        return data

    async def fake_post_call_failure_hook(**_: Any) -> None:
        return None

    async def fake_route_request(*, data, **kwargs):  # type: ignore[override]
        raise ProxyMissingRequiredParamError(route="/image/generations", param="prompt")

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        post_call_failure_hook=fake_post_call_failure_hook,
    )

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/images/generations",
        "headers": [],
    }
    body = orjson.dumps({"model": "dall-e-3"})

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
        "litellm.proxy.image_endpoints.endpoints.route_request", fake_route_request
    )

    with pytest.raises(ProxyException) as exc_info:
        await endpoints.image_generation(
            request=request,
            fastapi_response=response,
            user_api_key_dict=user_api_key,
        )

    assert exc_info.value.code == "400"
    assert exc_info.value.param == "prompt"


def _image_edit_client(monkeypatch, captured: Dict[str, Any]) -> TestClient:
    class CaptureProcessing:
        def __init__(self, data: Dict[str, Any]) -> None:
            captured.update(data)

        async def base_process_llm_request(self, **_: Any) -> Dict[str, Any]:
            return {"data": [{"b64_json": "aGk="}]}

    monkeypatch.setattr(endpoints, "ProxyBaseLLMRequestProcessing", CaptureProcessing)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)

    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth()
    return TestClient(app)


def test_image_edit_image_array_alias_is_not_forwarded(monkeypatch):
    """The documented `image[]` alias must reach the provider only as `image`."""
    captured: Dict[str, Any] = {}

    response = _image_edit_client(monkeypatch, captured).post(
        "/v1/images/edits",
        files={"image[]": ("tree.png", b"\x89PNG\r\n\x1a\ntree", "image/png")},
        data={"model": "gpt-image-1", "prompt": "add a hat"},
    )

    assert response.status_code == 200
    assert "image[]" not in captured
    assert [buffer.getvalue() for buffer in captured["image"]] == [b"\x89PNG\r\n\x1a\ntree"]
    assert [buffer.name for buffer in captured["image"]] == ["tree.png"]


def test_image_edit_mask_array_alias_is_not_forwarded(monkeypatch):
    """`mask[]` has the same shape as `image[]` and must be dropped the same way."""
    captured: Dict[str, Any] = {}

    response = _image_edit_client(monkeypatch, captured).post(
        "/v1/images/edits",
        files={
            "image": ("tree.png", b"\x89PNG\r\n\x1a\ntree", "image/png"),
            "mask[]": ("mask.png", b"\x89PNG\r\n\x1a\nmask", "image/png"),
        },
        data={"model": "gpt-image-1", "prompt": "add a hat"},
    )

    assert response.status_code == 200
    assert "mask[]" not in captured
    assert [buffer.getvalue() for buffer in captured["mask"]] == [b"\x89PNG\r\n\x1a\nmask"]
    assert [buffer.getvalue() for buffer in captured["image"]] == [b"\x89PNG\r\n\x1a\ntree"]


def test_image_edit_canonical_file_fields_still_reach_the_provider(monkeypatch):
    """Dropping the bracketed aliases must not touch the canonical fields."""
    captured: Dict[str, Any] = {}

    response = _image_edit_client(monkeypatch, captured).post(
        "/v1/images/edits",
        files={
            "image": ("tree.png", b"\x89PNG\r\n\x1a\ntree", "image/png"),
            "mask": ("mask.png", b"\x89PNG\r\n\x1a\nmask", "image/png"),
        },
        data={"model": "gpt-image-1", "prompt": "add a hat"},
    )

    assert response.status_code == 200
    assert [buffer.getvalue() for buffer in captured["image"]] == [b"\x89PNG\r\n\x1a\ntree"]
    assert [buffer.getvalue() for buffer in captured["mask"]] == [b"\x89PNG\r\n\x1a\nmask"]
    assert captured["prompt"] == "add a hat"


def test_image_edit_multipart_n_reaches_the_provider_as_an_int(monkeypatch):
    """A multipart `n` must not arrive as the string Starlette parsed it into."""
    captured: Dict[str, Any] = {}

    response = _image_edit_client(monkeypatch, captured).post(
        "/v1/images/edits",
        files={"image": ("tree.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        data={"model": "nova-canvas", "prompt": "add a hat", "n": "2", "size": "1024x1024"},
    )

    assert response.status_code == 200
    assert captured["n"] == 2
    assert isinstance(captured["n"], int)
    assert captured["size"] == "1024x1024"
    assert captured["prompt"] == "add a hat"


def test_image_edit_multipart_n_that_is_not_a_number_is_left_alone(monkeypatch):
    """An unparseable `n` still reaches the provider, which rejects it as before."""
    captured: Dict[str, Any] = {}

    response = _image_edit_client(monkeypatch, captured).post(
        "/v1/images/edits",
        files={"image": ("tree.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        data={"model": "nova-canvas", "prompt": "add a hat", "n": "two"},
    )

    assert response.status_code == 200
    assert captured["n"] == "two"


@pytest.mark.asyncio
async def test_a_model_the_router_cannot_serve_answers_an_openai_typed_error(monkeypatch: pytest.MonkeyPatch):
    """A bare HTTPException carries no type or param, so the tail used to ship the
    literal string "None" in both fields."""

    async def fake_add_litellm_data_to_request(**kwargs: object) -> object:
        return kwargs["data"]

    async def fake_pre_call_hook(
        *, user_api_key_dict: UserAPIKeyAuth, data: dict[str, object], call_type: str
    ) -> dict[str, object]:
        return data

    async def fake_post_call_failure_hook(**_: object) -> None:
        return None

    async def failing_route_request(**_: object) -> None:
        raise HTTPException(
            status_code=404, detail={"error": "image_generation: Invalid model name passed in model=dall-e-3"}
        )

    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", fake_add_litellm_data_to_request)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        SimpleNamespace(pre_call_hook=fake_pre_call_hook, post_call_failure_hook=fake_post_call_failure_hook),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.route_request", failing_route_request)

    body = orjson.dumps({"model": "dall-e-3", "prompt": "a lighthouse at dusk"})

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/v1/images/generations", "headers": []}, receive)

    with pytest.raises(ProxyException) as raised:
        await endpoints.image_generation(
            request=request, fastapi_response=Response(), user_api_key_dict=UserAPIKeyAuth()
        )

    assert (raised.value.type, raised.value.param, raised.value.code) == ("invalid_request_error", None, "404")


@pytest.fixture
def propagating_proxy_logger() -> Iterator[None]:
    verbose_proxy_logger.propagate = True
    try:
        yield
    finally:
        verbose_proxy_logger.propagate = False


@pytest.mark.asyncio
async def test_failure_log_carries_the_callers_litellm_call_id(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, propagating_proxy_logger: None
) -> None:
    """LIT-7836: the /v1/images/generations error line must carry the litellm_call_id
    the client sent, both rendered in the message and as a structured record field."""
    call_id = "images-call-7836"

    async def fake_add_litellm_data_to_request(**kwargs: object) -> object:
        return kwargs["data"]

    async def fake_pre_call_hook(*, user_api_key_dict: UserAPIKeyAuth, data: dict[str, object], call_type: str) -> dict[str, object]:
        return data

    async def fake_post_call_failure_hook(**_: object) -> None:
        return None

    async def failing_route_request(**_: object) -> None:
        raise HTTPException(status_code=401, detail={"error": "invalid api key"})

    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", fake_add_litellm_data_to_request)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        SimpleNamespace(pre_call_hook=fake_pre_call_hook, post_call_failure_hook=fake_post_call_failure_hook),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.route_request", failing_route_request)

    body = orjson.dumps({"model": "dall-e-3", "prompt": "a lighthouse at dusk"})

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/images/generations",
            "headers": [(b"x-litellm-call-id", call_id.encode())],
        },
        receive,
    )

    with caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"), pytest.raises(ProxyException) as raised:
        await endpoints.image_generation(request=request, fastapi_response=Response(), user_api_key_dict=UserAPIKeyAuth())

    assert raised.value.headers["x-litellm-call-id"] == call_id
    record = next(r for r in caplog.records if "Exception occured" in r.getMessage())
    assert record.litellm_call_id == call_id
    assert call_id in record.getMessage()


@pytest.mark.asyncio
async def test_failure_before_the_provider_call_bills_the_callers_litellm_call_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LIT-7836: when the request is rejected while it is still being prepared, the
    failure hook must see the same litellm_call_id the response header answers with,
    otherwise the spend row is stored under a freshly minted id nobody can look up."""
    call_id = "images-early-7836"
    hook_request_data: list[Mapping[str, object]] = []

    async def rejecting_add_litellm_data_to_request(**_: object) -> object:
        raise HTTPException(status_code=400, detail={"error": "tag not allowed"})

    async def fake_post_call_failure_hook(*, request_data: Mapping[str, object], **_: object) -> None:
        hook_request_data.append(request_data)

    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", rejecting_add_litellm_data_to_request)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        SimpleNamespace(post_call_failure_hook=fake_post_call_failure_hook),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")

    body = orjson.dumps({"model": "dall-e-3", "prompt": "a lighthouse at dusk", "litellm_call_id": "from-the-body"})

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/images/generations",
            "headers": [(b"x-litellm-call-id", call_id.encode())],
        },
        receive,
    )

    with pytest.raises(ProxyException) as raised:
        await endpoints.image_generation(request=request, fastapi_response=Response(), user_api_key_dict=UserAPIKeyAuth())

    assert raised.value.headers["x-litellm-call-id"] == call_id
    assert [data["litellm_call_id"] for data in hook_request_data] == [call_id]
