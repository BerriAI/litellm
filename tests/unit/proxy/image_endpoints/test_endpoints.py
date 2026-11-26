import asyncio
import copy
import itertools
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from types import SimpleNamespace
from typing import Any, Dict, Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import orjson
import pytest
import respx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.gemini.empty_image_response import ImageRefusalError
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
from litellm.proxy.image_endpoints import endpoints
from litellm.proxy.proxy_server import ProxyConfig
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import ProxyLogging
from litellm.router_utils.cooldown_handlers import get_cooldown_deployments
from litellm.types.router import RetryPolicy
from litellm.types.utils import ImageObject, ImageResponse
from litellm.utils import custom_llm_setup


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


def test_image_edit_bracketed_image_config_fields_reach_the_provider_as_sent(monkeypatch):
    captured: Dict[str, Any] = {}

    response = _image_edit_client(monkeypatch, captured).post(
        "/v1/images/edits",
        files={"image": ("tree.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        data={"model": "gpt-image-1", "prompt": "add a hat", "imageConfig[aspectRatio]": "9:16"},
    )

    assert response.status_code == 200
    assert captured["imageConfig[aspectRatio]"] == "9:16"
    assert "imageConfig" not in captured


@pytest.mark.parametrize(
    "files, form, missing",
    [
        ({}, {"model": "stability.stable-style-transfer-v1:0", "prompt": "oil painting"}, "image"),
        (
            {"image": ("tree.png", b"\x89PNG\r\n\x1a\n", "image/png")},
            {"model": "stability.stable-image-remove-background-v1:0"},
            "prompt",
        ),
    ],
)
def test_image_edit_without_an_optional_field_reaches_the_provider_with_it_set_to_none(
    monkeypatch, files, form, missing
):
    captured: Dict[str, Any] = {}

    response = _image_edit_client(monkeypatch, captured).post("/v1/images/edits", files=files or None, data=form)

    assert response.status_code == 200, response.text
    assert missing in captured and captured[missing] is None, captured


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

    async def fake_pre_call_hook(
        *, user_api_key_dict: UserAPIKeyAuth, data: dict[str, object], call_type: str
    ) -> dict[str, object]:
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
        await endpoints.image_generation(
            request=request, fastapi_response=Response(), user_api_key_dict=UserAPIKeyAuth()
        )

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
        await endpoints.image_generation(
            request=request, fastapi_response=Response(), user_api_key_dict=UserAPIKeyAuth()
        )

    assert raised.value.headers["x-litellm-call-id"] == call_id
    assert [data["litellm_call_id"] for data in hook_request_data] == [call_id]


class _BilledThenFailingImageHandler(litellm.CustomLLM):
    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        api_key: str | None,
        api_base: str | None,
        optional_params: Mapping[str, object],
        logging_obj: object,
        timeout: object | None = None,
        client: object | None = None,
    ) -> ImageResponse:
        assert isinstance(logging_obj, LiteLLMLoggingObj)
        logging_obj.pre_call(input=prompt, api_key=api_key)
        logging_obj.record_partial_usage_for_failure(
            litellm.Usage(prompt_tokens=16, completion_tokens=2240, total_tokens=2256), 0.25
        )
        raise litellm.RateLimitError(message="refused after billing", llm_provider="image-regression", model=model)

    async def aimage_edit(
        self,
        model: str,
        image: object,
        prompt: str | None,
        model_response: ImageResponse,
        api_key: str | None,
        api_base: str | None,
        optional_params: Mapping[str, object],
        logging_obj: object,
        timeout: object | None = None,
        client: object | None = None,
    ) -> ImageResponse:
        return await self.aimage_generation(
            model, prompt or "", model_response, api_key, api_base, optional_params, logging_obj
        )


class _PricedImageHandler(litellm.CustomLLM):
    def __init__(self, served_cost: float | None = 0.5) -> None:
        super().__init__()
        self.served_cost: Final = served_cost

    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        api_key: str | None,
        api_base: str | None,
        optional_params: Mapping[str, object],
        logging_obj: object,
        timeout: object | None = None,
        client: object | None = None,
    ) -> ImageResponse:
        served: Final = ImageResponse(data=[ImageObject(url="https://example.com/tree.png")])
        served._hidden_params = {} if self.served_cost is None else {"response_cost": self.served_cost}
        return served

    async def aimage_edit(
        self,
        model: str,
        image: object,
        prompt: str | None,
        model_response: ImageResponse,
        api_key: str | None,
        api_base: str | None,
        optional_params: Mapping[str, object],
        logging_obj: object,
        timeout: object | None = None,
        client: object | None = None,
    ) -> ImageResponse:
        return await self.aimage_generation(
            model, prompt or "", model_response, api_key, api_base, optional_params, logging_obj
        )


class _ParamRecordingImageHandler(litellm.CustomLLM):
    received_params: Mapping[str, object] | None = None

    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        api_key: str | None,
        api_base: str | None,
        optional_params: Mapping[str, object],
        logging_obj: object,
        timeout: object | None = None,
        client: object | None = None,
    ) -> ImageResponse:
        self.received_params = dict(optional_params)
        return ImageResponse(data=[ImageObject(url="https://example.com/tree.png")])


class _UnbilledFailingImageHandler(litellm.CustomLLM):
    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        api_key: str | None,
        api_base: str | None,
        optional_params: Mapping[str, object],
        logging_obj: object,
        timeout: object | None = None,
        client: object | None = None,
    ) -> ImageResponse:
        raise litellm.ServiceUnavailableError(message="fallback down", llm_provider="image-regression", model=model)


class _SuccessUsageCapture(CustomLogger):
    logged_usage: tuple[object, object] | None = None
    logged: Final = threading.Event()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        payload: Final = kwargs.get("standard_logging_object")
        assert isinstance(payload, dict)
        self.logged_usage = (payload.get("prompt_tokens"), payload.get("completion_tokens"))
        self.logged.set()


def _image_regression_app(
    monkeypatch: pytest.MonkeyPatch,
    user_api_key_dict: UserAPIKeyAuth,
    handlers: Mapping[str, litellm.CustomLLM],
    fallbacks: tuple[Mapping[str, list[str]], ...] = (),
) -> FastAPI:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "provider_list", [*litellm.provider_list])
    monkeypatch.setattr(litellm, "_custom_providers", [*litellm._custom_providers])
    monkeypatch.setattr(
        litellm,
        "custom_provider_map",
        [{"provider": f"image-regression-{group}", "custom_handler": handler} for group, handler in handlers.items()],
    )
    custom_llm_setup()
    monkeypatch.setattr(litellm, "model_alias_map", {})
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {"model_name": group, "litellm_params": {"model": f"image-regression-{group}/image-model"}}
                for group in handlers
            ],
            fallbacks=[dict(fallback) for fallback in fallbacks],
            num_retries=0,
        ),
    )
    monkeypatch.setattr(proxy_server, "proxy_config", ProxyConfig())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=DualCache()))
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server, "version", "test-version")
    app: Final = FastAPI()
    app.include_router(endpoints.router)
    app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)
    app.dependency_overrides[user_api_key_auth] = lambda: user_api_key_dict
    return app


def _send_generation(client: TestClient) -> httpx.Response:
    return client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree"})


def _send_edit(client: TestClient) -> httpx.Response:
    return client.post(
        "/v1/images/edits",
        files={"image": ("tree.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        data={"model": "A", "prompt": "add a hat"},
    )


@pytest.mark.parametrize(
    "send",
    (pytest.param(_send_generation, id="generations"), pytest.param(_send_edit, id="edits")),
)
def test_failed_image_request_bills_what_the_provider_charged_in_its_spend_row_and_cost_header(
    monkeypatch: pytest.MonkeyPatch, send: Callable[[TestClient], httpx.Response]
) -> None:
    spend_writer: Final = MagicMock(update_database=AsyncMock())
    monkeypatch.setattr(litellm, "callbacks", [ProxyDBLogger(spend_writer=lambda: spend_writer)])
    app: Final = _image_regression_app(
        monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {"A": _BilledThenFailingImageHandler()}
    )

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = send(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 429, response.text
    db_call: Final = spend_writer.update_database.call_args.kwargs
    assert float(response.headers["x-litellm-response-cost"]) == db_call["response_cost"]
    payload: Final = get_logging_payload(
        kwargs=db_call["kwargs"],
        response_obj=db_call["completion_response"],
        start_time=db_call["start_time"],
        end_time=db_call["end_time"],
    )
    assert db_call["response_cost"] == 0.25
    assert (payload["status"], payload["prompt_tokens"], payload["completion_tokens"], payload["total_tokens"]) == (
        "failure",
        16,
        2240,
        2256,
    )


def test_failed_image_generation_that_billed_nothing_sends_no_cost_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    app: Final = _image_regression_app(
        monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {"A": _UnbilledFailingImageHandler()}
    )

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 503, response.text
    assert "x-litellm-call-id" in response.headers
    assert "x-litellm-response-cost" not in response.headers


@pytest.mark.parametrize("field", ("is_async_call", "original_function", "rules_obj", "start_time"))
def test_image_generation_never_sends_a_body_field_named_like_a_setup_parameter_to_the_provider(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    handler: Final = _ParamRecordingImageHandler()
    app: Final = _image_regression_app(monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {"A": handler})

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = client.post(
                "/v1/images/generations", json={"model": "A", "prompt": "draw a tree", field: True}
            )
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 500, response.text
    assert handler.received_params is None


@pytest.mark.parametrize(
    ("deployments", "num_retries", "allowed_fails", "cooled_down"),
    (
        pytest.param(("one", "two", "three"), 2, 0, {"one", "two", "three"}, id="every_retried_deployment_cools_down"),
        pytest.param(("one", "two"), 0, 1, set(), id="one_failure_counts_once"),
    ),
)
def test_failed_image_attempts_each_count_toward_their_deployments_cooldown(
    monkeypatch: pytest.MonkeyPatch,
    deployments: tuple[str, ...],
    num_retries: int,
    allowed_fails: int,
    cooled_down: set[str],
) -> None:
    from litellm.proxy import proxy_server
    from litellm.router_utils.cooldown_handlers import async_get_cooldown_deployments

    monkeypatch.setattr(litellm, "callbacks", [])
    app: Final = _image_regression_app(
        monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {"A": _BilledThenFailingImageHandler()}
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "A",
                "litellm_params": {"model": f"image-regression-A/{deployment}"},
                "model_info": {"id": deployment},
            }
            for deployment in deployments
        ],
        num_retries=num_retries,
        allowed_fails=allowed_fails,
        cooldown_time=60,
        retry_after=0,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 429, response.text
    assert set(asyncio.run(async_get_cooldown_deployments(router, parent_otel_span=None))) == cooled_down


@pytest.mark.parametrize(
    "send",
    (pytest.param(_send_generation, id="generations"), pytest.param(_send_edit, id="edits")),
)
@pytest.mark.parametrize(
    ("served_cost", "billed"),
    (pytest.param(0.5, 0.75, id="priced_fallback"), pytest.param(None, 0.25, id="unpriced_fallback")),
)
def test_image_cost_header_includes_the_billed_attempt_a_fallback_replaced(
    monkeypatch: pytest.MonkeyPatch,
    send: Callable[[TestClient], httpx.Response],
    served_cost: float | None,
    billed: float,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    app: Final = _image_regression_app(
        monkeypatch,
        UserAPIKeyAuth(api_key="sk-test"),
        {"A": _BilledThenFailingImageHandler(), "B": _PricedImageHandler(served_cost)},
        fallbacks=({"A": ["B"]},),
    )

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = send(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 200, response.text
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(billed)


def test_image_generation_keeps_a_keys_logging_callbacks_out_of_the_provider_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    handler: Final = _ParamRecordingImageHandler()
    key_with_failure_logging: Final = UserAPIKeyAuth(
        api_key="sk-test",
        metadata={"logging": [{"callback_name": "langfuse", "callback_type": "failure", "callback_vars": {}}]},
    )
    app: Final = _image_regression_app(monkeypatch, key_with_failure_logging, {"A": handler})

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = client.post(
                "/v1/images/generations", json={"model": "A", "prompt": "draw a tree", "size": "1024x1024"}
            )
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 200, response.text
    assert handler.received_params == {"size": "1024x1024"}


class _LoggedPromptCapture(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged_payloads: tuple[str, ...] = ()
        self.logged: Final = threading.Event()

    def _capture(self, kwargs: dict[str, object]) -> None:
        self.logged_payloads = (*self.logged_payloads, json.dumps(kwargs.get("standard_logging_object"), default=str))
        self.logged.set()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._capture(kwargs)

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._capture(kwargs)


def _router_with_deployment_settings(
    handler_group: str, deployment_settings: Mapping[str, object], num_retries: int = 0
) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": handler_group,
                "litellm_params": {"model": f"image-regression-{handler_group}/image-model", **deployment_settings},
            }
        ],
        num_retries=num_retries,
    )


@pytest.mark.parametrize(
    ("handler", "deployment_settings", "prompt_logged"),
    (
        pytest.param(_ParamRecordingImageHandler, {"turn_off_message_logging": True}, False, id="served_private"),
        pytest.param(_UnbilledFailingImageHandler, {"turn_off_message_logging": True}, False, id="failed_private"),
        pytest.param(_ParamRecordingImageHandler, {}, True, id="served_open"),
    ),
)
def test_image_log_follows_the_deployments_message_logging_setting(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[], litellm.CustomLLM],
    deployment_settings: Mapping[str, object],
    prompt_logged: bool,
) -> None:
    from litellm.proxy import proxy_server

    capture: Final = _LoggedPromptCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    app: Final = _image_regression_app(monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {"A": handler()})
    monkeypatch.setattr(proxy_server, "llm_router", _router_with_deployment_settings("A", deployment_settings))

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            _send_generation(client)
            assert capture.logged.wait(timeout=10)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert any("draw a tree" in payload for payload in capture.logged_payloads) is prompt_logged


def test_image_generation_runs_a_deployments_callbacks_without_sending_them_to_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "callbacks", [])
    fired: Final = threading.Event()
    handler: Final = _ParamRecordingImageHandler()
    app: Final = _image_regression_app(monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {"A": handler})

    def on_success(kwargs: object, response: object, start_time: object, end_time: object) -> None:
        fired.set()

    monkeypatch.setattr(
        proxy_server, "llm_router", _router_with_deployment_settings("A", {"success_callback": [on_success]})
    )

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = client.post(
                "/v1/images/generations", json={"model": "A", "prompt": "draw a tree", "size": "1024x1024"}
            )
            assert fired.wait(timeout=10)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 200, response.text
    assert handler.received_params == {"size": "1024x1024"}


def test_success_event_of_a_fallback_reports_the_tokens_it_billed(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: Final = _SuccessUsageCapture()
    capture.logged.clear()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    app: Final = _image_regression_app(
        monkeypatch,
        UserAPIKeyAuth(api_key="sk-test"),
        {"A": _BilledThenFailingImageHandler(), "B": _ParamRecordingImageHandler()},
        fallbacks=({"A": ["B"]},),
    )

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
            assert capture.logged.wait(timeout=10)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 200, response.text
    assert capture.logged_usage == (16, 2240)


class _ServedAttemptCapture(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.served_kwargs: Mapping[str, object] | None = None
        self.logged: Final = threading.Event()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.served_kwargs = dict(kwargs)
        self.logged.set()


def test_served_fallback_logs_as_a_success_without_the_failed_attempts_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: Final = _ServedAttemptCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    app: Final = _image_regression_app(
        monkeypatch,
        UserAPIKeyAuth(api_key="sk-test"),
        {"A": _UnbilledFailingImageHandler(), "B": _ParamRecordingImageHandler()},
        fallbacks=({"A": ["B"]},),
    )

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
            assert capture.logged.wait(timeout=10)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 200, response.text
    assert capture.served_kwargs is not None
    assert "exception" not in capture.served_kwargs


@pytest.mark.parametrize(
    ("first_price", "served_price", "billed"),
    (
        pytest.param(None, 0.25, 0.25, id="served_deployment_price"),
        pytest.param(0.75, 0.25, 0.25, id="fallback_uses_its_own_price"),
    ),
)
def test_image_is_billed_at_the_price_of_the_deployment_that_served_it(
    monkeypatch: pytest.MonkeyPatch, first_price: float | None, served_price: float, billed: float
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "callbacks", [])
    app: Final = _image_regression_app(
        monkeypatch,
        UserAPIKeyAuth(api_key="sk-test"),
        {"A": _UnbilledFailingImageHandler(), "B": _ParamRecordingImageHandler()},
    )
    prices: Final = {"A": first_price, "B": served_price}
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": group,
                    "litellm_params": {
                        "model": f"image-regression-{group}/image-model",
                        **({} if price is None else {"output_cost_per_image": price}),
                    },
                }
                for group, price in prices.items()
            ],
            fallbacks=[{"A": ["B"]}],
            num_retries=0,
        ),
    )

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 200, response.text
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(billed)


def test_a_failure_spend_row_keeps_the_billed_attempt_when_every_fallback_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spend_writer: Final = MagicMock(update_database=AsyncMock())
    monkeypatch.setattr(litellm, "callbacks", [ProxyDBLogger(spend_writer=lambda: spend_writer)])
    app: Final = _image_regression_app(
        monkeypatch,
        UserAPIKeyAuth(api_key="sk-test"),
        {"A": _BilledThenFailingImageHandler(), "B": _UnbilledFailingImageHandler()},
        fallbacks=({"A": ["B"]},),
    )

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 429, response.text
    db_call: Final = spend_writer.update_database.call_args.kwargs
    payload: Final = get_logging_payload(
        kwargs=db_call["kwargs"],
        response_obj=db_call["completion_response"],
        start_time=db_call["start_time"],
        end_time=db_call["end_time"],
    )
    assert (db_call["response_cost"], payload["prompt_tokens"], payload["completion_tokens"]) == (0.25, 16, 2240)


class _RefusedAfterBillingImageHandler(litellm.CustomLLM):
    def __init__(self, refusals: int) -> None:
        super().__init__()
        self.refusals_left: int = refusals

    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        api_key: str | None,
        api_base: str | None,
        optional_params: Mapping[str, object],
        logging_obj: object,
        timeout: object | None = None,
        client: object | None = None,
    ) -> ImageResponse:
        assert isinstance(logging_obj, LiteLLMLoggingObj)
        logging_obj.pre_call(input=prompt, api_key=api_key)
        if self.refusals_left == 0:
            served: Final = ImageResponse(data=[ImageObject(url="https://example.com/tree.png")])
            served._hidden_params = {"response_cost": 0.5}
            return served
        self.refusals_left -= 1
        logging_obj.record_partial_usage_for_failure(
            litellm.Usage(prompt_tokens=16, completion_tokens=2240, total_tokens=2256), 0.25
        )
        raise litellm.ContentPolicyViolationError(
            message="refused after billing", llm_provider="image-regression", model=model
        )

    async def aimage_edit(
        self,
        model: str,
        image: object,
        prompt: str | None,
        model_response: ImageResponse,
        api_key: str | None,
        api_base: str | None,
        optional_params: Mapping[str, object],
        logging_obj: object,
        timeout: object | None = None,
        client: object | None = None,
    ) -> ImageResponse:
        return await self.aimage_generation(
            model, prompt or "", model_response, api_key, api_base, optional_params, logging_obj
        )


def _router_retrying_refusals(retries: int) -> litellm.Router:
    return litellm.Router(
        model_list=[{"model_name": "A", "litellm_params": {"model": "image-regression-A/image-model"}}],
        num_retries=0,
        retry_after=0,
        retry_policy=RetryPolicy(ContentPolicyViolationErrorRetries=retries),
    )


def test_a_refusal_served_on_retry_bills_both_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    capture: Final = _SuccessUsageCapture()
    capture.logged.clear()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    app: Final = _image_regression_app(
        monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {"A": _RefusedAfterBillingImageHandler(refusals=1)}
    )
    monkeypatch.setattr(proxy_server, "llm_router", _router_retrying_refusals(retries=1))

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
            assert capture.logged.wait(timeout=10)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 200, response.text
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.75)
    assert capture.logged_usage == (16, 2240)


@pytest.mark.parametrize(
    "send",
    (pytest.param(_send_generation, id="generations"), pytest.param(_send_edit, id="edits")),
)
@pytest.mark.parametrize("retries", (1, 2))
def test_a_failure_spend_row_keeps_every_refused_retry(
    monkeypatch: pytest.MonkeyPatch, retries: int, send: Callable[[TestClient], httpx.Response]
) -> None:
    from litellm.proxy import proxy_server

    attempts: Final = retries + 1

    spend_writer: Final = MagicMock(update_database=AsyncMock())
    monkeypatch.setattr(litellm, "callbacks", [ProxyDBLogger(spend_writer=lambda: spend_writer)])
    app: Final = _image_regression_app(
        monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {"A": _RefusedAfterBillingImageHandler(refusals=attempts)}
    )
    monkeypatch.setattr(proxy_server, "llm_router", _router_retrying_refusals(retries=retries))

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = send(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 400, response.text
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.25 * attempts)
    db_call: Final = spend_writer.update_database.call_args.kwargs
    payload: Final = get_logging_payload(
        kwargs=db_call["kwargs"],
        response_obj=db_call["completion_response"],
        start_time=db_call["start_time"],
        end_time=db_call["end_time"],
    )
    assert db_call["response_cost"] == pytest.approx(0.25 * attempts)
    assert (payload["prompt_tokens"], payload["completion_tokens"]) == (16 * attempts, 2240 * attempts)


def test_every_failed_image_retry_reaches_the_failure_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    capture: Final = _LoggedPromptCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    app: Final = _image_regression_app(
        monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {"A": _UnbilledFailingImageHandler()}
    )
    monkeypatch.setattr(proxy_server, "llm_router", _router_with_deployment_settings("A", {}, num_retries=1))

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 503, response.text
    assert len(capture.logged_payloads) == 2


def test_a_failed_image_fallback_counts_toward_its_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "callbacks", [])
    app: Final = _image_regression_app(
        monkeypatch,
        UserAPIKeyAuth(api_key="sk-test"),
        {"A": _UnbilledFailingImageHandler(), "B": _UnbilledFailingImageHandler()},
    )
    router: Final = litellm.Router(
        model_list=[
            {"model_name": "A", "litellm_params": {"model": "image-regression-A/image-model"}},
            *(
                {
                    "model_name": "B",
                    "litellm_params": {"model": "image-regression-B/image-model"},
                    "model_info": {"id": deployment_id},
                }
                for deployment_id in ("b1", "b2")
            ),
        ],
        fallbacks=[{"A": ["B"]}],
        num_retries=0,
        allowed_fails=0,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 503, response.text
    assert len(set(get_cooldown_deployments(router, None)) & {"b1", "b2"}) == 1


_REFUSED_IMAGE_TOKENS: Final = 1110


def _group_usage_once_settled(
    router: litellm.Router, expected: tuple[int | None, int | None], timeout: float = 5.0
) -> tuple[int | None, int | None]:
    deadline: Final = time.monotonic() + timeout
    while time.monotonic() < deadline and asyncio.run(router.get_model_group_usage("A")) != expected:
        time.sleep(0.05)
    return asyncio.run(router.get_model_group_usage("A"))


@pytest.mark.parametrize(
    "send",
    (pytest.param(_send_generation, id="generations"), pytest.param(_send_edit, id="edits")),
)
def test_a_refused_gemini_image_counts_toward_its_model_group_usage(
    monkeypatch: pytest.MonkeyPatch, send: Callable[[TestClient], httpx.Response]
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    app: Final = _image_regression_app(monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {})
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "A",
                "litellm_params": {
                    "model": "gemini/gemini-3.1-flash-image",
                    "api_key": "test-key",
                    "api_base": "https://google.invalid/v1",
                },
            }
        ],
        num_retries=0,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    refusal: Final[Mapping[str, object]] = {
        "candidates": [{"finishReason": "IMAGE_SAFETY", "content": {"parts": []}}],
        "usageMetadata": {
            "promptTokenCount": _REFUSED_IMAGE_TOKENS - 333,
            "candidatesTokenCount": 333,
            "totalTokenCount": _REFUSED_IMAGE_TOKENS,
        },
    }

    try:
        with respx.mock(assert_all_called=True) as upstream, TestClient(app, raise_server_exceptions=False) as client:
            upstream.post(url__startswith="https://google.invalid/").mock(
                return_value=httpx.Response(200, json=refusal)
            )
            response: Final = send(client)
            usage: Final = _group_usage_once_settled(router, (_REFUSED_IMAGE_TOKENS, 1))
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 400, response.text
    assert usage == (_REFUSED_IMAGE_TOKENS, 1)


@pytest.mark.parametrize(
    ("deployment_settings", "prompt_stored"),
    (
        pytest.param({"turn_off_message_logging": True}, False, id="private"),
        pytest.param({}, True, id="open"),
    ),
)
def test_a_failed_image_spend_row_follows_the_deployments_message_logging_setting(
    monkeypatch: pytest.MonkeyPatch, deployment_settings: Mapping[str, object], prompt_stored: bool
) -> None:
    from litellm.proxy import proxy_server

    spend_writer: Final = MagicMock(update_database=AsyncMock())
    monkeypatch.setattr(litellm, "callbacks", [ProxyDBLogger(spend_writer=lambda: spend_writer)])
    app: Final = _image_regression_app(
        monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {"A": _UnbilledFailingImageHandler()}
    )
    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True})
    monkeypatch.setattr(proxy_server, "llm_router", _router_with_deployment_settings("A", deployment_settings))

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 503, response.text
    db_call: Final = spend_writer.update_database.call_args.kwargs
    payload: Final = get_logging_payload(
        kwargs=db_call["kwargs"],
        response_obj=db_call["completion_response"],
        start_time=db_call["start_time"],
        end_time=db_call["end_time"],
    )
    assert ("draw a tree" in str(payload["proxy_server_request"])) is prompt_stored


class _TimedOutImageHandler(litellm.CustomLLM):
    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        api_key: str | None,
        api_base: str | None,
        optional_params: Mapping[str, object],
        logging_obj: object,
        timeout: object | None = None,
        client: object | None = None,
    ) -> ImageResponse:
        assert isinstance(logging_obj, LiteLLMLoggingObj)
        logging_obj.pre_call(input=prompt, api_key=api_key or "", additional_args={})
        raise litellm.Timeout(
            message="read timed out", model=model, llm_provider="image-regression"
        ) from httpx.ReadTimeout("read timed out")


def test_an_image_timeout_writes_no_estimated_usage_to_its_spend_row(monkeypatch: pytest.MonkeyPatch) -> None:
    spend_writer: Final = MagicMock(update_database=AsyncMock())
    monkeypatch.setattr(litellm, "callbacks", [ProxyDBLogger(spend_writer=lambda: spend_writer)])
    app: Final = _image_regression_app(monkeypatch, UserAPIKeyAuth(api_key="sk-test"), {"A": _TimedOutImageHandler()})

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 408, response.text
    db_call: Final = spend_writer.update_database.call_args.kwargs
    payload: Final = get_logging_payload(
        kwargs=db_call["kwargs"],
        response_obj=db_call["completion_response"],
        start_time=db_call["start_time"],
        end_time=db_call["end_time"],
    )
    assert (db_call["response_cost"], payload["prompt_tokens"]) == (0, 0)


class _BilledImageRefusalHandler(litellm.CustomLLM):
    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        api_key: str | None,
        api_base: str | None,
        optional_params: Mapping[str, object],
        logging_obj: object,
        timeout: object | None = None,
        client: object | None = None,
    ) -> ImageResponse:
        assert isinstance(logging_obj, LiteLLMLoggingObj)
        logging_obj.pre_call(input=prompt, api_key=api_key)
        logging_obj.record_partial_usage_for_failure(
            litellm.Usage(prompt_tokens=16, completion_tokens=2240, total_tokens=2256), 0.25
        )
        raise ImageRefusalError(
            message="Gemini refused to return an image, finishReason=IMAGE_SAFETY",
            model=model,
            llm_provider="image-regression",
            response=httpx.Response(400, request=httpx.Request("POST", "https://google.invalid")),
            provider_specific_fields={},
        )


class _RejectingPostCallHook(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        raise HTTPException(status_code=400, detail="the served image broke a post-call rule")


@pytest.fixture
def fresh_prometheus_registry() -> Iterator[None]:
    from prometheus_client import REGISTRY

    def unregister_all() -> None:
        for collector in tuple(REGISTRY._collector_to_names):
            REGISTRY.unregister(collector)

    unregister_all()
    yield
    unregister_all()


def _spend_by_model() -> Mapping[str, float]:
    from prometheus_client import REGISTRY

    samples: Final = itertools.chain.from_iterable(metric.samples for metric in REGISTRY.collect())
    return {sample.labels["model"]: sample.value for sample in samples if sample.name == "litellm_spend_metric_total"}


def _spend_once_settled(expected: Mapping[str, float], timeout: float = 5.0) -> Mapping[str, float]:
    deadline: Final = time.monotonic() + timeout
    while time.monotonic() < deadline and _spend_by_model() != expected:
        time.sleep(0.05)
    return _spend_by_model()


@pytest.mark.usefixtures("fresh_prometheus_registry")
@pytest.mark.parametrize(
    ("handlers", "fallbacks", "rejected_after_serving", "spend"),
    (
        pytest.param({"A": _PricedImageHandler(0.5)}, (), True, {"image-model": 0.5}, id="served_then_rejected"),
        pytest.param(
            {"A": _BilledImageRefusalHandler(), "B": _PricedImageHandler(0.5)},
            ({"A": ["B"]},),
            True,
            {"image-model": 0.75},
            id="refused_then_served_by_fallback_then_rejected",
        ),
        pytest.param(
            {"A": _BilledImageRefusalHandler()},
            (),
            False,
            {"image-model": 0.25},
            id="refused_without_a_served_answer",
        ),
    ),
)
def test_an_image_rejected_after_serving_keeps_its_spend_under_the_served_model_counted_once(
    monkeypatch: pytest.MonkeyPatch,
    handlers: Mapping[str, litellm.CustomLLM],
    fallbacks: tuple[Mapping[str, list[str]], ...],
    rejected_after_serving: bool,
    spend: Mapping[str, float],
) -> None:
    from litellm.integrations.prometheus import PrometheusLogger

    monkeypatch.setattr(
        litellm, "callbacks", [PrometheusLogger(), *((_RejectingPostCallHook(),) if rejected_after_serving else ())]
    )
    app: Final = _image_regression_app(
        monkeypatch,
        UserAPIKeyAuth(api_key="sk-test", request_route="/v1/images/generations"),
        handlers,
        fallbacks=fallbacks,
    )

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = _send_generation(client)
            settled: Final = _spend_once_settled(spend)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 400, response.text
    assert settled == pytest.approx(spend)
