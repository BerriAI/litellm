import asyncio
import copy
import logging
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, Final, cast

import httpx
import orjson
import pytest
import respx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

import litellm
from litellm._logging import session_id_var, trace_id_var, verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.image_endpoints import endpoints
from litellm.proxy.proxy_server import PrismaClient, ProxyConfig
from litellm.proxy.utils import ProxyLogging
from litellm.redis.stores.auth_cache import AuthObject
from litellm.router_utils.router_callbacks.track_deployment_metrics import get_deployment_failures_for_current_minute
from litellm.types.guardrails import GuardrailEventHooks
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


class _ImageRouterHandler(litellm.CustomLLM):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.received_params: list[Mapping[str, object]] = []
        self.billed_before_failure: tuple[litellm.Usage, float] | None = None
        self.owed_after_failure: tuple[litellm.Usage, float] | None = None
        self.served_cost: float | None = None
        self.refusal: litellm.ContentPolicyViolationError | None = None

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
        self.calls.append(model)
        self.received_params.append(dict(optional_params))
        if isinstance(logging_obj, LiteLLMLoggingObj):
            logging_obj.pre_call(input=prompt, api_key=api_key)
        if model.endswith("failing"):
            if self.refusal is not None:
                raise self.refusal
            if self.billed_before_failure is not None and isinstance(logging_obj, LiteLLMLoggingObj):
                logging_obj.record_partial_usage_for_failure(*self.billed_before_failure)
            if self.owed_after_failure is not None and isinstance(logging_obj, LiteLLMLoggingObj):
                logging_obj.record_served_rounds_for_failure((self.owed_after_failure,))
            raise litellm.RateLimitError(message="primary failed", llm_provider="image-regression", model=model)
        served: Final = ImageResponse(created=1, data=[ImageObject(url="https://images.example/generated.png")])
        if self.served_cost is not None:
            served._hidden_params = {"response_cost": self.served_cost}
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


def _image_generation_client(
    monkeypatch: pytest.MonkeyPatch,
    user_api_key_dict: UserAPIKeyAuth,
    team_settings: Mapping[str, object] | None = None,
    working_params: Mapping[str, object] | None = None,
    failing_params: Mapping[str, object] | None = None,
) -> tuple[TestClient, _ImageRouterHandler]:
    from litellm.proxy import proxy_server

    handler: Final = _ImageRouterHandler()
    cache: Final = UserApiKeyCache()
    monkeypatch.setattr(litellm, "provider_list", [*litellm.provider_list])
    monkeypatch.setattr(litellm, "_custom_providers", [*litellm._custom_providers])
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "image-regression", "custom_handler": handler}])
    custom_llm_setup()
    if team_settings is not None:
        team_id: Final = user_api_key_dict.team_id
        assert team_id is not None
        cache.set_cache(
            key=AuthObject.team(team_id),
            value=LiteLLM_TeamTableCachedObj(team_id=team_id, router_settings=dict(team_settings)),
            local_only=True,
            model_type=LiteLLM_TeamTableCachedObj,
        )
        monkeypatch.setattr(proxy_server, "prisma_client", cast(PrismaClient, object()))

    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "A",
                "litellm_params": {"model": "image-regression/failing", **(failing_params or {})},
                "model_info": {"id": "image-regression-failing"},
            },
            {
                "model_name": "B",
                "litellm_params": {"model": "image-regression/working", **(working_params or {})},
                "model_info": {"id": "image-regression-working"},
            },
        ],
        num_retries=0,
    )
    monkeypatch.setattr(litellm, "model_alias_map", {})
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", ProxyConfig())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=cache))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server, "version", "test-version")

    app: Final = FastAPI()
    app.include_router(endpoints.router)
    app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)
    app.dependency_overrides[user_api_key_auth] = lambda: user_api_key_dict
    return TestClient(app, raise_server_exceptions=False), handler


def _shows_prompt(value: object) -> bool:
    return "draw a tree" in orjson.dumps(value, default=str).decode()


@dataclass(frozen=True, slots=True)
class _LoggedRow:
    outcome: str
    prompt_shown: tuple[bool, bool, bool]
    callback_params: Mapping[str, object]

    @classmethod
    def of(cls, outcome: str, kwargs: Mapping[str, object]) -> "_LoggedRow":
        payload: Final = kwargs.get("standard_logging_object")
        assert isinstance(payload, Mapping)
        callback_params: Final = kwargs.get("standard_callback_dynamic_params")
        assert isinstance(callback_params, Mapping)
        return cls(
            outcome,
            (
                _shows_prompt(payload.get("messages")),
                _shows_prompt(kwargs.get("messages")),
                _shows_prompt(kwargs.get("input")),
            ),
            MappingProxyType(dict(callback_params)),
        )


class _ImageOutcomeLogger(CustomLogger):
    def __init__(self) -> None:
        self.events: Queue[tuple[object, object, object]] = Queue()
        self.messages: Queue[object] = Queue()
        self.correlation: Queue[tuple[str, str]] = Queue()
        self.rows: Queue[_LoggedRow] = Queue()

    async def async_log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self.rows.put(_LoggedRow.of("failure", kwargs))
        self.messages.put(kwargs.get("messages"))
        self.correlation.put((trace_id_var.get(), session_id_var.get()))
        self.events.put((kwargs.get("call_type"), kwargs.get("litellm_call_id"), kwargs.get("exception")))

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self.rows.put(_LoggedRow.of("success", kwargs))
        self.messages.put(kwargs.get("messages"))
        self.events.put((kwargs.get("call_type"), kwargs.get("litellm_call_id"), response_obj))


class _RewritingImageLogger(_ImageOutcomeLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: str
    ) -> dict[str, object]:
        return {**data, "messages": [{"role": "user", "content": "sanitized prompt"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ("/v1/images/generations", "/images/generations", "/openai/deployments/B/images/generations")
)
async def test_router_refused_image_generation_reaches_failure_callbacks(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    from litellm.proxy import proxy_server

    logger: Final = _ImageOutcomeLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    client, handler = _image_generation_client(
        monkeypatch, UserAPIKeyAuth(), working_params={"max_budget": 1, "budget_duration": "1d"}
    )
    router: Final = proxy_server.llm_router
    assert router is not None and router.router_budget_logger is not None
    router.router_budget_logger.dual_cache.set_cache("deployment_spend:image-regression-working:1d", 2)

    with client:
        response: Final = client.post(
            path, json={"model": "B", "prompt": "draw a tree"}, headers={"x-litellm-call-id": "refused-image"}
        )
        assert response.status_code == 429, response.text
        call_type, call_id, exception = logger.events.get(timeout=5)

    assert (call_type, call_id) == ("aimage_generation", "refused-image")
    assert isinstance(exception, ValueError) and "crossed budget" in str(exception)
    assert handler.calls == []
    assert logger.events.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("supply_logging", (False, True))
async def test_image_router_refusal_without_or_with_caller_logging(
    monkeypatch: pytest.MonkeyPatch, supply_logging: bool
) -> None:
    from litellm.proxy import proxy_server

    logger: Final = _ImageOutcomeLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)
    _image_generation_client(monkeypatch, UserAPIKeyAuth(), working_params={"max_budget": 1, "budget_duration": "1d"})
    router: Final = proxy_server.llm_router
    assert router is not None and router.router_budget_logger is not None
    router.router_budget_logger.dual_cache.set_cache("deployment_spend:image-regression-working:1d", 2)
    kwargs: Final[dict[str, object]] = {"litellm_trace_id": "refused-trace", "litellm_session_id": "refused-session"}
    if supply_logging:
        logging_obj, _ = litellm.utils.function_setup(
            "aimage_generation",
            litellm.utils.Rules(),
            datetime.now(),
            model="B",
            prompt="draw a tree",
            litellm_call_id="supplied-image",
            **kwargs,
        )
        kwargs.update(litellm_logging_obj=logging_obj, litellm_call_id="supplied-image")

    context_before: Final = (trace_id_var.get(), session_id_var.get())
    with pytest.raises(ValueError, match="crossed budget"):
        await router.aimage_generation(model="B", prompt="draw a tree", **kwargs)
    assert (trace_id_var.get(), session_id_var.get()) == context_before
    call_type, call_id, exception = await asyncio.to_thread(logger.events.get, timeout=5)
    assert call_type == "aimage_generation" and isinstance(exception, ValueError)
    assert call_id == "supplied-image" if supply_logging else isinstance(call_id, str) and call_id
    assert logger.events.empty()
    assert logger.correlation.get_nowait() == ("refused-trace", "refused-session")


@pytest.mark.parametrize(("model", "status_code"), (("A", 429), ("B", 200)))
@pytest.mark.parametrize("rewrite_prompt", (False, True))
def test_image_generation_provider_outcome_is_logged_once(
    monkeypatch: pytest.MonkeyPatch, model: str, status_code: int, rewrite_prompt: bool
) -> None:
    logger: Final = _RewritingImageLogger() if rewrite_prompt else _ImageOutcomeLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    client, handler = _image_generation_client(monkeypatch, UserAPIKeyAuth())
    with client:
        response: Final = client.post(
            "/v1/images/generations",
            json={"model": model, "prompt": "draw a tree"},
            headers={"x-litellm-call-id": "provider-image"},
        )
        assert response.status_code == status_code, response.text
        call_type, call_id, outcome = logger.events.get(timeout=5)
    assert (call_type, call_id) == ("aimage_generation", "provider-image")
    assert isinstance(outcome, litellm.RateLimitError if model == "A" else ImageResponse)
    assert logger.messages.get_nowait() == [
        {"role": "user", "content": "sanitized prompt" if rewrite_prompt else "draw a tree"}
    ]
    assert len(handler.calls) == 1
    assert logger.events.empty()


def test_image_generation_missing_prompt_keeps_its_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client, handler = _image_generation_client(monkeypatch, UserAPIKeyAuth())
    with client:
        response: Final = client.post("/v1/images/generations", json={"model": "B"})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "prompt"
    assert handler.calls == []


def test_image_generation_uses_key_router_settings_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    client, handler = _image_generation_client(
        monkeypatch,
        UserAPIKeyAuth(router_settings={"fallbacks": [{"A": ["B"]}]}),
    )

    response: Final = client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree"})

    assert response.status_code == 200, response.text
    assert (
        response.json()
        == ImageResponse(
            created=1,
            data=[ImageObject(url="https://images.example/generated.png")],
        ).model_dump()
    ), response.text
    assert handler.calls == ["failing", "working"], response.text


def test_image_generation_uses_team_router_settings_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    team_id: Final = "image-regression-team"
    client, handler = _image_generation_client(
        monkeypatch,
        UserAPIKeyAuth(team_id=team_id),
        {"fallbacks": [{"A": ["B"]}]},
    )

    response: Final = client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree"})

    assert response.status_code == 200, response.text
    assert (
        response.json()
        == ImageResponse(
            created=1,
            data=[ImageObject(url="https://images.example/generated.png")],
        ).model_dump()
    ), response.text
    assert handler.calls == ["failing", "working"], response.text


def test_image_generation_without_router_settings_does_not_use_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    client, handler = _image_generation_client(monkeypatch, UserAPIKeyAuth())

    response: Final = client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree"})

    assert response.status_code == 429, response.text
    assert handler.calls == ["failing"], response.text


def test_image_generation_uses_key_model_group_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    client, handler = _image_generation_client(
        monkeypatch,
        UserAPIKeyAuth(router_settings={"model_group_alias": {"alias": "B"}}),
    )

    response: Final = client.post("/v1/images/generations", json={"model": "alias", "prompt": "draw a tree"})

    assert response.status_code == 200, response.text
    assert (
        response.json()
        == ImageResponse(
            created=1,
            data=[ImageObject(url="https://images.example/generated.png")],
        ).model_dump()
    ), response.text
    assert handler.calls == ["working"], response.text


def test_image_generation_uses_key_num_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    logger: Final = _ImageOutcomeLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    client, handler = _image_generation_client(
        monkeypatch,
        UserAPIKeyAuth(router_settings={"num_retries": 2}),
    )

    with client:
        response: Final = client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree"})
        events: Final = [logger.events.get(timeout=5)]

    assert response.status_code == 429, response.text
    assert handler.calls == ["failing", "failing", "failing"], response.text
    assert all(isinstance(event[2], litellm.RateLimitError) for event in events)
    assert logger.events.empty()
    router: Final = proxy_server.llm_router
    assert router is not None
    assert get_deployment_failures_for_current_minute(router, "image-regression-failing") == 3


def test_failed_image_generation_spend_row_records_aimage_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    spend_writer: Final = MagicMock(update_database=AsyncMock())
    monkeypatch.setattr(litellm, "callbacks", [ProxyDBLogger(spend_writer=lambda: spend_writer)])
    client, handler = _image_generation_client(monkeypatch, UserAPIKeyAuth(api_key="sk-test"))
    try:
        with client:
            response: Final = client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree"})
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert (response.status_code, handler.calls) == (429, ["failing"]), response.text
    db_call: Final = spend_writer.update_database.call_args.kwargs
    payload: Final = get_logging_payload(
        kwargs=db_call["kwargs"],
        response_obj=db_call["completion_response"],
        start_time=db_call["start_time"],
        end_time=db_call["end_time"],
    )
    assert (payload["status"], payload["call_type"]) == ("failure", "aimage_generation")


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
def test_failed_image_request_bills_what_the_provider_served_in_its_spend_row_and_cost_header(
    monkeypatch: pytest.MonkeyPatch, send: Callable[[TestClient], httpx.Response]
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    spend_writer: Final = MagicMock(update_database=AsyncMock())
    monkeypatch.setattr(litellm, "callbacks", [ProxyDBLogger(spend_writer=lambda: spend_writer)])
    client, handler = _image_generation_client(monkeypatch, UserAPIKeyAuth(api_key="sk-test"))
    handler.billed_before_failure = (litellm.Usage(prompt_tokens=16, completion_tokens=2240, total_tokens=2256), 0.25)
    try:
        with client:
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


_PRIVATE: Final = MappingProxyType({"turn_off_message_logging": True})
_OPEN: Final = MappingProxyType({})


@pytest.mark.parametrize(
    ("model", "settings", "outcome", "prompt_shown"),
    (
        pytest.param("B", _PRIVATE, "success", False, id="served_private"),
        pytest.param("B", _OPEN, "success", True, id="served_open"),
        pytest.param("A", _PRIVATE, "failure", False, id="failed_private"),
        pytest.param("A", _OPEN, "failure", True, id="failed_open"),
    ),
)
def test_image_log_follows_the_deployments_message_logging_setting(
    monkeypatch: pytest.MonkeyPatch, model: str, settings: Mapping[str, object], outcome: str, prompt_shown: bool
) -> None:
    logger: Final = _ImageOutcomeLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    client, handler = _image_generation_client(
        monkeypatch, UserAPIKeyAuth(), working_params=settings, failing_params=settings
    )

    with client:
        response: Final = client.post("/v1/images/generations", json={"model": model, "prompt": "draw a tree"})
        row: Final = logger.rows.get(timeout=5)

    assert response.status_code == (200 if outcome == "success" else 429), response.text
    assert (row.outcome, row.prompt_shown) == (outcome, (prompt_shown,) * 3)
    assert logger.rows.empty()


@pytest.mark.parametrize(
    ("failing_settings", "working_settings", "failure_shown", "served_shown"),
    (
        pytest.param(_OPEN, _PRIVATE, True, False, id="open_to_private"),
        pytest.param(_PRIVATE, _OPEN, False, True, id="private_to_open"),
    ),
)
def test_image_fallback_logs_each_attempt_by_its_own_deployments_message_logging(
    monkeypatch: pytest.MonkeyPatch,
    failing_settings: Mapping[str, object],
    working_settings: Mapping[str, object],
    failure_shown: bool,
    served_shown: bool,
) -> None:
    logger: Final = _ImageOutcomeLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    client, handler = _image_generation_client(
        monkeypatch,
        UserAPIKeyAuth(router_settings={"fallbacks": [{"A": ["B"]}]}),
        working_params=working_settings,
        failing_params=failing_settings,
    )

    with client:
        response: Final = client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree"})
        rows: Final = (logger.rows.get(timeout=5), logger.rows.get(timeout=5))

    assert response.status_code == 200, response.text
    assert handler.calls == ["failing", "working"]
    assert [(row.outcome, row.prompt_shown) for row in rows] == [
        ("failure", (failure_shown,) * 3),
        ("success", (served_shown,) * 3),
    ]


@pytest.mark.parametrize(
    ("key_metadata", "logged_public_key"),
    (
        pytest.param({}, "pk-deployment", id="deployment_key"),
        pytest.param(
            {
                "logging": [
                    {
                        "callback_name": "image-regression-key-logger",
                        "callback_type": "success",
                        "callback_vars": {"langfuse_public_key": "pk-key"},
                    }
                ]
            },
            "pk-key",
            id="key_wins",
        ),
    ),
)
@pytest.mark.parametrize(("model", "outcome"), (("B", "success"), ("A", "failure")))
def test_image_log_carries_the_deployments_callback_params_unless_the_key_sets_them(
    monkeypatch: pytest.MonkeyPatch,
    key_metadata: Mapping[str, object],
    logged_public_key: str,
    model: str,
    outcome: str,
) -> None:
    logger: Final = _ImageOutcomeLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    settings: Final = {"langfuse_public_key": "pk-deployment"}
    client, _ = _image_generation_client(
        monkeypatch, UserAPIKeyAuth(metadata=dict(key_metadata)), working_params=settings, failing_params=settings
    )

    with client:
        client.post("/v1/images/generations", json={"model": model, "prompt": "draw a tree"})
        row: Final = logger.rows.get(timeout=5)

    assert (row.outcome, row.callback_params.get("langfuse_public_key")) == (outcome, logged_public_key)


class _DeploymentLogger(CustomLogger):
    def __init__(self, report: Callable[[str], None]) -> None:
        self.report: Final = report

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.report("success")

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.report("failure")


@pytest.mark.parametrize(
    ("key_metadata", "deployment_outcomes"),
    (
        pytest.param({}, {"success": ("success", "async_success"), "failure": ("failure",)}, id="deployment_callbacks"),
        pytest.param(
            {
                "logging": [
                    {
                        "callback_name": "image-regression-key-logger",
                        "callback_type": "success_and_failure",
                        "callback_vars": {},
                    }
                ]
            },
            {"success": (), "failure": ()},
            id="key_callbacks_win",
        ),
    ),
)
@pytest.mark.parametrize(("model", "outcome"), (("B", "success"), ("A", "failure")))
def test_image_deployment_callbacks_run_without_reaching_the_provider(
    monkeypatch: pytest.MonkeyPatch,
    key_metadata: Mapping[str, object],
    deployment_outcomes: Mapping[str, tuple[str, ...]],
    model: str,
    outcome: str,
) -> None:
    fired: Final = {name: threading.Event() for name in ("success", "async_success", "failure", "logged")}
    registered_outcomes: Final[Queue[str]] = Queue()

    def report(registered_outcome: str) -> None:
        registered_outcomes.put(registered_outcome)

    def on_success(kwargs: object, response: object, start_time: object, end_time: object) -> None:
        fired["success"].set()

    async def on_async_success(kwargs: object, response: object, start_time: object, end_time: object) -> None:
        fired["async_success"].set()

    def on_failure(kwargs: object, response: object, start_time: object, end_time: object) -> None:
        fired["failure"].set()

    def on_logged(kwargs: object, response: object, start_time: object, end_time: object) -> None:
        fired["logged"].set()

    for name in ("input_callback", "_async_success_callback", "_async_failure_callback", "callbacks"):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "success_callback", [on_logged])
    monkeypatch.setattr(litellm, "failure_callback", [on_logged])
    settings: Final = {
        "success_callback": [on_success, on_async_success],
        "failure_callback": [on_failure],
        "callbacks": [_DeploymentLogger(report)],
    }
    client, handler = _image_generation_client(
        monkeypatch, UserAPIKeyAuth(metadata=dict(key_metadata)), working_params=settings, failing_params=settings
    )

    with client:
        response: Final = client.post("/v1/images/generations", json={"model": model, "prompt": "draw a tree"})
        assert fired["logged"].wait(timeout=5), response.text
        assert registered_outcomes.get(timeout=5) == outcome

    deployment_fired: Final = tuple(name for name in ("success", "async_success", "failure") if fired[name].is_set())
    assert deployment_fired == deployment_outcomes[outcome]
    assert handler.received_params == [{}]


def test_image_fallback_runs_only_each_attempts_own_deployment_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])
    fired: Final = {name: threading.Event() for name in ("failing_success", "failing_failure", "working_success")}

    def on_failing_success(kwargs: object, response: object, start_time: object, end_time: object) -> None:
        fired["failing_success"].set()

    def on_failing_failure(kwargs: object, response: object, start_time: object, end_time: object) -> None:
        fired["failing_failure"].set()

    def on_working_success(kwargs: object, response: object, start_time: object, end_time: object) -> None:
        fired["working_success"].set()

    client, handler = _image_generation_client(
        monkeypatch,
        UserAPIKeyAuth(router_settings={"fallbacks": [{"A": ["B"]}]}),
        working_params={"success_callback": [on_working_success]},
        failing_params={"success_callback": [on_failing_success], "failure_callback": [on_failing_failure]},
    )

    with client:
        response: Final = client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree"})
        assert fired["failing_failure"].wait(timeout=5), response.text
        assert fired["working_success"].wait(timeout=5), response.text

    assert handler.calls == ["failing", "working"]
    assert not fired["failing_success"].is_set()


def test_repeated_image_requests_register_a_deployments_named_logger_once(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("input_callback", "success_callback", "failure_callback", "_async_success_callback", "callbacks"):
        monkeypatch.setattr(litellm, name, [])
    client, _ = _image_generation_client(
        monkeypatch, UserAPIKeyAuth(), working_params={"callbacks": ["anthropic_cache_control_hook"]}
    )

    with client:
        responses: Final = tuple(
            client.post("/v1/images/generations", json={"model": "B", "prompt": "draw a tree"}) for _ in range(2)
        )

    assert [response.status_code for response in responses] == [200, 200]
    registered: Final = tuple(type(callback).__name__ for callback in litellm._async_success_callback)
    assert registered == ("AnthropicCacheControlHook",)


@pytest.mark.parametrize(
    "send",
    (pytest.param(_send_generation, id="generations"), pytest.param(_send_edit, id="edits")),
)
def test_refused_image_request_returns_the_refusal_details_to_the_client(
    monkeypatch: pytest.MonkeyPatch, send: Callable[[TestClient], httpx.Response]
) -> None:
    from litellm.llms.gemini.empty_image_response import ImageRefusalError

    details: Final = {"finish_reason": "IMAGE_SAFETY", "finish_message": "try another prompt"}
    monkeypatch.setattr(litellm, "callbacks", [])
    client, handler = _image_generation_client(monkeypatch, UserAPIKeyAuth(api_key="sk-test"))
    handler.refusal = ImageRefusalError(
        message="Gemini refused to return an image, finishReason=IMAGE_SAFETY",
        model="image-regression/failing",
        llm_provider="gemini",
        response=httpx.Response(400, request=httpx.Request("POST", "https://google.invalid")),
        provider_specific_fields=details,
        billed_answer=None,
    )
    try:
        with client:
            response: Final = send(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 400, response.text
    assert response.json()["error"]["provider_specific_fields"] == details


def test_failed_image_generation_that_billed_nothing_sends_no_cost_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    client, _ = _image_generation_client(monkeypatch, UserAPIKeyAuth(api_key="sk-test"))
    try:
        with client:
            response: Final = _send_generation(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 429, response.text
    assert "x-litellm-call-id" in response.headers
    assert "x-litellm-response-cost" not in response.headers


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
    client, handler = _image_generation_client(
        monkeypatch, UserAPIKeyAuth(api_key="sk-test", router_settings={"fallbacks": [{"A": ["B"]}]})
    )
    handler.owed_after_failure = (litellm.Usage(prompt_tokens=16, completion_tokens=0, total_tokens=16), 0.25)
    handler.served_cost = served_cost
    try:
        with client:
            response: Final = send(client)
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 200, response.text
    assert handler.calls == ["failing", "working"]
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(billed)


def test_image_edit_json_body_with_image_references_reaches_openai_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """regression: a JSON edit request with image and mask references reaches OpenAI as the same JSON body."""
    from litellm.proxy import proxy_server

    images: Final = [
        {"image_url": "https://images.example/tree.png"},
        {"image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="},
    ]
    mask: Final = {"image_url": "https://images.example/mask.png"}
    cache: Final = UserApiKeyCache()
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-image-1",
                "litellm_params": {
                    "model": "openai/gpt-image-1",
                    "api_key": "sk-upstream",
                    "api_base": "https://edit.example/v1",
                },
            }
        ],
        num_retries=0,
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_alias_map", {})
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", ProxyConfig())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=cache))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server, "version", "test-version")
    app: Final = FastAPI()
    app.include_router(endpoints.router)
    app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth()

    with respx.mock(assert_all_called=False) as respx_mock:
        upstream: Final = respx_mock.post("https://edit.example/v1/images/edits").mock(
            return_value=httpx.Response(200, json={"created": 1712697600, "data": [{"b64_json": "ZWRpdGVk"}]})
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            response: Final = client.post(
                "/v1/images/edits",
                json={"model": "gpt-image-1", "prompt": "add a hat", "images": images, "mask": mask},
            )

    assert [call.request.headers["content-type"].split(";")[0] for call in upstream.calls] == ["application/json"], (
        response.status_code,
        response.text,
        [call.request.content for call in upstream.calls],
    )
    assert orjson.loads(upstream.calls.last.request.content) == {
        "model": "gpt-image-1",
        "prompt": "add a hat",
        "images": images,
        "mask": mask,
    }
    assert response.status_code == 200, response.text
    assert [image["b64_json"] for image in response.json()["data"]] == ["ZWRpdGVk"]


_STREAMED_IMAGE: Final = "RklOQUw="
_STREAMED_USAGE: Final = {
    "total_tokens": 220,
    "input_tokens": 20,
    "output_tokens": 200,
    "input_tokens_details": {"text_tokens": 20, "image_tokens": 0},
}
_STREAM_ROUTES: Final = {"generation": ("generations", "image_generation"), "edit": ("edits", "image_edit")}


def _image_event(event_type: str, **fields: object) -> bytes:
    payload: Final = {
        "type": event_type,
        "created_at": 1700000000,
        "size": "1024x1024",
        "quality": "low",
        "background": "opaque",
        "output_format": "png",
        **fields,
    }
    return f"event: {event_type}\ndata: {orjson.dumps(payload).decode()}\n\n".encode()


def _streaming_image_client(monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, kind: str) -> TestClient:
    """A proxy with one OpenAI image deployment whose provider answers with a partial and a completed event."""
    from litellm.proxy import proxy_server

    path, prefix = _STREAM_ROUTES[kind]
    respx_mock.post(f"https://openai.test/v1/images/{path}").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_image_event(f"{prefix}.partial_image", b64_json="UA==", partial_image_index=0)
            + _image_event(f"{prefix}.completed", b64_json=_STREAMED_IMAGE, usage=_STREAMED_USAGE),
        )
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    cache: Final = UserApiKeyCache()
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "image-model",
                "litellm_params": {
                    "model": "openai/gpt-image-1",
                    "api_key": "test",
                    "api_base": "https://openai.test/v1",
                },
            }
        ],
        num_retries=0,
    )
    monkeypatch.setattr(litellm, "model_alias_map", {})
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", ProxyConfig())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=cache))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server, "version", "test-version")

    app: Final = FastAPI()
    app.include_router(endpoints.router)
    app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth()
    return TestClient(app, raise_server_exceptions=False)


def _streamed_image_request(client: TestClient, kind: str) -> httpx.Response:
    if kind == "generation":
        return client.post(
            "/v1/images/generations",
            json={"model": "image-model", "prompt": "draw a tree", "stream": True, "partial_images": 1},
        )
    return client.post(
        "/v1/images/edits",
        data={"model": "image-model", "prompt": "make it blue", "stream": "true", "partial_images": "1"},
        files={"image": ("tree.png", b"\x89PNG\r\n\x1a\ntree", "image/png")},
    )


def _sse_events(body: str) -> list[tuple[str, dict[str, object]]]:
    frames: Final = [frame.split("\n") for frame in body.split("\n\n") if frame]
    return [(lines[0].removeprefix("event: "), orjson.loads(lines[1].removeprefix("data: "))) for lines in frames]


class _ImageOutputGuardrail(CustomGuardrail):
    def __init__(self, blocks: bool) -> None:
        super().__init__(guardrail_name="image-output", event_hook=GuardrailEventHooks.post_call, default_on=True)
        self.blocks: Final = blocks
        self.checked: Queue[object] = Queue()

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        self.checked.put(response)
        if self.blocks:
            raise HTTPException(status_code=400, detail={"error": "image blocked by guardrail"})
        return response


@pytest.mark.parametrize("kind", ["generation", "edit"])
def test_streamed_image_request_is_answered_with_the_provider_events(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, kind: str
) -> None:
    prefix: Final = _STREAM_ROUTES[kind][1]
    with _streaming_image_client(monkeypatch, respx_mock, kind) as client:
        response: Final = _streamed_image_request(client, kind)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events: Final = _sse_events(response.text)
    assert [name for name, _ in events] == [f"{prefix}.partial_image", f"{prefix}.completed"]
    assert [payload["type"] for _, payload in events] == [f"{prefix}.partial_image", f"{prefix}.completed"]
    assert events[-1][1]["b64_json"] == _STREAMED_IMAGE
    assert events[-1][1]["usage"]["total_tokens"] == _STREAMED_USAGE["total_tokens"]


@pytest.mark.parametrize("kind", ["generation", "edit"])
def test_streamed_image_reaches_the_client_only_after_a_post_call_guardrail_passes_it(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, kind: str
) -> None:
    prefix: Final = _STREAM_ROUTES[kind][1]
    guardrail: Final = _ImageOutputGuardrail(blocks=False)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    with _streaming_image_client(monkeypatch, respx_mock, kind) as client:
        response: Final = _streamed_image_request(client, kind)

    assert response.status_code == 200, response.text
    events: Final = _sse_events(response.text)
    assert [name for name, _ in events] == [f"{prefix}.completed"]
    assert events[0][1]["b64_json"] == _STREAMED_IMAGE
    checked: Final = guardrail.checked.get_nowait()
    assert isinstance(checked, ImageResponse)
    assert checked.data[0].b64_json == _STREAMED_IMAGE


@pytest.mark.parametrize("kind", ["generation", "edit"])
def test_streamed_image_blocked_by_a_post_call_guardrail_never_reaches_the_client(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, kind: str
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_ImageOutputGuardrail(blocks=True)])
    with _streaming_image_client(monkeypatch, respx_mock, kind) as client:
        response: Final = _streamed_image_request(client, kind)

    assert response.status_code == 400, response.text
    assert "image blocked by guardrail" in response.text
    assert _STREAMED_IMAGE not in response.text
    assert "UA==" not in response.text


class _ImageReplacingLogger(CustomLogger):
    def __init__(self, returns_dict: bool) -> None:
        super().__init__()
        self.returns_dict: Final = returns_dict

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        assert isinstance(response, ImageResponse)
        replaced: Final = ImageResponse(
            created=response.created, data=[ImageObject(b64_json="UkVQTEFDRUQ=")], usage=response.usage
        )
        return replaced.model_dump() if self.returns_dict else replaced


@pytest.mark.parametrize("returns_dict", [False, True], ids=["image_response", "dict"])
@pytest.mark.parametrize("kind", ["generation", "edit"])
def test_streamed_image_replaced_by_a_post_call_hook_is_the_one_the_client_gets(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, kind: str, returns_dict: bool
) -> None:
    prefix: Final = _STREAM_ROUTES[kind][1]
    monkeypatch.setattr(litellm, "callbacks", [_ImageReplacingLogger(returns_dict)])
    with _streaming_image_client(monkeypatch, respx_mock, kind) as client:
        response: Final = _streamed_image_request(client, kind)

    assert response.status_code == 200, response.text
    events: Final = _sse_events(response.text)
    assert [name for name, _ in events] == [f"{prefix}.partial_image", f"{prefix}.completed"]
    assert events[-1][1]["b64_json"] == "UkVQTEFDRUQ="
    assert events[-1][1]["usage"]["total_tokens"] == _STREAMED_USAGE["total_tokens"]
    assert _STREAMED_IMAGE not in response.text


@pytest.mark.parametrize("kind", ["generation", "edit"])
def test_streamed_image_request_answered_with_json_gets_every_image_as_a_completed_event(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, kind: str
) -> None:
    path, prefix = _STREAM_ROUTES[kind]
    with _streaming_image_client(monkeypatch, respx_mock, kind) as client:
        respx_mock.post(f"https://openai.test/v1/images/{path}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "created": 1700000000,
                    "data": [{"b64_json": "Rklyc3Q="}, {"b64_json": _STREAMED_IMAGE}],
                    "usage": _STREAMED_USAGE,
                },
            )
        )
        response: Final = _streamed_image_request(client, kind)

    assert response.status_code == 200, response.text
    events: Final = _sse_events(response.text)
    assert [name for name, _ in events] == [f"{prefix}.completed", f"{prefix}.completed"]
    assert [payload["b64_json"] for _, payload in events] == ["Rklyc3Q=", _STREAMED_IMAGE]
    assert ["usage" in payload for _, payload in events] == [False, True]
    assert events[-1][1]["usage"]["total_tokens"] == _STREAMED_USAGE["total_tokens"]


_OPENAI_STREAMING_BACKUP: Final = MappingProxyType(
    {"model": "openai/gpt-image-1", "api_key": "test", "api_base": "https://openai.test/v1"}
)


@pytest.mark.parametrize("kind", ["generation", "edit"])
@pytest.mark.parametrize(
    ("backup", "cost_header"),
    (
        pytest.param(MappingProxyType({}), 0.75, id="backup_answers_one_image"),
        pytest.param(_OPENAI_STREAMING_BACKUP, None, id="backup_streams_events"),
    ),
)
def test_streamed_image_cost_header_after_a_fallback_is_the_whole_bill_or_absent(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    kind: str,
    backup: Mapping[str, object],
    cost_header: float | None,
) -> None:
    path, prefix = _STREAM_ROUTES[kind]
    respx_mock.post(f"https://openai.test/v1/images/{path}").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_image_event(f"{prefix}.completed", b64_json=_STREAMED_IMAGE, usage=_STREAMED_USAGE),
        )
    )
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    client, handler = _image_generation_client(
        monkeypatch,
        UserAPIKeyAuth(api_key="sk-test", router_settings={"fallbacks": [{"A": ["B"]}]}),
        working_params=backup,
    )
    handler.owed_after_failure = (litellm.Usage(prompt_tokens=16, completion_tokens=0, total_tokens=16), 0.25)
    handler.served_cost = 0.5
    try:
        with client:
            response: Final = (
                client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree", "stream": True})
                if kind == "generation"
                else client.post(
                    "/v1/images/edits",
                    data={"model": "A", "prompt": "make it blue", "stream": "true"},
                    files={"image": ("tree.png", b"\x89PNG\r\n\x1a\ntree", "image/png")},
                )
            )
    finally:
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert handler.calls[0] == "failing"
    sent_cost: Final = response.headers.get("x-litellm-response-cost")
    assert (None if sent_cost is None else float(sent_cost)) == cost_header


class _StreamFlagRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.stream_flags: Queue[object] = Queue()

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        self.stream_flags.put(data.get("stream"))
        return response


def test_image_generation_streamed_through_extra_body_is_answered_and_checked_as_a_stream(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    recorder: Final = _StreamFlagRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    with _streaming_image_client(monkeypatch, respx_mock, "generation") as client:
        response: Final = client.post(
            "/v1/images/generations",
            json={"model": "image-model", "prompt": "draw a tree", "extra_body": {"stream": True}},
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events: Final = _sse_events(response.text)
    assert [name for name, _ in events] == ["image_generation.partial_image", "image_generation.completed"]
    assert events[-1][1]["b64_json"] == _STREAMED_IMAGE
    assert recorder.stream_flags.get_nowait() is True
