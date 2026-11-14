import asyncio
import copy
import logging
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any, Dict, Final, cast

import orjson
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.image_endpoints import endpoints
from litellm.proxy.proxy_server import PrismaClient, ProxyConfig
from litellm.proxy.utils import ProxyLogging
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
        if model.endswith("failing"):
            raise litellm.RateLimitError(message="primary failed", llm_provider="image-regression", model=model)
        return ImageResponse(created=1, data=[ImageObject(url="https://images.example/generated.png")])


def _image_generation_client(
    monkeypatch: pytest.MonkeyPatch,
    user_api_key_dict: UserAPIKeyAuth,
    team_settings: Mapping[str, object] | None = None,
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
            key=f"team_id:{team_id}",
            value=LiteLLM_TeamTableCachedObj(team_id=team_id, router_settings=dict(team_settings)),
            local_only=True,
            model_type=LiteLLM_TeamTableCachedObj,
        )
        monkeypatch.setattr(proxy_server, "prisma_client", cast(PrismaClient, object()))

    router: Final = litellm.Router(
        model_list=[
            {"model_name": "A", "litellm_params": {"model": "image-regression/failing"}},
            {"model_name": "B", "litellm_params": {"model": "image-regression/working"}},
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
    app.dependency_overrides[user_api_key_auth] = lambda: user_api_key_dict
    return TestClient(app, raise_server_exceptions=False), handler


def test_image_generation_uses_key_router_settings_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    client, handler = _image_generation_client(
        monkeypatch,
        UserAPIKeyAuth(router_settings={"fallbacks": [{"A": ["B"]}]}),
    )

    response: Final = client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree"})

    assert response.status_code == 200, response.text
    assert response.json() == ImageResponse(
        created=1,
        data=[ImageObject(url="https://images.example/generated.png")],
    ).model_dump(), response.text
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
    assert response.json() == ImageResponse(
        created=1,
        data=[ImageObject(url="https://images.example/generated.png")],
    ).model_dump(), response.text
    assert handler.calls == ["failing", "working"], response.text


def test_image_generation_without_router_settings_does_not_use_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    client, handler = _image_generation_client(monkeypatch, UserAPIKeyAuth())

    response: Final = client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree"})

    assert response.status_code == 500, response.text
    assert handler.calls == ["failing"], response.text


def test_image_generation_uses_key_model_group_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    client, handler = _image_generation_client(
        monkeypatch,
        UserAPIKeyAuth(router_settings={"model_group_alias": {"alias": "B"}}),
    )

    response: Final = client.post("/v1/images/generations", json={"model": "alias", "prompt": "draw a tree"})

    assert response.status_code == 200, response.text
    assert response.json() == ImageResponse(
        created=1,
        data=[ImageObject(url="https://images.example/generated.png")],
    ).model_dump(), response.text
    assert handler.calls == ["working"], response.text


def test_image_generation_uses_key_num_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    client, handler = _image_generation_client(
        monkeypatch,
        UserAPIKeyAuth(router_settings={"num_retries": 2}),
    )

    response: Final = client.post("/v1/images/generations", json={"model": "A", "prompt": "draw a tree"})

    assert response.status_code == 500, response.text
    assert handler.calls == ["failing", "failing", "failing"], response.text
