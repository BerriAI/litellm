import asyncio
from collections.abc import Awaitable, Mapping
from types import MappingProxyType
from typing import Final, Literal
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from pydantic import TypeAdapter

from litellm.exceptions import BadRequestError
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    inherit_message_logging_privacy,
    initialize_standard_callback_dynamic_params,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.redact_messages import should_redact_message_logging
from litellm.proxy import common_request_processing, proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import get_or_create_request_stash, get_request_stash
from litellm.proxy.native_compaction import with_proxy_compaction_executor
from litellm.router import Router
from litellm.router_strategy.complexity_router.context_compaction import compaction_executor, reject_recursive_compactor
from litellm.types.utils import ModelResponse


async def _child(protocol: Literal["chat", "messages"] = "chat") -> Mapping[str, object]:
    executor: Final = compaction_executor.get()
    assert executor is not None
    metadata: Final = TypeAdapter(dict[str, str]).validate_json(b'{"user_api_key_user_id":"fixture-user"}')
    return await executor(protocol, MappingProxyType({
        "model": "compactor", "metadata" if protocol == "chat" else "litellm_metadata": metadata,
    }))


def _request(app: FastAPI, headers: tuple[tuple[bytes, bytes], ...] = ()) -> Request:
    return Request(TypeAdapter(dict[str, object]).validate_python(MappingProxyType({
        "type": "http", "app": app, "scheme": "https", "server": ("proxy.test", 443),
        "path": "/gateway/parent", "root_path": "/gateway", "query_string": b"",
        "client": ("192.0.2.1", 4321), "headers": headers,
    })))


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ("chat", "messages"))
async def test_child_preserves_credentials_and_isolates_context(protocol: Literal["chat", "messages"]) -> None:
    app: Final = FastAPI()
    stash: Final = get_or_create_request_stash()
    path: Final = "/v1/chat/completions" if protocol == "chat" else "/v1/messages"

    @app.post(path)
    async def endpoint(request: Request) -> Mapping[str, object]:
        assert get_request_stash() is None and compaction_executor.get() is None
        assert request.client == ("192.0.2.1", 4321) and request.url.scheme == "https"
        assert request.scope["root_path"] == "/gateway" and request.cookies["session"] == "fixture"
        assert request.headers["authorization"] == "Bearer sk-compaction-fixture"
        assert "x-litellm-call-id" not in request.headers
        assert "litellm-disable-message-redaction" not in request.headers
        assert int(request.headers["content-length"]) == len(await request.body())
        payload: Final = TypeAdapter(Mapping[str, object]).validate_json(await request.body())
        assert payload["metadata" if protocol == "chat" else "litellm_metadata"] == MappingProxyType(
            {"user_api_key_user_id": "fixture-user"}
        )
        with pytest.raises(BadRequestError, match="regular model group"):
            reject_recursive_compactor("auto-router")
        return MappingProxyType({"summary": "compacted"})

    request: Final = _request(app, (
        (b"authorization", b"Bearer sk-compaction-fixture"), (b"cookie", b"session=fixture"),
        (b"content-length", b"99999"), (b"x-litellm-call-id", b"parent"),
        (b"litellm-disable-message-redaction", b"true"),
    ))
    with inherit_message_logging_privacy(True):
        assert (await with_proxy_compaction_executor(_child(protocol), request))["summary"] == "compacted"
    assert get_request_stash() is stash and compaction_executor.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ("chat", "messages"))
async def test_real_proxy_denies_compactor_outside_key_models(
    monkeypatch: pytest.MonkeyPatch, protocol: Literal["chat", "messages"]
) -> None:
    monkeypatch.setattr(proxy_server.app, "dependency_overrides", {})
    cache: Final = DualCache()
    token: Final = proxy_server.hash_token("sk-compaction-fixture")
    await cache.async_set_cache(key=token, value=UserAPIKeyAuth.model_validate(
        MappingProxyType({"token": token, "models": ("answer",)})
    ))
    monkeypatch.setattr(proxy_server, "master_key", "sk-master-fixture")
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "llm_router", None)
    paid_call: Final = AsyncMock(side_effect=AssertionError("Denied compactor reached inference"))
    monkeypatch.setattr(common_request_processing, "route_request", paid_call)
    request: Final = _request(proxy_server.app, ((b"authorization", b"Bearer sk-compaction-fixture"),))
    with pytest.raises(BadRequestError, match=r"child request failed \(HTTP 403\)"):
        await with_proxy_compaction_executor(_child(protocol), request)
    paid_call.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("forged_wire_flag", (False, True))
@pytest.mark.parametrize("protocol", ("chat", "messages"))
async def test_real_proxy_child_inherits_privacy_without_client_logging_controls(
    monkeypatch: pytest.MonkeyPatch, forged_wire_flag: bool, protocol: Literal["chat", "messages"]
) -> None:
    monkeypatch.setattr(proxy_server.app, "dependency_overrides", {})
    dispatched: Final = asyncio.Event()

    async def route(
        data: Mapping[str, object], llm_router: Router | None, user_model: str | None,
        route_type: str, user_api_key_dict: UserAPIKeyAuth | None,
    ) -> Awaitable[ModelResponse]:
        dispatched.set()
        logging: Final = data["litellm_logging_obj"]
        assert isinstance(logging, Logging)
        assert logging.standard_callback_dynamic_params.get("turn_off_message_logging") is True
        assert should_redact_message_logging(TypeAdapter(dict[str, object]).validate_python(MappingProxyType({
            "litellm_params": data, "standard_callback_dynamic_params": logging.standard_callback_dynamic_params,
        })))
        return asyncio.sleep(0, result=ModelResponse(id="private-summary", model="compactor"))

    async def private_child() -> Mapping[str, object]:
        executor: Final = compaction_executor.get()
        assert executor is not None
        payload: Final = TypeAdapter(Mapping[str, object]).validate_json(
            b'{"model":"compactor","messages":[{"role":"user","content":"history"}],'
            b'"metadata":{"turn_off_message_logging":true}}'
        )
        return await executor(protocol, MappingProxyType({
            "litellm_metadata" if protocol == "messages" and key == "metadata" else key: value
            for key, value in payload.items()
            if forged_wire_flag or key != "metadata"
        }))

    monkeypatch.setattr(proxy_server, "master_key", "sk-compaction-fixture")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "general_settings", TypeAdapter(dict[str, bool]).validate_python(
        MappingProxyType({})
    ))
    monkeypatch.setattr(common_request_processing, "route_request", route)
    request: Final = _request(proxy_server.app, ((b"authorization", b"Bearer sk-compaction-fixture"),))
    with inherit_message_logging_privacy(True):
        if forged_wire_flag:
            with pytest.raises(BadRequestError, match=r"child request failed \(HTTP 401\)"):
                await with_proxy_compaction_executor(private_child(), request)
            assert not dispatched.is_set()
            return
        assert (await with_proxy_compaction_executor(private_child(), request))["id"] == "private-summary"


@pytest.mark.asyncio
async def test_cancelling_parent_cancels_and_drains_child() -> None:
    app: Final = FastAPI()
    started: Final = asyncio.Event()
    stopped: Final = asyncio.Event()

    @app.post("/v1/chat/completions")
    async def endpoint() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    with inherit_message_logging_privacy(True):
        parent: Final = asyncio.create_task(with_proxy_compaction_executor(_child(), _request(app)))
    await asyncio.wait_for(started.wait(), timeout=5)
    parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent
    assert stopped.is_set() and compaction_executor.get() is None
    assert initialize_standard_callback_dynamic_params().get("turn_off_message_logging") is None
