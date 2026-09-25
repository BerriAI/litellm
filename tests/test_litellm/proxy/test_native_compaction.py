import asyncio
from collections.abc import Awaitable, Mapping
from types import MappingProxyType
from typing import Final, Literal

import pytest
from fastapi import FastAPI, Request
from pydantic import TypeAdapter

import litellm
from litellm.caching.caching import DualCache
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.initialize_dynamic_callback_params import inherit_message_logging_privacy
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.redact_messages import should_redact_message_logging
from litellm.proxy import common_request_processing, proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_model
from litellm.proxy._types import ProxyException
from litellm.proxy.hooks.parallel_request_limiter_v3 import get_or_create_request_stash, get_request_stash
from litellm.proxy.native_compaction import with_proxy_compaction_executor
from litellm.router import Router
from litellm.router_strategy.complexity_router.context_compaction import compaction_executor, reject_recursive_compactor
from litellm.types.utils import ModelResponse

_HEADERS: Final = (
    (b"authorization", b"Bearer sk-compaction-fixture"), (b"cookie", b"session=fixture"),
    (b"content-length", b"99999"), (b"x-litellm-call-id", b"parent"),
    (b"litellm-disable-message-redaction", b"true"), (b"x-litellm-num-retries", b"8"),
    (b"X-LiteLLM-Timeout", b"600"), (b"x-litellm-stream-timeout", b"500"),
)


async def _child(
    protocol: Literal["chat", "messages"] = "chat", forged: bool = False, parent_model: str | None = None
) -> Mapping[str, object]:
    executor: Final = compaction_executor.get()
    assert executor is not None
    payload: Final = TypeAdapter(Mapping[str, object]).validate_json(
        b'{"model":"compactor","messages":[{"role":"user","content":"history"}],'
        b'"num_retries":0,"timeout":7,"stream_timeout":7,"disable_fallbacks":true,"stream":false,'
        b'"metadata":{"turn_off_message_logging":true}}'
    )
    return await executor(protocol, MappingProxyType({
        "litellm_metadata" if protocol == "messages" and key == "metadata" else key: value
        for key, value in payload.items() if forged or key != "metadata"
    }), parent_model)


def _request(app: FastAPI) -> Request:
    return Request(TypeAdapter(dict[str, object]).validate_python(MappingProxyType({
        "type": "http", "app": app, "scheme": "https", "server": ("proxy.test", 443),
        "path": "/gateway/parent", "root_path": "/gateway", "query_string": b"parent=1",
        "client": ("192.0.2.1", 4321), "headers": _HEADERS,
    })))


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ("chat", "messages"))
async def test_child_preserves_credentials_and_isolates_context(protocol: Literal["chat", "messages"]) -> None:
    app: Final = FastAPI()
    stash: Final = get_or_create_request_stash()

    @app.post("/v1/chat/completions" if protocol == "chat" else "/v1/messages")
    async def endpoint(request: Request) -> Mapping[str, object]:
        assert get_request_stash() is None and compaction_executor.get() is None
        assert request.client == ("192.0.2.1", 4321) and request.url.scheme == "https"
        assert request.scope["root_path"] == "/gateway" and request.cookies["session"] == "fixture"
        assert request.headers["authorization"] == "Bearer sk-compaction-fixture" and not request.query_params
        assert "x-litellm-call-id" not in request.headers
        assert "litellm-disable-message-redaction" not in request.headers
        assert int(request.headers["content-length"]) == len(await request.body())
        with pytest.raises(BadRequestError, match="regular model group"):
            reject_recursive_compactor("auto-router")
        return MappingProxyType({"summary": "compacted"})

    with inherit_message_logging_privacy(True):
        assert (await with_proxy_compaction_executor(_child(protocol), _request(app)))["summary"] == "compacted"
    assert get_request_stash() is stash and compaction_executor.get() is None
    reject_recursive_compactor("auto-router")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ("chat", "messages"))
@pytest.mark.parametrize("policy", ("allowed", "denied", "forged", "router_alias", "unrelated_alias"))
async def test_real_proxy_child_auth_privacy_and_body_policy(
    monkeypatch: pytest.MonkeyPatch, protocol: Literal["chat", "messages"], policy: str,
) -> None:
    cache: Final = DualCache()
    token: Final = proxy_server.hash_token("sk-compaction-fixture")
    models: Final = {"denied": ("answer",), "router_alias": ("auto",), "unrelated_alias": ("other-auto",)}.get(policy, ("compactor",))
    auth: Final = UserAPIKeyAuth.model_validate(MappingProxyType({"token": token, "models": models}))
    await cache.async_set_cache(key=token, value=auth)
    dispatched: Final = asyncio.Event()
    allowed: Final = policy in ("allowed", "router_alias")

    async def route(
        data: Mapping[str, object], llm_router: Router | None, user_model: str | None,
        route_type: str, user_api_key_dict: UserAPIKeyAuth | None,
    ) -> Awaitable[ModelResponse]:
        dispatched.set()
        assert allowed
        if policy == "router_alias":
            with pytest.raises(ProxyException):
                await can_key_call_model("unrelated-compactor", None, auth, None)
        assert (data["num_retries"], data["timeout"], data["stream_timeout"]) == (0, 7, 7)
        assert data["disable_fallbacks"] is True and data["stream"] is False
        logging: Final = data["litellm_logging_obj"]
        assert isinstance(logging, Logging)
        assert logging.standard_callback_dynamic_params.get("turn_off_message_logging") is True
        assert should_redact_message_logging(TypeAdapter(dict[str, object]).validate_python(MappingProxyType({
            "litellm_params": data, "standard_callback_dynamic_params": logging.standard_callback_dynamic_params,
        })))
        return asyncio.sleep(0, result=ModelResponse(id="private-summary", model="compactor"))

    monkeypatch.setattr(litellm, "max_budget", 0)
    monkeypatch.setattr(proxy_server.app, "dependency_overrides", {})
    monkeypatch.setattr(proxy_server, "master_key", "sk-master-fixture")
    monkeypatch.setattr(litellm, "max_budget", 0.0)
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    # Pin the proxy-wide budget: authentication only reads the global spend when a
    # proxy max budget is configured, and that read goes through the stub prisma
    # client above. A budget left set by an earlier test on the same worker would
    # turn this fixture's child requests into 401s.
    monkeypatch.setattr(litellm, "max_budget", 0.0)
    monkeypatch.setattr(common_request_processing, "route_request", route)
    with inherit_message_logging_privacy(True):
        call: Final = with_proxy_compaction_executor(
            _child(protocol, policy == "forged", "auto" if policy.endswith("alias") else None), _request(proxy_server.app)
        )
        if allowed:
            assert (await call)["id"] == "private-summary"
        else:
            status: Final = 401 if policy == "forged" else 403
            with pytest.raises(BadRequestError, match=rf"child request failed \(HTTP {status}\)"):
                await call
    assert dispatched.is_set() is allowed
    assert compaction_executor.get() is None
    if policy.endswith("alias"):
        with pytest.raises(ProxyException):
            await can_key_call_model("compactor", None, auth, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_cancelling_parent_cancels_and_drains_child(timeout: bool) -> None:
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

    parent: Final = asyncio.create_task(with_proxy_compaction_executor(_child(), _request(app)))
    await asyncio.wait_for(started.wait(), timeout=5)
    if timeout:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(parent, timeout=0)
    else:
        parent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent
    assert stopped.is_set() and compaction_executor.get() is None
