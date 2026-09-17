from __future__ import annotations

import asyncio
import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

import httpx
from fastapi import Request
from pydantic import JsonValue, TypeAdapter
from starlette.types import ASGIApp

import litellm
from litellm._uuid import uuid
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.litellm_core_utils.internal_call_metadata import (
    effective_turn_off_message_logging,
    parent_session_kwargs,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    RequestRateLimiterStash,
    finish_summary_cleanup,
    get_request_stash,
    isolated_summary_request_stash,
)
from litellm.proxy.spend_tracking.budget_reservation import (
    release_budget_reservation_on_cancel,  # pyright: ignore[reportUnknownVariableType]  # legacy reservation dict is untyped
)
from litellm.router_strategy.complexity_router.context_compaction import NativeRequest
from litellm.types.utils import CONTEXT_COMPACTION_CALL_ORIGIN

_ASGI_APP: Final[TypeAdapter[ASGIApp]] = TypeAdapter(ASGIApp)
_STRING: Final = TypeAdapter(str)
_JSON_BODY: Final = TypeAdapter(Mapping[str, JsonValue])
_HEADERS: Final = TypeAdapter(Mapping[str, str])
_METADATA: Final = TypeAdapter(Mapping[str, object])
_NATIVE_BODY_FIELDS: Final = frozenset(
    {"input", "instructions", "messages", "max_tokens", "system", "tools", "compaction"}
)
_OMITTED_HEADERS: Final = frozenset(
    {
        "content-length",
        "content-type",
        "transfer-encoding",
        "x-litellm-call-id",
        "x-app",
        "x-claude-code-agent-id",
        "x-litellm-model",
        "x-litellm-model-id",
        "litellm-disable-message-redaction",
    }
)


@runtime_checkable
class SummaryRequestLimiter(Protocol):
    async def release_summary_request_capacity(
        self, auth: UserAPIKeyAuth, request_data: Mapping[str, object], provider_completed: bool
    ) -> None: ...


@dataclass(slots=True)
class _SummaryRequest:
    call_id: str
    redact: bool | None
    auth: UserAPIKeyAuth | None = None
    data: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    provider_completed: bool = False
    closed: asyncio.Event = field(default_factory=asyncio.Event)

    def bind(self, auth: UserAPIKeyAuth, data: Mapping[str, object]) -> None:
        self.auth = auth
        self.data = data
        if (
            self.redact is True
            or effective_turn_off_message_logging(data) is True
            or litellm.turn_off_message_logging is True
        ):
            self.redact = True


_summary_request: Final[ContextVar[_SummaryRequest | None]] = ContextVar(
    "litellm_context_compaction_request", default=None
)


def summary_request_updates(
    request: Request,
    auth: UserAPIKeyAuth,
    data: Mapping[str, object],
    metadata_key: str,
) -> Mapping[str, object]:
    summary: Final = _summary_request.get()
    if summary is None or summary.closed.is_set() or request.headers.get("x-litellm-call-id") != summary.call_id:
        return MappingProxyType({})
    summary.bind(auth, data)
    metadata: Final = _METADATA.validate_python(data.get(metadata_key) or MappingProxyType({}))
    stamped_metadata: Final = _METADATA.validate_python(
        MappingProxyType({**metadata, INTERNAL_CALL_ORIGIN_METADATA_KEY: CONTEXT_COMPACTION_CALL_ORIGIN})
    )
    return MappingProxyType(
        {
            name: value
            for name, value in (
                (metadata_key, stamped_metadata),
                ("num_retries", 0),
                ("max_retries", 0),
                ("disable_fallbacks", True),
                ("turn_off_message_logging", True if summary.redact is True else None),
            )
            if value is not None
        }
    )


def mark_summary_provider_completed() -> None:
    summary: Final = _summary_request.get()
    if summary is not None and not summary.closed.is_set():
        summary.provider_completed = True


def summary_request_is_private() -> bool:
    summary: Final = _summary_request.get()
    return summary is not None and not summary.closed.is_set() and summary.redact is True


async def _settle_cancelled_summary(
    summary: _SummaryRequest,
    limiter: SummaryRequestLimiter | None,
) -> None:
    auth: Final = summary.auth
    if auth is None:
        return
    if not summary.provider_completed:
        await release_budget_reservation_on_cancel(auth.budget_reservation)
    if limiter is not None:
        await limiter.release_summary_request_capacity(auth, summary.data, summary.provider_completed)


@dataclass(frozen=True, slots=True)
class ProxySummaryExecutor:
    request: Request
    request_data: Mapping[str, object]
    limiter: SummaryRequestLimiter | None
    parent_stash: RequestRateLimiterStash | None
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    call_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def __call__(self, model: str, request: NativeRequest, timeout: float) -> Mapping[str, object]:
        if self.closed.is_set() or _summary_request.get() is not None:
            raise ValueError("Context compaction executor is closed or nested")
        return await asyncio.wait_for(self._serialized_execute(model, request, timeout), timeout)

    async def _serialized_execute(self, model: str, request: NativeRequest, timeout: float) -> Mapping[str, object]:
        async with self.call_lock:
            if self.closed.is_set():
                raise ValueError("Context compaction executor is closed")
            return await self._execute(model, request, timeout)

    async def _execute(self, model: str, request: NativeRequest, timeout: float) -> Mapping[str, object]:
        dynamic_redact: Final = effective_turn_off_message_logging(self.request_data)
        redact: Final = True if litellm.turn_off_message_logging is True else dynamic_redact
        summary: Final = _SummaryRequest(call_id=str(uuid.uuid4()), redact=redact)
        parent_call_id: Final = self.request_data.get("litellm_call_id")
        scope: Final = _METADATA.validate_python(self.request.scope)
        root_path: Final = _STRING.validate_python(scope.get("root_path", "")).rstrip("/")
        endpoint: Final = "/v1/responses/compact" if request.protocol == "responses" else "/v1/messages"
        url: Final = str(self.request.url.replace(path=f"{root_path}{endpoint}"))
        native_headers: Final = httpx.Headers(
            _HEADERS.validate_python(request.payload.get("extra_headers") or MappingProxyType({}))
        )
        beta: Final = ",".join(
            dict.fromkeys(
                token.strip()
                for value in (self.request.headers.get("anthropic-beta"), native_headers.get("anthropic-beta"))
                if value
                for token in value.split(",")
                if token.strip()
            )
        )
        headers: Final = httpx.Headers(
            tuple(
                (name, value)
                for name, value in self.request.headers.items()
                if name.lower() not in _OMITTED_HEADERS and name.lower() != "anthropic-beta"
            )
            + ((("anthropic-beta", beta),) if beta else ())
            + (("x-litellm-call-id", summary.call_id), ("content-type", "application/json"))
        )
        session: Final = parent_session_kwargs(self.request_data)
        region: Final = self.request_data.get("allowed_model_region")
        user: Final = self.request_data.get("user")
        payload: Final = MappingProxyType(
            {
                **MappingProxyType(
                    {name: value for name, value in request.payload.items() if name in _NATIVE_BODY_FIELDS}
                ),
                "allowed_model_region": region if isinstance(region, str) else None,
                "user": user if isinstance(user, str) else None,
                "litellm_session_id": session.get("litellm_session_id"),
                "litellm_trace_id": session.get("litellm_trace_id"),
                "model": model,
                "timeout": timeout,
                "stream": False if request.protocol == "messages" else None,
                "num_retries": 0,
                "max_retries": 0,
                "disable_fallbacks": True,
                "metadata": MappingProxyType({"tags": tuple(get_tags_from_request_body(self.request_data))}),
            }
        )
        body: Final = json.dumps(
            MappingProxyType({name: value for name, value in payload.items() if value is not None}),
            default=dict,
            allow_nan=False,
        ).encode()
        address: Final = self.request.client
        transport: Final = httpx.ASGITransport(
            app=_ASGI_APP.validate_python(scope["app"]),
            root_path=root_path,
            client=(address.host, address.port) if address is not None else ("127.0.0.1", 0),
            raise_app_exceptions=False,
        )
        async with isolated_summary_request_stash(
            self.parent_stash, parent_call_id if isinstance(parent_call_id, str) else None, summary.call_id
        ):
            token: Final = _summary_request.set(summary)
            try:
                async with transport:
                    response: Final = await transport.handle_async_request(
                        httpx.Request("POST", url, headers=headers, content=body)
                    )
                    try:
                        if not response.is_success:
                            raise litellm.APIError(
                                status_code=response.status_code,
                                message="Context compaction request failed",
                                model=model,
                                llm_provider="litellm",
                            )
                        return _JSON_BODY.validate_json(await response.aread())
                    finally:
                        await response.aclose()
            except (asyncio.CancelledError, TimeoutError):
                await finish_summary_cleanup(_settle_cancelled_summary(summary, self.limiter))
                raise
            finally:
                summary.closed.set()
                _summary_request.reset(token)


@contextmanager
def proxy_summary_executor_scope(
    request: Request,
    request_data: Mapping[str, object],
    limiter: object,
) -> Generator[None]:
    from litellm.router_strategy.complexity_router.context_compaction import use_summary_executor

    if _summary_request.get() is not None:
        yield
        return
    executor: Final = ProxySummaryExecutor(
        request=request,
        request_data=request_data,
        limiter=limiter if isinstance(limiter, SummaryRequestLimiter) else None,
        parent_stash=get_request_stash(),
    )
    try:
        with use_summary_executor(executor):
            yield
    finally:
        executor.closed.set()
