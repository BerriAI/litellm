import asyncio
from collections.abc import Awaitable, Mapping
from contextvars import Context
from types import MappingProxyType
from typing import Final, Literal, TypeVar

from fastapi import Request
from pydantic import TypeAdapter, ValidationError
from starlette.types import ASGIApp

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    inherit_message_logging_privacy,
    initialize_standard_callback_dynamic_params,
)
from litellm.llms.custom_httpx.asgi_handler import get_async_asgi_client
from litellm.proxy.litellm_pre_call_utils import UNTRUSTED_REQUEST_HEADER_CONTROL_FIELDS
from litellm.router_strategy.complexity_router.context_compaction import (
    compaction_executor,
    native_compaction_call,
)

_ResultT: Final = TypeVar("_ResultT")
_ASGI_APP: Final = TypeAdapter[ASGIApp](ASGIApp)
_ROOT_PATH: Final = TypeAdapter(str)
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
_REMOVED_HEADERS: Final = frozenset(
    (
        b"content-length",
        b"x-litellm-call-id",
        b"x-litellm-num-retries",
        b"x-litellm-timeout",
        b"x-litellm-stream-timeout",
    )
)


async def with_proxy_compaction_executor(call: Awaitable[_ResultT], request: Request) -> _ResultT:
    async def execute(
        protocol: Literal["chat", "messages"], payload: Mapping[str, object], parent_model: str | None = None
    ) -> Mapping[str, object]:
        logging_disabled: Final = initialize_standard_callback_dynamic_params().get("turn_off_message_logging") is True

        async def dispatch() -> Mapping[str, object]:
            scope: Final = _JSON_OBJECT.validate_python(request.scope)
            root_path: Final = _ROOT_PATH.validate_python(scope.get("root_path", ""))
            path: Final = "/v1/chat/completions" if protocol == "chat" else "/v1/messages"
            url: Final = str(request.url.replace(path=root_path.rstrip("/") + path, query="", fragment=""))
            headers: Final = tuple(
                (name, value)
                for name, value in request.headers.raw
                if name.lower() not in _REMOVED_HEADERS
                and not (logging_disabled and name.decode("latin-1").lower() in UNTRUSTED_REQUEST_HEADER_CONTROL_FIELDS)
            )
            with (
                native_compaction_call(parent_model, str(payload["model"])),
                inherit_message_logging_privacy(logging_disabled),
            ):
                with get_async_asgi_client(
                    app=_ASGI_APP.validate_python(scope["app"]),
                    root_path=root_path,
                    client=request.client,
                ) as client:
                    async with client.stream(
                        "POST", url, headers=headers, json=_JSON_OBJECT.validate_python(payload)
                    ) as response:
                        if not response.is_success:
                            raise BadRequestError(
                                message=f"Native compaction child request failed (HTTP {response.status_code})",
                                model="context_compaction",
                                llm_provider="",
                            )
                        body: Final = await response.aread()
                        try:
                            return MappingProxyType(_JSON_OBJECT.validate_json(body))
                        except ValidationError:
                            raise BadRequestError(
                                message="Native compaction child returned an invalid JSON object",
                                model="context_compaction",
                                llm_provider="",
                            ) from None

        task: Final = Context().run(asyncio.create_task, dispatch())
        return await task

    token: Final = compaction_executor.set(execute)
    try:
        return await call
    finally:
        compaction_executor.reset(token)
