from collections.abc import Awaitable, Mapping
from types import MappingProxyType
from typing import Final, Literal, TypeVar

from fastapi import Request
from pydantic import TypeAdapter, ValidationError

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.initialize_dynamic_callback_params import initialize_standard_callback_dynamic_params
from litellm.proxy.child_request import send_child_request
from litellm.router_strategy.complexity_router.context_compaction import (
    compaction_executor,
    native_compaction_call,
)

_ResultT: Final = TypeVar("_ResultT")
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])


async def with_proxy_compaction_executor(call: Awaitable[_ResultT], request: Request) -> _ResultT:
    async def execute(
        protocol: Literal["chat", "messages"], payload: Mapping[str, object], parent_model: str | None = None
    ) -> Mapping[str, object]:
        logging_disabled: Final = initialize_standard_callback_dynamic_params().get("turn_off_message_logging") is True
        response: Final = await send_child_request(
            request,
            "/v1/chat/completions" if protocol == "chat" else "/v1/messages",
            payload,
            logging_disabled=logging_disabled,
            scope=native_compaction_call(parent_model, str(payload["model"])),
        )
        if not 200 <= response.status_code < 300:
            raise BadRequestError(
                message=f"Native compaction child request failed (HTTP {response.status_code})",
                model="context_compaction",
                llm_provider="",
            )
        try:
            return MappingProxyType(_JSON_OBJECT.validate_json(response.body))
        except ValidationError:
            raise BadRequestError(
                message="Native compaction child returned an invalid JSON object",
                model="context_compaction",
                llm_provider="",
            ) from None

    token: Final = compaction_executor.set(execute)
    try:
        return await call
    finally:
        compaction_executor.reset(token)
