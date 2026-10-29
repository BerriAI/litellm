import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextvars import ContextVar
from functools import wraps
from typing import Final, Literal, ParamSpec, TypeAlias, TypeVar

from fastapi import HTTPException

from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException, ModifyResponseException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks

MCPSurface: TypeAlias = Literal["managed", "streaming", "native", "rest"]
_SURFACE: Final[ContextVar[MCPSurface]] = ContextVar("mcp_block_policy_surface", default="native")
_HOOKS: Final = (GuardrailEventHooks.pre_mcp_call, GuardrailEventHooks.during_mcp_call)
_P = ParamSpec("_P")  # rebind-ok: ParamSpec declarations cannot carry a Final annotation
_T = TypeVar("_T")  # rebind-ok: TypeVar declarations cannot carry a Final annotation


class MCPGuardrailBlock(GuardrailRaisedException):
    def __init__(self, guardrail_name: str | None, event_hook: GuardrailEventHooks, message: str) -> None:
        super().__init__(guardrail_name=guardrail_name, message=message, blocked_content=True)
        self.guardrail_mode = event_hook.value


class UnsupportedMCPBlockPolicy(HTTPException):
    def __init__(self, guardrail_name: str | None, surface: MCPSurface) -> None:
        super().__init__(
            status_code=400,
            detail={
                "error": "unsupported_mcp_block_policy",
                "guardrail_name": guardrail_name,
                "message": f"on_mcp_block='raise' is not supported for {surface} MCP requests. "
                "Use a non-streaming LiteLLM-managed Chat Completions or Responses tool loop, "
                "or configure on_mcp_block='tool_error'.",
            },
        )


def mcp_execution_surface(surface: MCPSurface) -> Callable[[Callable[_P, Awaitable[_T]]], Callable[_P, Awaitable[_T]]]:
    def decorate(function: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
        @wraps(function)
        async def wrapped(
            *args: _P.args,
            **kwargs: _P.kwargs,  # kwargs-ok: ParamSpec preserves the wrapped callable contract
        ) -> _T:
            token: Final = _SURFACE.set(surface)
            try:
                return await function(*args, **kwargs)
            finally:
                _SURFACE.reset(token)

        return wrapped

    return decorate


def mcp_raise_callbacks(callbacks: Iterable[object]) -> tuple[CustomGuardrail, ...]:
    return tuple(
        callback for callback in callbacks if isinstance(callback, CustomGuardrail) and callback.on_mcp_block == "raise"
    )


async def gather_mcp_guardrails(calls: Iterable[Awaitable[object]]) -> None:
    results: Final = await asyncio.gather(*calls, return_exceptions=True)
    refusal: Final = next((result for result in results if isinstance(result, MCPGuardrailBlock)), None)
    if refusal is not None:
        raise refusal
    failure: Final = next((result for result in results if isinstance(result, BaseException)), None)
    if failure is not None:
        raise failure


def validate_mcp_block_policy(
    callbacks: Iterable[object], data: Mapping[str, object], surface: MCPSurface | None = None
) -> None:
    effective_surface: Final = surface or _SURFACE.get()
    if effective_surface in ("managed", "rest"):
        return
    for callback in mcp_raise_callbacks(callbacks):
        if any(callback.should_run_guardrail(data=data, event_type=hook) for hook in _HOOKS):  # pyright: ignore[reportUnknownMemberType]  # legacy hook leaves its request parameter unannotated
            raise UnsupportedMCPBlockPolicy(callback.guardrail_name, effective_surface)


def _is_refusal(error: Exception) -> bool:
    match error:
        case GuardrailRaisedException():
            return error.blocked_content
        case BlockedPiiEntityError() | ModifyResponseException():
            return True
        case HTTPException():
            return error.status_code in (400, 403, 422) and error.__cause__ is None
        case _:
            return False


def mcp_block_error(
    error: Exception, callback: CustomGuardrail, event_hook: GuardrailEventHooks
) -> MCPGuardrailBlock | None:
    if (
        callback.on_mcp_block != "raise"
        or event_hook not in _HOOKS
        or isinstance(error, (MCPGuardrailBlock, UnsupportedMCPBlockPolicy))
        or not _is_refusal(error)
    ):
        return None
    message: Final = str(error.detail) if isinstance(error, HTTPException) else str(error)
    return MCPGuardrailBlock(callback.guardrail_name, event_hook, message)


async def run_mcp_guardrail(
    callback: CustomGuardrail, call: Awaitable[object], event_hook: GuardrailEventHooks
) -> object:
    try:
        result: Final = await call
    except Exception as error:
        refusal: Final = mcp_block_error(error, callback, event_hook)
        if refusal is not None:
            raise refusal from error
        raise
    returned_error: Final = (
        result
        if isinstance(result, Exception)
        else HTTPException(400, detail=result)
        if isinstance(result, str)
        else None
    )
    if returned_error is not None:
        returned_refusal: Final = mcp_block_error(returned_error, callback, event_hook)
        if returned_refusal is not None:
            raise returned_refusal from returned_error
    return result
