from collections.abc import Collection, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Literal

from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException, ModifyResponseException


@dataclass(frozen=True, slots=True)
class MCPGuardrailBlock:
    guardrail_name: str | None
    on_mcp_block: Literal["raise", "tool_error"]


def get_mcp_guardrail_block(exc: BaseException) -> MCPGuardrailBlock | None:
    context: Final[object] = getattr(exc, "_mcp_guardrail_block", None)
    return context if isinstance(context, MCPGuardrailBlock) else None


def guardrail_provider_error(exc: Exception) -> Exception:
    exc.__dict__["_guardrail_provider_error"] = True  # rebind-ok: preserve the provider exception type
    return exc


def _is_mcp_content_block(exc: Exception) -> bool:
    """Recognize typed verdicts and legacy HTTP block statuses, excluding backend failures."""
    if getattr(exc, "_guardrail_provider_error", False) or exc.__cause__ is not None:
        return False
    if isinstance(exc, GuardrailRaisedException):
        return exc.blocked_content
    if isinstance(exc, (BlockedPiiEntityError, ModifyResponseException)):
        return True
    return is_fastapi_http_exception(exc, (400, 403, 422))


def record_mcp_guardrail_block(
    exc: Exception,
    guardrail_name: str | None,
    on_mcp_block: Literal["raise", "tool_error"],
    call_type: str,
    hook_type: str,
) -> None:
    if call_type != "call_mcp_tool" or hook_type not in ("pre_call", "during_call"):
        return
    if get_mcp_guardrail_block(exc) is None and _is_mcp_content_block(exc):
        block: Final = MCPGuardrailBlock(guardrail_name, on_mcp_block)
        exc.__dict__["_mcp_guardrail_block"] = block  # rebind-ok: preserve the provider exception type


@contextmanager
def mcp_guardrail_block_context(
    guardrail_name: str | None,
    on_mcp_block: Literal["raise", "tool_error"],
    call_type: str,
    hook_type: str,
) -> Generator[None, None, None]:
    try:
        yield
    except Exception as exc:
        record_mcp_guardrail_block(exc, guardrail_name, on_mcp_block, call_type, hook_type)
        raise


def first_mcp_raise_block(exceptions: Collection[BaseException]) -> BaseException | None:
    return next(
        (
            exc
            for exc in exceptions
            if (block := get_mcp_guardrail_block(exc)) is not None and block.on_mcp_block == "raise"
        ),
        None,
    )


def raise_mcp_guardrail_block(exc: Exception, *, stream: bool) -> None:
    block: Final = get_mcp_guardrail_block(exc)
    if stream or block is None or block.on_mcp_block != "raise":
        return
    from fastapi import HTTPException

    detail: Final[object] = exc.detail if isinstance(exc, HTTPException) else str(exc)
    raise HTTPException(
        status_code=400,
        detail={"error": detail, "guardrail_name": block.guardrail_name},
    ) from exc


def is_fastapi_http_exception(e: Exception, block_status_codes: Collection[int]) -> bool:
    try:
        from fastapi.exceptions import HTTPException
    except ImportError:
        return False
    return isinstance(e, HTTPException) and e.status_code in block_status_codes


def enrich_http_exception_with_guardrail_context(exc: BaseException, callback: object) -> None:
    try:
        from fastapi.exceptions import HTTPException
    except ImportError:
        return
    if not isinstance(exc, HTTPException):
        return
    detail: Final = getattr(exc, "detail", None)
    if not isinstance(detail, dict):
        return
    guardrail_name: Final[object] = getattr(callback, "guardrail_name", None)
    if guardrail_name:
        detail.setdefault("guardrail_name", guardrail_name)
    event_hook: Final[object] = getattr(callback, "event_hook", None)
    if event_hook:
        detail.setdefault("guardrail_mode", event_hook)


def pre_call_rejection(message: str, guardrail_name: str | None) -> Exception:
    try:
        from fastapi.exceptions import HTTPException
    except ImportError:
        return GuardrailRaisedException(
            guardrail_name=guardrail_name, message=message, should_wrap_with_default_message=False
        )
    return HTTPException(status_code=400, detail={"error": message})
