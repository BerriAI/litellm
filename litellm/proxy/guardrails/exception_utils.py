from collections.abc import Collection
from typing import Final

from litellm.exceptions import GuardrailRaisedException


def is_fastapi_http_exception(e: Exception, block_status_codes: Collection[int]) -> bool:
    try:
        from fastapi.exceptions import HTTPException
    except ImportError:
        return False
    return isinstance(e, HTTPException) and e.status_code in block_status_codes


def enrich_http_exception_with_guardrail_context(exc: BaseException, callback: object) -> None:
    if not hasattr(exc, "mcp_block_behavior"):
        setattr(  # noqa: B010  # exception types do not declare per-deployment MCP context
            exc, "mcp_block_behavior", getattr(callback, "mcp_block_behavior", None)
        )
    try:
        from fastapi.exceptions import HTTPException
    except ImportError:
        return
    if not isinstance(exc, HTTPException):
        return
    guardrail_name: Final[object] = getattr(callback, "guardrail_name", None)
    detail: Final = getattr(exc, "detail", None)
    if not isinstance(detail, dict):
        if isinstance(guardrail_name, str):
            setattr(exc, "guardrail_name", guardrail_name)  # noqa: B010  # HTTPException does not declare guardrail_name
        return
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
