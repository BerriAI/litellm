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
