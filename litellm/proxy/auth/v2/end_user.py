from typing import Awaitable, Callable, Optional

from starlette.requests import Request

from .context import attach_end_user

# Extraction reuses the existing request-body/header logic; validation (the
# customer-table lookup) is injected so this stage is testable without a DB and
# so callers choose whether to validate.
Extractor = Callable[[dict, Optional[dict]], Optional[str]]
Validator = Callable[[str], Awaitable[Optional[str]]]


def _default_extractor(
    request_data: dict, request_headers: Optional[dict]
) -> Optional[str]:
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    return get_end_user_id_from_request_body(request_data, request_headers)


async def resolve_end_user(
    request: Request,
    request_data: Optional[dict],
    request_headers: Optional[dict] = None,
    *,
    extractor: Optional[Extractor] = None,
    validator: Optional[Validator] = None,
) -> Optional[str]:
    """Resolve the request's end-user (customer) and record it on the auth context.

    A dedicated stage rather than auth-gate work: it reads the already-published
    context via :func:`attach_end_user`. Returns the resolved id, or None when the
    request carries no end-user.
    """
    extract = extractor or _default_extractor
    raw = extract(request_data or {}, request_headers)
    if raw is None:
        return None
    resolved = await validator(raw) if validator is not None else raw
    attach_end_user(request, resolved)
    return resolved
