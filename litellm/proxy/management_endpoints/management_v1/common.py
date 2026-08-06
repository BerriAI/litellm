"""Contract machinery shared by every `/management/v1` route."""

from typing import Final
from urllib.parse import urlencode

from fastapi import Request
from fastapi.dependencies.utils import get_flat_dependant
from fastapi.responses import JSONResponse

from litellm.types.proxy.management_endpoints.management_v1 import (
    ListLinks,
    PageLinks,
    ProblemDetail,
)

MANAGEMENT_V1_PREFIX: Final = "/management/v1"
PROBLEM_CONTENT_TYPE: Final = "application/problem+json"
# A URN, not an https URL: RFC 9457 only asks that `type` identify the problem
# type, and an https URI promises documentation at that address. Switch to an
# https base only when pages actually exist to serve.
PROBLEM_TYPE_BASE: Final = "urn:litellm:error:"


class ManagementProblem(Exception):
    """Raised to return an RFC 9457 problem instead of the proxy's OpenAI error shape."""

    def __init__(self, problem: ProblemDetail) -> None:
        self.problem = problem
        super().__init__(problem.detail)


def problem_response(problem: ProblemDetail) -> JSONResponse:
    return JSONResponse(
        status_code=problem.status,
        content=problem.model_dump(exclude_none=True),
        media_type=PROBLEM_CONTENT_TYPE,
    )


def _declared_query_params(request: Request) -> frozenset[str]:
    route: Final = request.scope.get("route")
    dependant: Final = getattr(route, "dependant", None)
    if dependant is None:
        return frozenset()
    return frozenset(field.alias for field in get_flat_dependant(dependant, skip_repeats=True).query_params)


def escape_like(value: str) -> str:
    """Escape LIKE/ILIKE metacharacters. Ids routinely contain `_`, which is a wildcard unescaped."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def unknown_query_param_problem(unknown: tuple[str, ...], allowed: tuple[str, ...]) -> ProblemDetail:
    return ProblemDetail(
        type=f"{PROBLEM_TYPE_BASE}unknown-query-parameter",
        title="Unknown query parameter",
        status=400,
        detail=f"Unrecognized query parameter(s): {', '.join(unknown)}.",
        allowed=sorted(allowed),
    )


async def reject_unknown_query_params(request: Request) -> None:
    """Reject any query param the route did not declare.

    A silently ignored filter over-returns data, which is worse than a rejected
    request; a fresh surface is the only chance to be strict about it.
    """
    declared: Final = _declared_query_params(request)
    unknown: Final[tuple[str, ...]] = tuple(sorted(name for name in request.query_params if name not in declared))
    if not unknown:
        return
    raise ManagementProblem(unknown_query_param_problem(unknown=unknown, allowed=tuple(sorted(declared))))


def _page_url(request: Request, page: int) -> str:
    others: Final = tuple((key, value) for key, value in request.query_params.multi_items() if key != "page")
    return f"{request.url.path}?{urlencode((*others, ('page', page)))}"


def build_page_links(request: Request, page: int, has_more: bool) -> PageLinks:
    return PageLinks(
        self_link=_page_url(request, page),
        prev=_page_url(request, page - 1) if page > 1 else None,
        next=_page_url(request, page + 1) if has_more else None,
    )


def build_list_links(request: Request, page: int, total_pages: int) -> ListLinks:
    """Page-mode links. `last` clamps to page 1 on an empty result set so every link still resolves."""
    last: Final = max(total_pages, 1)
    return ListLinks(
        self_link=_page_url(request, page),
        first=_page_url(request, 1),
        prev=_page_url(request, page - 1) if page > 1 else None,
        next=_page_url(request, page + 1) if page < last else None,
        last=_page_url(request, last),
    )
