"""Constants specific to the `/management/v1` control-plane surface.

The contract machinery every list route shares lives in `litellm.proxy.list_api`.
"""

from collections.abc import Mapping, MutableMapping
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm.proxy.list_api.common import PROBLEM_CONTENT_TYPE, PROBLEM_TYPE_BASE
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail

MANAGEMENT_V1_PREFIX: Final = "/management/v1"
PROBLEM_DETAIL_SCHEMA_NAME: Final = "ProblemDetail"
PROBLEM_DETAIL_REF: Final = f"#/components/schemas/{PROBLEM_DETAIL_SCHEMA_NAME}"

_PROBLEM_TITLES: Final[Mapping[int, str]] = MappingProxyType(
    {
        403: "Forbidden",
        404: "Not found",
        409: "Conflict",
        422: "Invalid request body",
        500: "Internal server error",
        503: "Database not connected",
    }
)


def add_problem_detail_component(
    openapi_schema: MutableMapping[str, object],  # mutable-ok: fastapi's generated schema is a plain nested dict
) -> None:
    """Register `ProblemDetail` as an OpenAPI component so `problem_responses()` can `$ref` it.

    FastAPI only emits components for models reachable from a route's `response_model`, and a problem
    document never is one: it is the failure shape, declared out of band.
    """
    components: Final = openapi_schema.setdefault("components", {})  # mutable-ok: seeds a branch of fastapi's own dict
    if not isinstance(components, MutableMapping):
        return
    schemas: Final = components.setdefault("schemas", {})  # mutable-ok: seeds a branch of fastapi's own dict
    if isinstance(schemas, MutableMapping):
        schemas.setdefault(PROBLEM_DETAIL_SCHEMA_NAME, ProblemDetail.model_json_schema())


# FastAPI declares `responses=` as a dict of dicts and rewrites copies of the entries as it renders
# the schema, so every layer below has to be a plain mutable dict.
ProblemResponses: TypeAlias = dict[int | str, dict[str, object]]  # mutable-ok: fastapi's `responses=` contract


def problem_responses(*statuses: int) -> ProblemResponses:
    """OpenAPI `responses=` entries declaring each status as an RFC 9457 problem document.

    Spelled as a raw `$ref` rather than `model=`, because FastAPI renders a `model=` entry under the
    route's own media type and would document these as `application/json`.
    """
    return {status: _problem_entry(status) for status in statuses}  # mutable-ok: fastapi's `responses=` contract


def _problem_entry(status: int) -> dict[str, object]:  # mutable-ok: fastapi's `responses=` contract
    media_type: Final = {"schema": {"$ref": PROBLEM_DETAIL_REF}}  # mutable-ok: fastapi's `responses=` contract
    return {  # mutable-ok: fastapi's `responses=` contract
        "description": _PROBLEM_TITLES.get(status, "Error"),
        "content": {PROBLEM_CONTENT_TYPE: media_type},  # mutable-ok: fastapi's `responses=` contract
    }


def validation_problem(detail: str) -> ProblemDetail:
    """A rejected request body, as opposed to `unknown_query_param_problem` for the query string."""
    return ProblemDetail(
        type=f"{PROBLEM_TYPE_BASE}invalid-request-body",
        title=_PROBLEM_TITLES[422],
        status=422,
        detail=detail,
    )
