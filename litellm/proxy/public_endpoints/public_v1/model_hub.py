"""`GET /public/v1/model_hub`."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Final, Protocol

from fastapi import APIRouter, Depends, Request
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem
from litellm.proxy.list_api.in_memory import Cells, InMemoryListExecutor
from litellm.proxy.list_api.list_framework import (
    FilterSpec,
    ListSpec,
    Scope,
    ScopeAll,
    SortKey,
    handle_list,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.management_v1 import (
    ListResponse,
    ProblemDetail,
)
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    ModelGroupInfoProxy,
)

router: Final = APIRouter()


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """The health fields a model hub row carries, as the latest health check recorded them."""

    status: str | None
    response_time_ms: float | None
    checked_at: str | None


class HealthSnapshotLookup(Protocol):
    """The health half of the list, injected so the page slice decides how much of it runs."""

    async def latest_for(self, model_groups: Sequence[str]) -> Mapping[str, HealthSnapshot]: ...


@dataclass(frozen=True, slots=True)
class PrismaHealthSnapshotLookup:
    prisma_client: PrismaClient

    async def latest_for(self, model_groups: Sequence[str]) -> Mapping[str, HealthSnapshot]:
        checks: Final = await self.prisma_client.get_latest_health_checks_for_models(model_groups)
        return MappingProxyType(
            {
                check.model_name: HealthSnapshot(
                    status=check.status,
                    response_time_ms=check.response_time_ms,
                    checked_at=check.checked_at.isoformat() if check.checked_at else None,
                )
                for check in checks
            }
        )


class _HealthFields(TypedDict):
    health_status: ReadOnly[str | None]
    health_response_time: ReadOnly[float | None]
    health_checked_at: ReadOnly[str | None]


def _with_health(row: ModelGroupInfoProxy, health: HealthSnapshot | None) -> ModelGroupInfoProxy:
    if health is None:
        return row
    update: Final[_HealthFields] = {
        "health_status": health.status,
        "health_response_time": health.response_time_ms,
        "health_checked_at": health.checked_at,
    }
    return row.model_copy(update=update)


@dataclass(frozen=True, slots=True)
class HealthEnricher:
    """Resolves health for exactly the rows handed to it, which is the page and never the match set."""

    lookup: HealthSnapshotLookup

    async def __call__(self, rows: Sequence[ModelGroupInfoProxy]) -> Sequence[ModelGroupInfoProxy]:
        health: Final = await self.lookup.latest_for(tuple(row.model_group for row in rows))
        return tuple(_with_health(row, health.get(row.model_group)) for row in rows)


def _cells(row: ModelGroupInfoProxy) -> Cells:
    return MappingProxyType(
        {
            "model_group": row.model_group,
            "mode": row.mode,
            "providers": tuple(row.providers),
            "max_input_tokens": row.max_input_tokens,
            "max_output_tokens": row.max_output_tokens,
            "input_cost_per_token": row.input_cost_per_token,
            "output_cost_per_token": row.output_cost_per_token,
        }
    )


def _serialize(row: ModelGroupInfoProxy) -> ModelGroupInfoProxy:
    """The row shape is the wire shape: the rows served are the router's own model group records."""
    return row


def _scope(_caller: UserAPIKeyAuth) -> Scope:
    """Unconditional, and `/public/v1` is the one surface where that is allowed.

    Every row here is already a model group the operator published, so a public browse
    caller seeing all of them is the answer, not a gap in the scoping.
    """
    return ScopeAll()


MODEL_HUB_FILTERS: Final[Mapping[str, FilterSpec]] = MappingProxyType(
    {
        "mode": FilterSpec(type=str, ops=frozenset(("eq", "in"))),
        "providers": FilterSpec(type=str, ops=frozenset(("contains",))),
    }
)

MODEL_HUB_LIST_SPEC: Final[ListSpec[ModelGroupInfoProxy, ModelGroupInfoProxy]] = ListSpec(
    resource="model groups",
    sortable=frozenset(
        (
            "model_group",
            "mode",
            "max_input_tokens",
            "max_output_tokens",
            "input_cost_per_token",
            "output_cost_per_token",
        )
    ),
    searchable=frozenset(("model_group",)),
    filters=MODEL_HUB_FILTERS,
    default_sort=(SortKey(field="model_group", descending=False),),
    default_page_size=50,
    max_page_size=100,
    scope=_scope,
    serialize=_serialize,
    tiebreaker="model_group",
)


def _executor(
    rows: Sequence[ModelGroupInfoProxy],
    prisma_client: PrismaClient | None,
) -> InMemoryListExecutor[ModelGroupInfoProxy]:
    if prisma_client is None:
        return InMemoryListExecutor(rows=rows, cells=_cells)
    return InMemoryListExecutor(
        rows=rows,
        cells=_cells,
        enrich_page=HealthEnricher(lookup=PrismaHealthSnapshotLookup(prisma_client=prisma_client)),
    )


@router.get(
    "/model_hub",
    tags=("public", "model management"),
    dependencies=(Depends(user_api_key_auth),),
    response_model=ListResponse[ModelGroupInfoProxy],
)
async def public_model_hub_list(
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ListResponse[ModelGroupInfoProxy]:
    """
    The public model groups this proxy publishes, paged, sortable, searchable and
    filterable, for the public Model Hub page.

    No authentication. The older `/public/model_hub` answers with every public model
    group as a bare array, which is a multi-megabyte response and an unusable page
    once a proxy publishes a few thousand of them.

    `sort` takes a comma-separated list of `model_group`, `mode`, `max_input_tokens`,
    `max_output_tokens`, `input_cost_per_token` or `output_cost_per_token`, each
    optionally prefixed with `-` for descending, and defaults to `model_group`
    ascending. `model_group` is appended to every sort as the tiebreaker. `q` is a
    case-insensitive substring match on `model_group`. `page_size` defaults to 50 and
    is capped at 100. Filters are `filter[mode]`, `filter[mode][in]` and
    `filter[providers][contains]`.

    Example curl:
    ```
    curl --location --globoff \
        'http://0.0.0.0:4000/public/v1/model_hub?sort=-input_cost_per_token&filter[mode][in]=chat&page_size=25'
    ```
    """
    try:
        from litellm.proxy.proxy_server import (
            _get_model_group_info,
            llm_router,
            prisma_client,
        )

        if llm_router is None:
            raise ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}no-llm-router",
                    title="No models configured",
                    status=400,
                    detail=CommonProxyErrors.no_llm_router.value,
                )
            )

        rows: Final[Sequence[ModelGroupInfoProxy]] = (
            ()
            if litellm.public_model_groups is None
            else tuple(
                _get_model_group_info(
                    llm_router=llm_router,
                    all_models_str=litellm.public_model_groups,
                    model_group=None,
                )
            )
        )

        return await handle_list(
            spec=MODEL_HUB_LIST_SPEC,
            executor=_executor(rows, prisma_client),
            request=request,
            caller=user_api_key_dict,
        )

    except ManagementProblem:
        raise
    except Exception as e:  # noqa: BLE001  # a router error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.public_endpoints.public_v1.model_hub.public_model_hub_list(): Exception occured - %s", e
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to list public model groups.",
            )
        )
