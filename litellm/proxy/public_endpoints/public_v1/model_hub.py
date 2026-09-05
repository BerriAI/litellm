"""`GET /public/v1/model_hub`."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Final, Literal, Protocol

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
    handle_facet,
    handle_list,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.management_v1 import (
    FacetListResponse,
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


FEATURE_PREFIX: Final = "supports_"


def _features(row: ModelGroupInfoProxy) -> tuple[str, ...]:
    """A row's capabilities as one repeated field, so selecting two of them matches either.

    The hub's feature control has always been a multi-select over the `supports_*` flags.
    One boolean filter per flag would AND them, which is the opposite of what it does.
    """
    return tuple(
        sorted(
            name.removeprefix(FEATURE_PREFIX)
            for name, value in row.model_dump().items()
            if name.startswith(FEATURE_PREFIX) and value is True
        )
    )


def _cells(row: ModelGroupInfoProxy) -> Cells:
    return MappingProxyType(
        {
            "model_group": row.model_group,
            "mode": row.mode,
            "providers": tuple(row.providers),
            "features": _features(row),
            "max_input_tokens": row.max_input_tokens,
            "max_output_tokens": row.max_output_tokens,
            "input_cost_per_token": row.input_cost_per_token,
            "output_cost_per_token": row.output_cost_per_token,
            "rpm": row.rpm,
            "tpm": row.tpm,
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
        "providers": FilterSpec(type=str, ops=frozenset(("contains", "in"))),
        "features": FilterSpec(type=str, ops=frozenset(("in",))),
    }
)

MODEL_HUB_FACETS: Final[Mapping[str, str]] = MappingProxyType(
    {"providers": "providers", "modes": "mode", "features": "features"}
)

MODEL_HUB_LIST_SPEC: Final[ListSpec[ModelGroupInfoProxy, ModelGroupInfoProxy]] = ListSpec(
    resource="model groups",
    sortable=frozenset(
        (
            "model_group",
            "mode",
            "providers",
            "max_input_tokens",
            "max_output_tokens",
            "input_cost_per_token",
            "output_cost_per_token",
            "rpm",
            "tpm",
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


def _published_rows() -> Sequence[ModelGroupInfoProxy]:
    from litellm.proxy.proxy_server import (
        _get_model_group_info,  # pyright: ignore[reportPrivateUsage]  # /public/model_hub imports it the same way
        llm_router,
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
    if litellm.public_model_groups is None:
        return ()
    return tuple(
        _get_model_group_info(
            llm_router=llm_router,
            all_models_str=litellm.public_model_groups,
            model_group=None,
        )
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
    tags=["public", "model management"],  # mutable-ok: fastapi types tags as list[str | Enum]
    dependencies=(Depends(user_api_key_auth),),
    response_model=ListResponse[ModelGroupInfoProxy],
)
async def public_model_hub_list(
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ListResponse[ModelGroupInfoProxy]:
    """
    The public model groups this proxy publishes, paged, sortable, searchable and
    filterable, for the public Model Hub page. No authentication.

    A rejected request answers with the parameters, sort fields and filter operators
    it would have accepted, so the accepted set stays discoverable from the endpoint
    itself rather than from a copy of the spec kept here.

    Example curl:
    ```
    curl --location --globoff \
        'http://0.0.0.0:4000/public/v1/model_hub?sort=-input_cost_per_token&filter[mode][in]=chat&page_size=25'
    ```
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        return await handle_list(
            spec=MODEL_HUB_LIST_SPEC,
            executor=_executor(_published_rows(), prisma_client),
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


@router.get(
    "/model_hub/{facet}",
    tags=["public", "model management"],  # mutable-ok: fastapi types tags as list[str | Enum]
    dependencies=(Depends(user_api_key_auth),),
    response_model=FacetListResponse,
)
async def public_model_hub_facet(
    request: Request,
    facet: Literal["providers", "modes", "features"],
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> FacetListResponse:
    """
    The distinct providers, modes or features across the published model groups, for the
    Model Hub's filter dropdowns. No authentication.

    Carries the same filters and search as the list route, so a dropdown offers exactly
    the values the table can show: asking for providers under `filter[mode][in]=chat`
    lists only the providers that serve a chat model.

    Example curl:
    ```
    curl --location --globoff \
        'http://0.0.0.0:4000/public/v1/model_hub/providers?filter[mode][in]=chat&page_size=50'
    ```
    """
    try:
        return await handle_facet(
            spec=MODEL_HUB_LIST_SPEC,
            executor=InMemoryListExecutor(rows=_published_rows(), cells=_cells),
            request=request,
            caller=user_api_key_dict,
            field=MODEL_HUB_FACETS[facet],
        )

    except ManagementProblem:
        raise
    except Exception as e:  # noqa: BLE001  # a router error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.public_endpoints.public_v1.model_hub.public_model_hub_facet(): Exception occured - %s", e
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to list public model group values.",
            )
        )
