import asyncio
from collections.abc import Mapping, Sequence
from types import MappingProxyType, TracebackType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    Protocol,
    cast,  # noqa: TID251  # Prisma's generated TransactionManager doesn't expose typed table accessors; we narrow to our protocol
)

from fastapi import APIRouter, Depends, status
from prisma import models
from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import (
    CommonProxyErrors,
    LitellmTableNames,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    ReconcileOutcome,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.model_management_endpoints import (
    clear_cache,
    raise_if_reload_degraded_serving,
)
from litellm.proxy.management_helpers.audit_logs import create_object_audit_log
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import VirtualModelRepository
from litellm.router_utils.auto_router_model_naming import AUTO_ROUTER_MODEL_PREFIX
from litellm.types.proxy.management_endpoints.virtual_model_endpoints import (
    VirtualModelCreateRequest,
    VirtualModelMemberRequest,
    VirtualModelMemberResponse,
    VirtualModelPatchRequest,
    VirtualModelResponse,
)

if TYPE_CHECKING:
    from prisma.models import (
        LiteLLM_VirtualModelTable as PrismaVirtualModelTable,
    )

router: Final = APIRouter()


class _VirtualModelTableClient(Protocol):
    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> "PrismaVirtualModelTable | None": ...

    async def update(
        self,
        where: Mapping[str, object],
        data: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> "PrismaVirtualModelTable | None": ...


class _VirtualModelMemberTableClient(Protocol):
    async def create_many(self, data: Sequence[Mapping[str, object]]) -> int: ...


class _TransactionTables(Protocol):
    async def query_raw(self, query: str, *args: object) -> object: ...

    @property
    def litellm_virtualmodeltable(self) -> _VirtualModelTableClient: ...

    @property
    def litellm_virtualmodelmembertable(self) -> _VirtualModelMemberTableClient: ...


class _TransactionManager(Protocol):
    async def __aenter__(self) -> _TransactionTables: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


_PROXY_MODEL_ROWS: Final = TypeAdapter(tuple[models.LiteLLM_ProxyModelTable, ...])
_VIRTUAL_MODEL_ROWS: Final = TypeAdapter(tuple[models.LiteLLM_VirtualModelTable, ...])
_VIRTUAL_MODEL_ROW: Final = TypeAdapter(models.LiteLLM_VirtualModelTable)


def _validate_proxy_models(raw: object) -> tuple[models.LiteLLM_ProxyModelTable, ...]:
    return _PROXY_MODEL_ROWS.validate_python(raw)


def _validate_virtual_models(raw: object) -> tuple[models.LiteLLM_VirtualModelTable, ...]:
    return _VIRTUAL_MODEL_ROWS.validate_python(raw)


def _validate_virtual_model(raw: object) -> models.LiteLLM_VirtualModelTable:
    if raw is None:
        raise ValueError("Expected non-None virtual model")
    return _VIRTUAL_MODEL_ROW.validate_python(raw)


async def _resolve_live_deployments_by_id(
    prisma_client: PrismaClient,
    all_member_ids: frozenset[str],
) -> Mapping[str, str]:
    if not all_member_ids:
        return MappingProxyType({})

    deployments_raw: Final = await prisma_client.writer_db.litellm_proxymodeltable.find_many(  # pyright: ignore[reportAny]  # Prisma accessors return Any
        where={"model_id": {"in": list(all_member_ids)}}
    )
    deployments: Final = _validate_proxy_models(deployments_raw)  # pyright: ignore[reportAny]  # deployments_raw is Any from Prisma
    return MappingProxyType({d.model_id: d.model_name for d in deployments if d.model_name})


def _require_admin_family(user: UserAPIKeyAuth) -> None:
    if user.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        raise ProxyException(
            message="Only proxy admin roles can view virtual models",
            type=ProxyErrorTypes.auth_error.value,
            code=status.HTTP_403_FORBIDDEN,
            param=None,
        )


def _require_admin(user: UserAPIKeyAuth) -> None:
    if user.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise ProxyException(
            message="Only proxy admins can manage virtual models",
            type=ProxyErrorTypes.auth_error.value,
            code=status.HTTP_403_FORBIDDEN,
            param=None,
        )


async def _validate_members(
    prisma_client: PrismaClient,
    members: tuple[VirtualModelMemberRequest, ...],
) -> tuple[BaseModel, ...]:
    model_ids: Final = tuple(m.model_id for m in members)

    if len(model_ids) != len(frozenset(model_ids)):
        raise ProxyException(
            message="Duplicate model_ids in members list",
            type=ProxyErrorTypes.validation_error.value,
            code=status.HTTP_400_BAD_REQUEST,
            param="members",
        )

    deployments_raw: Final = await prisma_client.writer_db.litellm_proxymodeltable.find_many(  # pyright: ignore[reportAny]  # Prisma accessors return Any
        where={"model_id": {"in": list(model_ids)}}
    )
    deployments: Final = _validate_proxy_models(deployments_raw)  # pyright: ignore[reportAny]  # deployments_raw is Any from Prisma

    found_ids: Final = frozenset(d.model_id for d in deployments)
    missing_ids: Final = frozenset(model_ids) - found_ids
    if missing_ids:
        raise ProxyException(
            message=f"Models not found in LiteLLM_ProxyModelTable. "
            f"Only DB-managed deployments are eligible members. Missing ids: {sorted(missing_ids)}",
            type=ProxyErrorTypes.validation_error.value,
            code=status.HTTP_400_BAD_REQUEST,
            param="members",
        )

    for deployment in deployments:
        model_info: Mapping[str, object] = deployment.model_info if isinstance(deployment.model_info, dict) else {}
        if model_info.get("team_id"):  # pyright: ignore[reportUnknownMemberType]  # Mapping[str, object].get returns Unknown
            raise ProxyException(
                message=f"Model {deployment.model_id} is team-scoped (team_id={model_info['team_id']}). "
                f"Team-scoped deployments cannot be members of virtual models.",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param="members",
            )

        litellm_params: Mapping[str, object] = (
            deployment.litellm_params if isinstance(deployment.litellm_params, dict) else {}
        )
        model_param = litellm_params.get("model", "")  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]  # Mapping[str, object].get returns Unknown
        if isinstance(model_param, str) and model_param.startswith(AUTO_ROUTER_MODEL_PREFIX):
            raise ProxyException(
                message=f"Model {deployment.model_id} is an auto-router marker and "
                f"cannot be a member of a virtual model.",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param="members",
            )

    return tuple(deployments)


async def _reconcile_after_write() -> ReconcileOutcome:
    try:
        return await clear_cache()
    except Exception as e:  # noqa: BLE001  # reconcile failure must not fail the write; the degraded outcome is returned to the caller
        verbose_proxy_logger.warning("Failed to reconcile after virtual model write: %s", e)
        return ReconcileOutcome(still_desired=None, live_after=None)


async def _patch_virtual_model_in_transaction(
    tx: _TransactionTables,
    virtual_model_id: str,
    request_body: VirtualModelPatchRequest,
    update_data: Mapping[str, object],
) -> tuple[models.LiteLLM_VirtualModelTable, models.LiteLLM_VirtualModelTable]:
    from prisma.errors import UniqueViolationError

    existing_raw: Final = await tx.litellm_virtualmodeltable.find_unique(
        where={"virtual_model_id": virtual_model_id},
        include={"members": True},
    )

    if existing_raw is None:
        raise ProxyException(
            message=f"Virtual model {virtual_model_id} not found",
            type=ProxyErrorTypes.not_found_error.value,
            code=status.HTTP_404_NOT_FOUND,
            param="virtual_model_id",
        )

    existing: Final = _validate_virtual_model(existing_raw)
    existing_member_ids: Final = frozenset(m.model_id for m in existing.members or [])

    if request_body.add_members:
        for member in request_body.add_members:
            if member.model_id in existing_member_ids:
                raise ProxyException(
                    message=f"Model {member.model_id} is already a member of this virtual model",
                    type=ProxyErrorTypes.validation_error.value,
                    code=status.HTTP_400_BAD_REQUEST,
                    param="members",
                )

        if len(existing_member_ids) + len(request_body.add_members) > 50:
            raise ProxyException(
                message=f"Virtual model cannot have more than 50 members. "
                f"Current: {len(existing_member_ids)}, adding: {len(request_body.add_members)}",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param="add_members",
            )

        try:
            await tx.litellm_virtualmodelmembertable.create_many(
                data=[
                    {
                        "virtual_model_id": virtual_model_id,
                        "model_id": member.model_id,
                        "priority": member.priority,
                    }
                    for member in request_body.add_members
                ]
            )
        except UniqueViolationError:
            raise ProxyException(
                message="One or more models are already members of this virtual model",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param="members",
            )

    try:
        updated_raw: Final = await tx.litellm_virtualmodeltable.update(
            where={"virtual_model_id": virtual_model_id},
            data=update_data,
            include={"members": {"order_by": {"priority": "asc"}}},
        )
        if updated_raw is None:
            raise ProxyException(
                message=f"Virtual model {virtual_model_id} was deleted during update",
                type=ProxyErrorTypes.not_found_error.value,
                code=status.HTTP_404_NOT_FOUND,
                param="virtual_model_id",
            )
        updated: Final = _validate_virtual_model(updated_raw)
    except UniqueViolationError:
        name_value: Final = request_body.virtual_model_name
        message: Final = (
            f"Virtual model with name '{name_value}' already exists"
            if name_value is not None
            else "Virtual model with this name already exists"
        )
        raise ProxyException(
            message=message,
            type=ProxyErrorTypes.validation_error.value,
            code=status.HTTP_409_CONFLICT,
            param="virtual_model_name",
        )

    return existing, updated


def _build_virtual_model_response(
    row: models.LiteLLM_VirtualModelTable,
    deployments_by_id: Mapping[str, str],
) -> VirtualModelResponse:
    member_responses: Final = tuple(
        VirtualModelMemberResponse(
            model_id=member.model_id,
            model_name=deployments_by_id.get(member.model_id),
            priority=member.priority,
        )
        for member in row.members or []
    )

    live_count: Final = sum(1 for member in row.members or [] if deployments_by_id.get(member.model_id) is not None)

    return VirtualModelResponse(
        virtual_model_id=row.virtual_model_id,
        virtual_model_name=row.virtual_model_name,
        description=row.description,
        members=tuple(member_responses),
        routing_strategy=row.routing_strategy,
        num_retries=row.num_retries,
        timeout=row.timeout,
        stream_timeout=row.stream_timeout,
        allowed_fails=row.allowed_fails,
        cooldown_time=row.cooldown_time,
        live_member_count=live_count,
        created_at=str(row.created_at) if row.created_at else None,
        created_by=row.created_by,
        updated_at=str(row.updated_at) if row.updated_at else None,
        updated_by=row.updated_by,
    )


@router.post(
    "/virtual_model",
    tags=["virtual model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_virtual_model(
    request_body: VirtualModelCreateRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> VirtualModelResponse:
    """Create a virtual model with ordered failover across existing deployments.

    Virtual models expose a single public name that routes to member deployments
    in priority order, with automatic failover on errors.
    """
    from litellm.proxy import proxy_server

    _require_admin(user_api_key_dict)

    prisma_client: Final = proxy_server.prisma_client
    if prisma_client is None:
        raise ProxyException(
            message=CommonProxyErrors.db_not_connected_error.value,
            type=ProxyErrorTypes.no_db_connection.value,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        )

    deployments: Final = await _validate_members(prisma_client, request_body.members)

    user_id: Final = user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME

    live_before_reload: Final = proxy_server.live_model_ids_snapshot() if proxy_server.llm_router else frozenset()  # pyright: ignore[reportUnknownVariableType]  # live_model_ids_snapshot returns frozenset[str] | Unknown

    try:
        created_row: Final = await VirtualModelRepository(prisma_client, use_writer=True).table.create(
            data={
                "virtual_model_name": request_body.virtual_model_name,
                "description": request_body.description,
                "routing_strategy": request_body.routing_strategy,
                "num_retries": request_body.num_retries,
                "timeout": request_body.timeout,
                "stream_timeout": request_body.stream_timeout,
                "allowed_fails": request_body.allowed_fails,
                "cooldown_time": request_body.cooldown_time,
                "created_by": user_id,
                "updated_by": user_id,
                "members": {
                    "create": [
                        {
                            "model_id": member.model_id,
                            "priority": member.priority,
                        }
                        for member in request_body.members
                    ]
                },
            }
        )
    except Exception as e:
        from prisma.errors import UniqueViolationError

        if isinstance(e, UniqueViolationError):
            raise ProxyException(
                message=f"Virtual model with name '{request_body.virtual_model_name}' already exists",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_409_CONFLICT,
                param="virtual_model_name",
            )
        raise

    vm_id: Final = created_row.virtual_model_id

    reconcile_outcome: Final = await _reconcile_after_write()

    asyncio.create_task(
        create_object_audit_log(
            object_id=vm_id,
            action="created",
            user_api_key_dict=user_api_key_dict,
            table_name=LitellmTableNames.VIRTUAL_MODEL_TABLE_NAME,
            before_value=None,
            after_value=created_row.model_dump_json(exclude_none=True),
            litellm_changed_by=user_id,
            litellm_proxy_admin_name=LITELLM_PROXY_ADMIN_NAME,
        )
    )

    raise_if_reload_degraded_serving(
        before=live_before_reload,  # pyright: ignore[reportUnknownArgumentType]  # live_before_reload is unknown from live_model_ids_snapshot
        written_models=[],
        action="create",
        still_desired=reconcile_outcome.still_desired,
        live_after=reconcile_outcome.live_after,
    )

    deployments_by_id: Final = MappingProxyType({d.model_id: d.model_name for d in deployments})  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType,reportUnknownMemberType,reportAttributeAccessIssue]  # deployments is tuple[BaseModel, ...] from Prisma cast; model_id/model_name access on BaseModel
    member_responses: Final = tuple(
        VirtualModelMemberResponse(
            model_id=member.model_id,
            model_name=deployments_by_id.get(member.model_id),  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]  # deployments_by_id is MappingProxyType from Prisma cast
            priority=member.priority,
        )
        for member in request_body.members
    )

    return VirtualModelResponse(
        virtual_model_id=vm_id,
        virtual_model_name=request_body.virtual_model_name,
        description=request_body.description,
        members=member_responses,
        routing_strategy=request_body.routing_strategy,
        num_retries=request_body.num_retries,
        timeout=request_body.timeout,
        stream_timeout=request_body.stream_timeout,
        allowed_fails=request_body.allowed_fails,
        cooldown_time=request_body.cooldown_time,
        live_member_count=len(member_responses),
        created_at=str(created_row.created_at) if created_row.created_at else None,
        created_by=user_id,
        updated_at=str(created_row.updated_at) if created_row.updated_at else None,
        updated_by=user_id,
    )


@router.get(
    "/virtual_model",
    tags=["virtual model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_virtual_models(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> tuple[VirtualModelResponse, ...]:
    from litellm.proxy import proxy_server

    _require_admin_family(user_api_key_dict)

    prisma_client: Final = proxy_server.prisma_client
    if prisma_client is None:
        raise ProxyException(
            message=CommonProxyErrors.db_not_connected_error.value,
            type=ProxyErrorTypes.no_db_connection.value,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        )

    rows_raw: Final = await VirtualModelRepository(prisma_client).table.find_many(
        include={"members": {"order_by": {"priority": "asc"}}}
    )
    rows: Final = _validate_virtual_models(rows_raw)

    all_member_ids: Final = frozenset(
        member.model_id
        for row in rows
        for member in row.members or []  # comprehension-ok: flattening members across rows
    )

    live_deployments_by_id: Final = await _resolve_live_deployments_by_id(prisma_client, all_member_ids)

    return tuple(_build_virtual_model_response(row, live_deployments_by_id) for row in rows)


@router.get(
    "/virtual_model/{virtual_model_id}",
    tags=["virtual model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_virtual_model(
    virtual_model_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> VirtualModelResponse:
    from litellm.proxy import proxy_server

    _require_admin_family(user_api_key_dict)

    prisma_client: Final = proxy_server.prisma_client
    if prisma_client is None:
        raise ProxyException(
            message=CommonProxyErrors.db_not_connected_error.value,
            type=ProxyErrorTypes.no_db_connection.value,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        )

    row_raw: Final = await VirtualModelRepository(prisma_client).table.find_unique(
        where={"virtual_model_id": virtual_model_id},
        include={"members": {"order_by": {"priority": "asc"}}},
    )

    if row_raw is None:
        raise ProxyException(
            message=f"Virtual model {virtual_model_id} not found",
            type=ProxyErrorTypes.not_found_error.value,
            code=status.HTTP_404_NOT_FOUND,
            param="virtual_model_id",
        )

    row: Final = _validate_virtual_model(row_raw)

    live_deployments_by_id: Final = await _resolve_live_deployments_by_id(
        prisma_client, frozenset(member.model_id for member in row.members or [])
    )
    return _build_virtual_model_response(row, live_deployments_by_id)


@router.patch(
    "/virtual_model/{virtual_model_id}",
    tags=["virtual model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def patch_virtual_model(
    virtual_model_id: str,
    request_body: VirtualModelPatchRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> VirtualModelResponse:
    from litellm.proxy import proxy_server

    _require_admin(user_api_key_dict)

    prisma_client: Final = proxy_server.prisma_client
    if prisma_client is None:
        raise ProxyException(
            message=CommonProxyErrors.db_not_connected_error.value,
            type=ProxyErrorTypes.no_db_connection.value,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        )

    if request_body.add_members:
        await _validate_members(prisma_client, request_body.add_members)

    user_id: Final = user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME
    update_data: Final = {
        "updated_by": user_id,
        **(
            {"virtual_model_name": request_body.virtual_model_name}
            if request_body.virtual_model_name is not None
            else {}
        ),
        **({"description": request_body.description} if request_body.description is not None else {}),
        **({"routing_strategy": request_body.routing_strategy} if request_body.routing_strategy is not None else {}),
        **({"num_retries": request_body.num_retries} if request_body.num_retries is not None else {}),
        **({"timeout": request_body.timeout} if request_body.timeout is not None else {}),
        **({"stream_timeout": request_body.stream_timeout} if request_body.stream_timeout is not None else {}),
        **({"allowed_fails": request_body.allowed_fails} if request_body.allowed_fails is not None else {}),
        **({"cooldown_time": request_body.cooldown_time} if request_body.cooldown_time is not None else {}),
    }

    tx_manager: Final[_TransactionManager] = cast(  # cast-ok: generated Prisma tx lacks typed table accessors
        _TransactionManager, prisma_client.tx()
    )
    async with tx_manager as tx:
        if request_body.add_members:
            await tx.query_raw(
                'SELECT virtual_model_id FROM "LiteLLM_VirtualModelTable" WHERE virtual_model_id = $1 FOR UPDATE',
                virtual_model_id,
            )

        existing, updated = await _patch_virtual_model_in_transaction(tx, virtual_model_id, request_body, update_data)

    from litellm.proxy.common_utils.config_sync_pubsub import publish_config_change_for_object_type

    await publish_config_change_for_object_type("litellm_virtualmodeltable")
    if request_body.add_members:
        await publish_config_change_for_object_type("litellm_virtualmodelmembertable")

    reconcile_outcome: Final = await _reconcile_after_write()

    asyncio.create_task(
        create_object_audit_log(
            object_id=virtual_model_id,
            action="updated",
            user_api_key_dict=user_api_key_dict,
            table_name=LitellmTableNames.VIRTUAL_MODEL_TABLE_NAME,
            before_value=existing.model_dump_json(exclude_none=True),
            after_value=updated.model_dump_json(exclude_none=True),
            litellm_changed_by=user_id,
            litellm_proxy_admin_name=LITELLM_PROXY_ADMIN_NAME,
        )
    )

    raise_if_reload_degraded_serving(
        before=frozenset(),
        written_models=[],
        action="update",
        still_desired=reconcile_outcome.still_desired,
        live_after=reconcile_outcome.live_after,
    )

    all_member_ids: Final = frozenset(member.model_id for member in updated.members or [])
    deployments_by_id: Final = await _resolve_live_deployments_by_id(prisma_client, all_member_ids)

    return _build_virtual_model_response(updated, deployments_by_id)


@router.delete(
    "/virtual_model/{virtual_model_id}",
    tags=["virtual model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_virtual_model(
    virtual_model_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> VirtualModelResponse:
    from litellm.proxy import proxy_server

    _require_admin(user_api_key_dict)

    prisma_client: Final = proxy_server.prisma_client
    if prisma_client is None:
        raise ProxyException(
            message=CommonProxyErrors.db_not_connected_error.value,
            type=ProxyErrorTypes.no_db_connection.value,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        )

    existing_raw: Final = await VirtualModelRepository(prisma_client).table.find_unique(
        where={"virtual_model_id": virtual_model_id},
        include={"members": True},
    )

    if existing_raw is None:
        raise ProxyException(
            message=f"Virtual model {virtual_model_id} not found",
            type=ProxyErrorTypes.not_found_error.value,
            code=status.HTTP_404_NOT_FOUND,
            param="virtual_model_id",
        )

    existing: Final = _validate_virtual_model(existing_raw)

    deleted_raw: Final = await VirtualModelRepository(prisma_client, use_writer=True).table.delete(
        where={"virtual_model_id": virtual_model_id},
    )

    if deleted_raw is None:
        raise ProxyException(
            message=f"Virtual model {virtual_model_id} was already deleted",
            type=ProxyErrorTypes.not_found_error.value,
            code=status.HTTP_404_NOT_FOUND,
            param="virtual_model_id",
        )

    deleted: Final = _validate_virtual_model(deleted_raw)

    reconcile_outcome: Final = await _reconcile_after_write()

    user_id: Final = user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME
    asyncio.create_task(
        create_object_audit_log(
            object_id=virtual_model_id,
            action="deleted",
            user_api_key_dict=user_api_key_dict,
            table_name=LitellmTableNames.VIRTUAL_MODEL_TABLE_NAME,
            before_value=existing.model_dump_json(exclude_none=True),
            after_value=None,
            litellm_changed_by=user_id,
            litellm_proxy_admin_name=LITELLM_PROXY_ADMIN_NAME,
        )
    )

    raise_if_reload_degraded_serving(
        before=frozenset(),
        written_models=[],
        action="delete",
        still_desired=reconcile_outcome.still_desired,
        live_after=reconcile_outcome.live_after,
    )

    member_responses: Final = tuple(
        VirtualModelMemberResponse(
            model_id=m.model_id,
            model_name=None,
            priority=m.priority,
        )
        for m in existing.members or []
    )

    return VirtualModelResponse(
        virtual_model_id=deleted.virtual_model_id,
        virtual_model_name=deleted.virtual_model_name,
        description=deleted.description,
        members=member_responses,
        routing_strategy=deleted.routing_strategy,
        num_retries=deleted.num_retries,
        timeout=deleted.timeout,
        stream_timeout=deleted.stream_timeout,
        allowed_fails=deleted.allowed_fails,
        cooldown_time=deleted.cooldown_time,
        live_member_count=0,
        created_by=deleted.created_by,
        updated_by=deleted.updated_by,
    )
