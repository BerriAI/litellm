"""CRUD endpoints for custom RBAC roles. Proxy admin only."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Final, Protocol

from fastapi import APIRouter, Depends, HTTPException, status

from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.custom_rbac import (
    get_config_custom_rbac_roles,
    invalidate_custom_rbac_engine_cache,
    is_reserved_role_name,
    validate_role_permissions,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import get_prisma_client_or_throw
from litellm.repositories.prisma_args import prisma_args, prisma_str_list
from litellm.repositories.table_repositories import CustomRBACRoleRepository
from litellm.types.custom_rbac import (
    CustomRBACRoleCreateRequest,
    CustomRBACRoleDeleteRequest,
    CustomRBACRoleDeleteResponse,
    CustomRBACRoleListResponse,
    CustomRBACRoleResponse,
    CustomRBACRoleUpdateRequest,
)

router: Final = APIRouter(tags=["custom rbac role management"])  # mutable-ok: APIRouter concatenates its tags list

_ORDER_BY_ROLE_NAME: Final[Mapping[str, object]] = MappingProxyType({"role_name": "asc"})


class _RoleRecord(Protocol):
    def dict(self) -> Mapping[str, object]: ...


class _CustomRoleTable(Protocol):
    async def find_unique(self, where: Mapping[str, object]) -> _RoleRecord | None: ...

    async def find_many(self, order: Mapping[str, object]) -> Sequence[_RoleRecord]: ...

    async def create(self, data: Mapping[str, object]) -> _RoleRecord: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> _RoleRecord: ...

    async def delete(self, where: Mapping[str, object]) -> object: ...


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=CommonProxyErrors.not_allowed_access.value,
        )


def _role_table() -> _CustomRoleTable:
    prisma_client: Final = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)
    return CustomRBACRoleRepository(prisma_client).table


def _where_role(role_name: str) -> dict[str, object]:  # mutable-ok: prisma requires a plain dict
    return prisma_args(MappingProxyType({"role_name": role_name}))


def _to_response(record: _RoleRecord) -> CustomRBACRoleResponse:
    return CustomRBACRoleResponse.model_validate(record.dict())


def _reject_reserved_or_invalid(role_name: str, allowed_routes: Sequence[str]) -> None:
    if is_reserved_role_name(role_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"role_name={role_name} is a built-in LiteLLM role and cannot be redefined",
        )
    if any(role.role_name == role_name for role in get_config_custom_rbac_roles()):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"role_name={role_name} is defined in general_settings.custom_rbac_roles, edit the config instead",
        )
    invalid: Final = validate_role_permissions(allowed_routes=allowed_routes)
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"allowed_routes entries must be '*', a LiteLLMRoutes group, or a route path. Invalid: {invalid}",
        )


async def _reject_unknown_inherits(
    role_name: str,
    inherits: Sequence[str],
    table: _CustomRoleTable,
) -> None:
    if not inherits:
        return
    known: Final = (
        frozenset(role.role_name for role in get_config_custom_rbac_roles())
        | frozenset(str(record.dict()["role_name"]) for record in await table.find_many(order=_ORDER_BY_ROLE_NAME))
        | frozenset((role_name,))
    )
    unknown: Final = tuple(parent for parent in inherits if parent not in known)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"inherits references roles that do not exist: {unknown}",
        )


@router.post("/custom_role/new", response_model=CustomRBACRoleResponse, status_code=status.HTTP_201_CREATED)
async def new_custom_role(
    data: CustomRBACRoleCreateRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> CustomRBACRoleResponse:
    _require_proxy_admin(user_api_key_dict)
    _reject_reserved_or_invalid(role_name=data.role_name, allowed_routes=data.allowed_routes)

    table: Final = _role_table()
    if await table.find_unique(where=_where_role(data.role_name)) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"role_name={data.role_name} already exists",
        )
    await _reject_unknown_inherits(role_name=data.role_name, inherits=data.inherits, table=table)

    record: Final = await table.create(
        data=prisma_args(
            MappingProxyType(
                {
                    "role_name": data.role_name,
                    "description": data.description,
                    "allowed_routes": prisma_str_list(data.allowed_routes),
                    "inherits": prisma_str_list(data.inherits),
                    "created_by": user_api_key_dict.user_id,
                    "updated_by": user_api_key_dict.user_id,
                }
            )
        )
    )
    invalidate_custom_rbac_engine_cache()
    return _to_response(record)


@router.post("/custom_role/update", response_model=CustomRBACRoleResponse)
async def update_custom_role(
    data: CustomRBACRoleUpdateRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> CustomRBACRoleResponse:
    _require_proxy_admin(user_api_key_dict)
    _reject_reserved_or_invalid(role_name=data.role_name, allowed_routes=data.allowed_routes or ())

    table: Final = _role_table()
    if await table.find_unique(where=_where_role(data.role_name)) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"role_name={data.role_name} not found",
        )
    await _reject_unknown_inherits(role_name=data.role_name, inherits=data.inherits or (), table=table)

    changes: Final = MappingProxyType(
        {
            key: value
            for key, value in (
                ("description", data.description),
                ("allowed_routes", None if data.allowed_routes is None else prisma_str_list(data.allowed_routes)),
                ("inherits", None if data.inherits is None else prisma_str_list(data.inherits)),
                ("updated_by", user_api_key_dict.user_id),
            )
            if value is not None
        }
    )
    record: Final = await table.update(where=_where_role(data.role_name), data=prisma_args(changes))
    invalidate_custom_rbac_engine_cache()
    return _to_response(record)


@router.get("/custom_role/list", response_model=CustomRBACRoleListResponse)
async def list_custom_roles(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> CustomRBACRoleListResponse:
    _require_proxy_admin(user_api_key_dict)

    config_roles: Final = tuple(
        CustomRBACRoleResponse(
            role_name=role.role_name,
            description=role.description,
            allowed_routes=role.allowed_routes,
            inherits=role.inherits,
            source="config",
        )
        for role in get_config_custom_rbac_roles()
    )
    records: Final = await _role_table().find_many(order=_ORDER_BY_ROLE_NAME)
    return CustomRBACRoleListResponse(roles=config_roles + tuple(_to_response(record) for record in records))


@router.post("/custom_role/delete", response_model=CustomRBACRoleDeleteResponse)
async def delete_custom_role(
    data: CustomRBACRoleDeleteRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> CustomRBACRoleDeleteResponse:
    _require_proxy_admin(user_api_key_dict)

    table: Final = _role_table()
    if await table.find_unique(where=_where_role(data.role_name)) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"role_name={data.role_name} not found",
        )
    await table.delete(where=_where_role(data.role_name))
    invalidate_custom_rbac_engine_cache()
    return CustomRBACRoleDeleteResponse(role_name=data.role_name)
