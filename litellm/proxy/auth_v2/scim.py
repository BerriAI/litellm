from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar

from fastapi import APIRouter, Query, Request, Response, Security, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from scim2_models import (
    Bulk,
    ChangePassword,
    Context,
    Error,
    Filter,
    Group,
    ListResponse,
    Patch,
    PatchOp,
    Resource,
    ResourceType,
    Schema,
    ServiceProviderConfig,
    Sort,
    User,
)

from .resolver import ProvisioningStore
from .security import get_current_principal

R = TypeVar("R", bound=Resource)


def _store(request: Request) -> ProvisioningStore:
    return request.app.state.auth_v2.resolver


def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=Error(status=str(status_code), detail=detail).model_dump(),
    )


async def _parse(request: Request, model: Type[R]) -> R:
    body = await request.json()
    return model.model_validate(body, scim_ctx=Context.RESOURCE_CREATION_REQUEST)


def _set_path(data: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    node = data
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _remove_path(data: Dict[str, Any], path: str) -> None:
    keys = path.split(".")
    node = data
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            return
        node = child
    node.pop(keys[-1], None)


def _apply_patch(resource: R, patch: PatchOp) -> R:
    data: Dict[str, Any] = resource.model_dump()
    for op in patch.operations:
        action = op.op.value if hasattr(op.op, "value") else str(op.op)
        if op.path is not None and ("[" in op.path or "]" in op.path):
            raise ValueError(f"unsupported SCIM patch path filter: {op.path}")
        if action == "remove":
            if op.path:
                _remove_path(data, op.path)
            continue
        if op.path is None and isinstance(op.value, dict):
            data.update(op.value)
        elif op.path is not None:
            _set_path(data, op.path, op.value)
    return type(resource).model_validate(data)


def _dump(resource: Resource, ctx: Context) -> Dict[str, Any]:
    return resource.model_dump(scim_ctx=ctx)


def _build_discovery_router() -> APIRouter:
    router = APIRouter()

    @router.get("/ServiceProviderConfig")
    async def service_provider_config() -> Response:
        config = ServiceProviderConfig(
            patch=Patch(supported=True),
            bulk=Bulk(supported=False, max_operations=0, max_payload_size=0),
            filter=Filter(supported=False, max_results=0),
            change_password=ChangePassword(supported=False),
            sort=Sort(supported=False),
            etag=None,
            authentication_schemes=[],
        )
        return JSONResponse(content=config.model_dump())

    @router.get("/ResourceTypes")
    async def resource_types() -> Response:
        types = [
            ResourceType(
                id="User",
                name="User",
                endpoint="/Users",
                schema="urn:ietf:params:scim:schemas:core:2.0:User",
            ),
            ResourceType(
                id="Group",
                name="Group",
                endpoint="/Groups",
                schema="urn:ietf:params:scim:schemas:core:2.0:Group",
            ),
        ]
        listing: ListResponse[ResourceType] = ListResponse[ResourceType](
            total_results=len(types),
            start_index=1,
            items_per_page=len(types),
            resources=types,
        )
        return JSONResponse(content=listing.model_dump())

    @router.get("/Schemas")
    async def schemas() -> Response:
        resources = [User.to_schema(), Group.to_schema()]
        listing: ListResponse[Schema] = ListResponse[Schema](
            total_results=len(resources),
            start_index=1,
            items_per_page=len(resources),
            resources=resources,
        )
        return JSONResponse(content=listing.model_dump())

    return router


def _build_protected_router() -> APIRouter:
    protected = APIRouter(
        dependencies=[Security(get_current_principal, scopes=["scim:write"])],
    )

    @protected.post("/Users", status_code=status.HTTP_201_CREATED)
    async def create_user(request: Request) -> Response:
        try:
            user = await _parse(request, User)
        except ValidationError as exc:
            return _error(status.HTTP_400_BAD_REQUEST, str(exc))
        stored = await _store(request).upsert_user(user)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=_dump(stored, Context.RESOURCE_CREATION_RESPONSE),
        )

    @protected.get("/Users/{resource_id}")
    async def get_user(resource_id: str, request: Request) -> Response:
        user = await _store(request).get_user(resource_id)
        if user is None:
            return _error(status.HTTP_404_NOT_FOUND, f"User {resource_id} not found")
        return JSONResponse(content=_dump(user, Context.RESOURCE_QUERY_RESPONSE))

    @protected.patch("/Users/{resource_id}")
    async def patch_user(resource_id: str, request: Request) -> Response:
        store = _store(request)
        user = await store.get_user(resource_id)
        if user is None:
            return _error(status.HTTP_404_NOT_FOUND, f"User {resource_id} not found")
        try:
            patch = PatchOp[User].model_validate(await request.json())
            patched = _apply_patch(user, patch)
        except (ValidationError, ValueError) as exc:
            return _error(status.HTTP_400_BAD_REQUEST, str(exc))
        updated = await store.upsert_user(patched)
        return JSONResponse(content=_dump(updated, Context.RESOURCE_PATCH_RESPONSE))

    @protected.delete("/Users/{resource_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def deactivate_user(resource_id: str, request: Request) -> Response:
        store = _store(request)
        if await store.get_user(resource_id) is None:
            return _error(status.HTTP_404_NOT_FOUND, f"User {resource_id} not found")
        await store.deactivate_user(resource_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @protected.get("/Users")
    async def list_users(
        request: Request,
        filter_expr: Optional[str] = Query(default=None, alias="filter"),
    ) -> Response:
        users = await _store(request).list_users(filter_expr)
        listing: ListResponse[User] = ListResponse[User](
            total_results=len(users),
            start_index=1,
            items_per_page=len(users),
            resources=users or None,
        )
        return JSONResponse(content=_dump(listing, Context.RESOURCE_QUERY_RESPONSE))

    @protected.post("/Groups", status_code=status.HTTP_201_CREATED)
    async def create_group(request: Request) -> Response:
        try:
            group = await _parse(request, Group)
        except ValidationError as exc:
            return _error(status.HTTP_400_BAD_REQUEST, str(exc))
        stored = await _store(request).upsert_group(group)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=_dump(stored, Context.RESOURCE_CREATION_RESPONSE),
        )

    @protected.get("/Groups/{resource_id}")
    async def get_group(resource_id: str, request: Request) -> Response:
        group = await _store(request).get_group(resource_id)
        if group is None:
            return _error(status.HTTP_404_NOT_FOUND, f"Group {resource_id} not found")
        return JSONResponse(content=_dump(group, Context.RESOURCE_QUERY_RESPONSE))

    @protected.patch("/Groups/{resource_id}")
    async def patch_group(resource_id: str, request: Request) -> Response:
        store = _store(request)
        group = await store.get_group(resource_id)
        if group is None:
            return _error(status.HTTP_404_NOT_FOUND, f"Group {resource_id} not found")
        try:
            patch = PatchOp[Group].model_validate(await request.json())
            patched = _apply_patch(group, patch)
        except (ValidationError, ValueError) as exc:
            return _error(status.HTTP_400_BAD_REQUEST, str(exc))
        updated = await store.upsert_group(patched)
        return JSONResponse(content=_dump(updated, Context.RESOURCE_PATCH_RESPONSE))

    @protected.delete("/Groups/{resource_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_group(resource_id: str, request: Request) -> Response:
        store = _store(request)
        if await store.get_group(resource_id) is None:
            return _error(status.HTTP_404_NOT_FOUND, f"Group {resource_id} not found")
        await store.delete_group(resource_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @protected.get("/Groups")
    async def list_groups(
        request: Request,
        filter_expr: Optional[str] = Query(default=None, alias="filter"),
    ) -> Response:
        groups = await _store(request).list_groups(filter_expr)
        listing: ListResponse[Group] = ListResponse[Group](
            total_results=len(groups),
            start_index=1,
            items_per_page=len(groups),
            resources=groups or None,
        )
        return JSONResponse(content=_dump(listing, Context.RESOURCE_QUERY_RESPONSE))

    return protected


def build_scim_router() -> APIRouter:
    router = APIRouter(prefix="/scim/v2", tags=["scim"])
    router.include_router(_build_protected_router())
    router.include_router(_build_discovery_router())
    return router
