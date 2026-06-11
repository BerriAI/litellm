from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from scim2_models import Group, User

from ..services import scim
from .dependencies import scim_principal, scim_store

router = APIRouter(prefix="/scim/v2", tags=["scim"], route_class=scim.ScimErrorRoute)
_protected = [Depends(scim_principal)]


@router.post("/Users", status_code=status.HTTP_201_CREATED, dependencies=_protected)
async def create_user(request: Request) -> Response:
    try:
        user = scim.parse_resource(await request.json(), User)
    except ValidationError as exc:
        return scim.scim_error(status.HTTP_400_BAD_REQUEST, str(exc))
    stored = await scim_store(request).upsert_user(user)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=scim.creation_response(stored),
    )


@router.get("/Users/{resource_id}", dependencies=_protected)
async def get_user(resource_id: str, request: Request) -> Response:
    user = await scim_store(request).get_user(resource_id)
    if user is None:
        return scim.scim_error(status.HTTP_404_NOT_FOUND, f"User {resource_id} not found")
    return JSONResponse(content=scim.query_response(user))


@router.patch("/Users/{resource_id}", dependencies=_protected)
async def patch_user(resource_id: str, request: Request) -> Response:
    store = scim_store(request)
    user = await store.get_user(resource_id)
    if user is None:
        return scim.scim_error(status.HTTP_404_NOT_FOUND, f"User {resource_id} not found")
    try:
        patched = scim.apply_patch(user, await request.json())
    except (ValidationError, ValueError) as exc:
        return scim.scim_error(status.HTTP_400_BAD_REQUEST, str(exc))
    updated = await store.upsert_user(patched)
    return JSONResponse(content=scim.patch_response(updated))


@router.delete(
    "/Users/{resource_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=_protected,
)
async def deactivate_user(resource_id: str, request: Request) -> Response:
    store = scim_store(request)
    if await store.get_user(resource_id) is None:
        return scim.scim_error(status.HTTP_404_NOT_FOUND, f"User {resource_id} not found")
    await store.deactivate_user(resource_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/Users", dependencies=_protected)
async def list_users(
    request: Request,
    filter_expr: Optional[str] = Query(default=None, alias="filter"),
) -> Response:
    users = await scim_store(request).list_users(filter_expr)
    return JSONResponse(content=scim.list_response(User, users))


@router.post("/Groups", status_code=status.HTTP_201_CREATED, dependencies=_protected)
async def create_group(request: Request) -> Response:
    try:
        group = scim.parse_resource(await request.json(), Group)
    except ValidationError as exc:
        return scim.scim_error(status.HTTP_400_BAD_REQUEST, str(exc))
    stored = await scim_store(request).upsert_group(group)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=scim.creation_response(stored),
    )


@router.get("/Groups/{resource_id}", dependencies=_protected)
async def get_group(resource_id: str, request: Request) -> Response:
    group = await scim_store(request).get_group(resource_id)
    if group is None:
        return scim.scim_error(status.HTTP_404_NOT_FOUND, f"Group {resource_id} not found")
    return JSONResponse(content=scim.query_response(group))


@router.patch("/Groups/{resource_id}", dependencies=_protected)
async def patch_group(resource_id: str, request: Request) -> Response:
    store = scim_store(request)
    group = await store.get_group(resource_id)
    if group is None:
        return scim.scim_error(status.HTTP_404_NOT_FOUND, f"Group {resource_id} not found")
    try:
        patched = scim.apply_patch(group, await request.json())
    except (ValidationError, ValueError) as exc:
        return scim.scim_error(status.HTTP_400_BAD_REQUEST, str(exc))
    updated = await store.upsert_group(patched)
    return JSONResponse(content=scim.patch_response(updated))


@router.delete(
    "/Groups/{resource_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=_protected,
)
async def delete_group(resource_id: str, request: Request) -> Response:
    store = scim_store(request)
    if await store.get_group(resource_id) is None:
        return scim.scim_error(status.HTTP_404_NOT_FOUND, f"Group {resource_id} not found")
    await store.delete_group(resource_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/Groups", dependencies=_protected)
async def list_groups(
    request: Request,
    filter_expr: Optional[str] = Query(default=None, alias="filter"),
) -> Response:
    groups = await scim_store(request).list_groups(filter_expr)
    return JSONResponse(content=scim.list_response(Group, groups))


@router.get("/ServiceProviderConfig")
async def service_provider_config() -> Response:
    return JSONResponse(content=scim.service_provider_config())


@router.get("/ResourceTypes")
async def resource_types() -> Response:
    return JSONResponse(content=scim.resource_types())


@router.get("/Schemas")
async def schemas() -> Response:
    return JSONResponse(content=scim.schemas())
