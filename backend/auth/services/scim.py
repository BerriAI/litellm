from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, List, Type, TypeVar

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
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

R = TypeVar("R", bound=Resource)


def scim_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=Error(status=str(status_code), detail=detail).model_dump(),
    )


class ScimErrorRoute(APIRoute):
    """Render authentication failures with the SCIM Error schema (RFC 7644)."""

    def get_route_handler(  # type: ignore[override]
        self,
    ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def scim_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except HTTPException as exc:
                if exc.status_code not in (
                    status.HTTP_401_UNAUTHORIZED,
                    status.HTTP_403_FORBIDDEN,
                ):
                    raise
                response = scim_error(exc.status_code, str(exc.detail))
                if exc.headers:
                    response.headers.update(exc.headers)
                return response

        return scim_handler


def parse_resource(body: Any, model: Type[R]) -> R:
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


def _targets_read_only_id(op: Any) -> bool:
    if op.path is not None:
        return op.path.split(".")[0].strip().lower() == "id"
    return isinstance(op.value, dict) and any(str(k).lower() == "id" for k in op.value)


def apply_patch(resource: R, body: Any) -> R:
    patch = PatchOp[type(resource)].model_validate(body)
    data: Dict[str, Any] = resource.model_dump()
    for op in patch.operations:
        action = op.op.value if hasattr(op.op, "value") else str(op.op)
        if op.path is not None and ("[" in op.path or "]" in op.path):
            raise ValueError(f"unsupported SCIM patch path filter: {op.path}")
        if _targets_read_only_id(op):
            raise ValueError("the SCIM id attribute is read-only")
        if action == "remove":
            if op.path:
                _remove_path(data, op.path)
            continue
        if op.path is None and isinstance(op.value, dict):
            data.update(op.value)
        elif op.path is not None:
            _set_path(data, op.path, op.value)
    return type(resource).model_validate(data)


def creation_response(resource: Resource) -> Dict[str, Any]:
    return resource.model_dump(scim_ctx=Context.RESOURCE_CREATION_RESPONSE)


def query_response(resource: Resource) -> Dict[str, Any]:
    return resource.model_dump(scim_ctx=Context.RESOURCE_QUERY_RESPONSE)


def patch_response(resource: Resource) -> Dict[str, Any]:
    return resource.model_dump(scim_ctx=Context.RESOURCE_PATCH_RESPONSE)


def list_response(model: Type[R], items: List[R]) -> Dict[str, Any]:
    listing = ListResponse[model](
        total_results=len(items),
        start_index=1,
        items_per_page=len(items),
        resources=items or None,
    )
    return listing.model_dump(scim_ctx=Context.RESOURCE_QUERY_RESPONSE)


def service_provider_config() -> Dict[str, Any]:
    return ServiceProviderConfig(
        patch=Patch(supported=True),
        bulk=Bulk(supported=False, max_operations=0, max_payload_size=0),
        filter=Filter(supported=False, max_results=0),
        change_password=ChangePassword(supported=False),
        sort=Sort(supported=False),
        etag=None,
        authentication_schemes=[],
    ).model_dump()


def resource_types() -> Dict[str, Any]:
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
    return ListResponse[ResourceType](
        total_results=len(types),
        start_index=1,
        items_per_page=len(types),
        resources=types,
    ).model_dump()


def schemas() -> Dict[str, Any]:
    resources = [User.to_schema(), Group.to_schema()]
    return ListResponse[Schema](
        total_results=len(resources),
        start_index=1,
        items_per_page=len(resources),
        resources=resources,
    ).model_dump()
