from typing import Annotated, Final

from fastapi import APIRouter, Depends, Response, Security

from litellm.proxy.control_plane.auth.dependencies import require_scopes
from litellm.proxy.control_plane.auth.rbac import Principal, Scope
from litellm.proxy.control_plane.common.exceptions import ControlPlaneException, ErrorBody
from litellm.proxy.control_plane.settings import RoleSpec
from litellm.proxy.control_plane.store.dependencies import audited_write, get_runtime_config
from litellm.proxy.control_plane.store.models import ENTITY, Change
from litellm.proxy.control_plane.store.runtime import RuntimeConfig
from litellm.types.proxy.control_plane.roles import RoleListResponse, RoleResponse

router: Final = APIRouter(prefix="/roles", tags=["roles"])


@router.get("", response_model=RoleListResponse)
async def list_roles(
    _principal: Annotated[Principal, Security(require_scopes, scopes=[Scope.ROLES_WRITE.value])],
    runtime: Annotated[RuntimeConfig, Depends(get_runtime_config)],
) -> RoleListResponse:
    view: Final = runtime.state.view
    return RoleListResponse(
        items=tuple(
            RoleResponse(
                name=name,
                scopes=tuple(sorted(scope.value for scope in view.policy.scopes_for(name))),
                inherits=spec.inherits,
                source="config" if runtime.state.protected("role", name) else "runtime",
            )
            for name, spec in sorted(view.roles.items())
        )
    )


@router.put("/{name}", response_model=RoleResponse)
async def put_role(
    name: str,
    spec: RoleSpec,
    principal: Annotated[Principal, Security(audited_write, scopes=[Scope.ROLES_WRITE.value])],
    runtime: Annotated[RuntimeConfig, Depends(get_runtime_config)],
) -> RoleResponse:
    change: Final = Change(
        op="upsert",
        kind="role",
        name=name,
        entity=ENTITY.validate_json(spec.model_dump_json()),
        actor=principal.subject,
    )
    result: Final = await runtime.write(change, principal)
    if isinstance(result, ErrorBody):
        raise ControlPlaneException(result)
    return RoleResponse(
        name=name,
        scopes=tuple(sorted(scope.value for scope in result.snapshot.policy.scopes_for(name))),
        inherits=spec.inherits,
        source="runtime",
    )


@router.delete("/{name}", status_code=204)
async def delete_role(
    name: str,
    principal: Annotated[Principal, Security(audited_write, scopes=[Scope.ROLES_WRITE.value])],
    runtime: Annotated[RuntimeConfig, Depends(get_runtime_config)],
) -> Response:
    result: Final = await runtime.write(Change(op="delete", kind="role", name=name, actor=principal.subject), principal)
    if isinstance(result, ErrorBody):
        raise ControlPlaneException(result)
    return Response(status_code=204)
