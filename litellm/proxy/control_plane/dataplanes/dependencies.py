"""FastAPI glue for the dataplanes domain: registry and client injection, per-dataplane access checks."""

from typing import Annotated, Final

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials

from litellm.proxy.control_plane.auth.dependencies import bearer_scheme, get_principal
from litellm.proxy.control_plane.auth.rbac import Principal
from litellm.proxy.control_plane.common.exceptions import ControlPlaneException, forbidden, not_found, unauthorized
from litellm.proxy.control_plane.dataplanes.client import DataplaneClient
from litellm.proxy.control_plane.dataplanes.credentials import Caller
from litellm.proxy.control_plane.dataplanes.registry import Dataplane, DataplaneRegistry


def get_registry() -> DataplaneRegistry:
    raise RuntimeError("get_registry is not bound, build the app with create_app")


def get_dataplane_client() -> DataplaneClient:
    raise RuntimeError("get_dataplane_client is not bound, build the app with create_app")


async def get_caller(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> Caller:
    if credentials is None:
        raise ControlPlaneException(unauthorized("Missing bearer credentials."))
    return Caller(principal=principal, bearer=credentials.credentials)


async def dataplane_access(
    dataplane_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    registry: Annotated[DataplaneRegistry, Depends(get_registry)],
) -> Dataplane:
    dataplane: Final = registry.get(dataplane_id)
    if dataplane is None:
        raise ControlPlaneException(not_found(f"Unknown dataplane {dataplane_id!r}."))
    if not principal.dataplanes.allows(dataplane_id):
        raise ControlPlaneException(forbidden(f"Not bound to dataplane {dataplane_id!r}."))
    return dataplane
