"""Live SAP AI Core deployment discovery for the Admin UI.

`POST /sap/deployments` takes a service key, mints a short-lived bearer token, and lists the running
foundation-model deployments that key can see, so the UI can offer them as a pick list instead of the
operator hand-copying deployment URLs. The service key is used only to mint the token for this one
request. It is never logged (that would leak it into `--detailed_debug` request-body logs) or persisted.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Final, Protocol

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import litellm
from litellm.llms.sap.deployment import alist_deployments
from litellm.llms.sap.direct_connect import build_sap_auth, get_token_creator
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.health_endpoints._health_endpoints import (
    _reject_os_environ_references,  # pyright: ignore[reportPrivateUsage]  # canonical env-ref guard, shared with health endpoints
)

if TYPE_CHECKING:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.llms.sap.direct_connect import TokenCreatorFactory

router: Final = APIRouter()


class SapDeploymentDiscoveryRequest(BaseModel):
    service_key: str
    resource_group: str | None = None


class SapDeploymentInfo(BaseModel):
    model_name: str
    deployment_url: str
    id: str
    status: str
    created_at: datetime


class SapDeploymentDiscoveryResponse(BaseModel):
    deployments: tuple[SapDeploymentInfo, ...]


class _DeploymentDiscovery(Protocol):
    """The one behaviour the route depends on: mint a token and list RUNNING deployments.

    Injected via `Depends(_default_discovery)` so tests override it through `app.dependency_overrides`
    with a fake, exercising rejection, exception mapping, and response shaping without a live SAP call.
    """

    async def __call__(
        self,
        *,
        service_key: str,
        resource_group: str | None,
        token_creator_factory: TokenCreatorFactory,
        http_client: AsyncHTTPHandler,
    ) -> tuple[SapDeploymentInfo, ...]: ...


async def discover_running_deployments(
    *,
    service_key: str,
    resource_group: str | None,
    token_creator_factory: TokenCreatorFactory,
    http_client: AsyncHTTPHandler,
) -> tuple[SapDeploymentInfo, ...]:
    """Mint a token from `service_key` and return the RUNNING deployments it can see.

    An optional `resource_group` overrides the one the service key resolves to, so the UI can discover
    deployments in a non-`default` AI Core resource group. The token mint is a blocking OAuth call, so
    it runs in a worker thread to keep the event loop free. Only deployments that report a backend model
    name are returned, since the UI keys its pick list on it.
    """
    sap_headers, base_url, _resource_group = await asyncio.to_thread(
        build_sap_auth, service_key, token_creator_factory, resource_group=resource_group
    )
    deployments: Final = await alist_deployments(base_url=base_url, headers=sap_headers, http_client=http_client)
    return tuple(
        SapDeploymentInfo(
            model_name=deployment.model_name,
            deployment_url=deployment.deployment_url,
            id=deployment.id,
            status=deployment.status,
            created_at=deployment.created_at,
        )
        for deployment in deployments
        if deployment.model_name is not None
    )


def _default_discovery() -> _DeploymentDiscovery:
    return discover_running_deployments


def _error_detail(message: str) -> dict[str, str]:  # mutable-ok: FastAPI error response payload
    return {"error": message}  # mutable-ok: FastAPI error response payload


@router.post(
    "/sap/deployments",
    tags=["SAP AI Core"],  # mutable-ok: FastAPI route metadata
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI route metadata
)
async def list_sap_deployments_endpoint(
    request: SapDeploymentDiscoveryRequest,
    discover: _DeploymentDiscovery = Depends(_default_discovery),
) -> SapDeploymentDiscoveryResponse:
    _reject_os_environ_references(request.model_dump())
    try:
        deployments: Final = await discover(
            service_key=request.service_key,
            resource_group=request.resource_group,
            token_creator_factory=get_token_creator,
            http_client=litellm.module_level_aclient,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=_error_detail(f"Invalid SAP AI Core service key: {e}")) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=502,
            detail=_error_detail("Failed to obtain an SAP AI Core token from the service key."),
        ) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=_error_detail("Failed to list SAP AI Core deployments.")) from e
    return SapDeploymentDiscoveryResponse(deployments=deployments)
