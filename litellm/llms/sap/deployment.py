"""Auto-discovery of SAP AI Core foundation-model deployments.

Maps a model name (e.g. ``anthropic--claude-4.8-opus``) to the running
deployment that serves it by querying the AI Core Deployment API
(``GET /lm/deployments``), so callers never hardcode deployment ids.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, Field

from .chat.handler import GenAIHubOrchestrationError

if TYPE_CHECKING:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

FOUNDATION_MODELS_SCENARIO: Final = "foundation-models"
_RUNNING: Final = "RUNNING"
_DEPLOYMENTS_PATH: Final = "/lm/deployments"


class _BackendModel(BaseModel):
    name: str
    version: str = "latest"


class _BackendDetails(BaseModel):
    model: _BackendModel


class _DeploymentResources(BaseModel):
    backend_details: _BackendDetails = Field(alias="backendDetails")


class _DeploymentDetails(BaseModel):
    resources: _DeploymentResources | None = None


class DeploymentResource(BaseModel):
    id: str
    deployment_url: str = Field(alias="deploymentUrl")
    created_at: datetime = Field(alias="createdAt")
    scenario_id: str = Field(alias="scenarioId")
    status: str
    details: _DeploymentDetails | None = None

    @property
    def model_name(self) -> str | None:
        if self.details is not None and self.details.resources is not None:
            return self.details.resources.backend_details.model.name
        return None


class DeploymentQueryResponse(BaseModel):
    resources: tuple[DeploymentResource, ...] = Field(default_factory=tuple)


def select_deployment_url(resources: Sequence[DeploymentResource], model: str) -> str:
    """Pick the newest RUNNING deployment whose backend model name equals ``model``.

    Raises ``GenAIHubOrchestrationError`` (404) listing discoverable model names
    when nothing matches, so misconfigured model names fail loudly.
    """
    matches: Final = tuple(r for r in resources if r.status == _RUNNING and r.model_name == model)
    if not matches:
        available: Final = sorted(frozenset(r.model_name for r in resources if r.model_name is not None))
        raise GenAIHubOrchestrationError(
            status_code=404,
            message=(
                f"No running SAP AI Core deployment found for model '{model}'. "
                f"Discoverable running models: {available or 'none'}. "
                "Create/start a foundation-model deployment for this model in SAP AI Launchpad, then retry."
            ),
        )
    newest: Final = max(matches, key=lambda r: r.created_at)
    return newest.deployment_url


def _query_url(base_url: str) -> str:
    return f"{base_url}{_DEPLOYMENTS_PATH}"


def _query_params(scenario_id: str) -> dict[str, str]:  # mutable-ok: consumed as HTTPHandler.get params dict
    return {"scenarioId": scenario_id, "status": _RUNNING}  # mutable-ok: HTTPHandler.get params dict


def resolve_deployment_url(
    *,
    base_url: str,
    headers: dict[str, str],  # mutable-ok: consumed as HTTPHandler.get headers dict
    model: str,
    http_client: HTTPHandler,
    scenario_id: str = FOUNDATION_MODELS_SCENARIO,
) -> str:
    response: Final = http_client.get(_query_url(base_url), params=_query_params(scenario_id), headers=headers)
    parsed: Final = DeploymentQueryResponse.model_validate(response.json())
    return select_deployment_url(parsed.resources, model)


async def aresolve_deployment_url(
    *,
    base_url: str,
    headers: dict[str, str],  # mutable-ok: consumed as HTTPHandler.get headers dict
    model: str,
    http_client: AsyncHTTPHandler,
    scenario_id: str = FOUNDATION_MODELS_SCENARIO,
) -> str:
    response: Final = await http_client.get(_query_url(base_url), params=_query_params(scenario_id), headers=headers)
    parsed: Final = DeploymentQueryResponse.model_validate(response.json())
    return select_deployment_url(parsed.resources, model)


def list_deployments(
    *,
    base_url: str,
    headers: dict[str, str],  # mutable-ok: consumed as HTTPHandler.get headers dict
    http_client: HTTPHandler,
    scenario_id: str = FOUNDATION_MODELS_SCENARIO,
) -> tuple[DeploymentResource, ...]:
    """Return every RUNNING foundation-model deployment, for the discovery UI to list and pick from."""
    response: Final = http_client.get(_query_url(base_url), params=_query_params(scenario_id), headers=headers)
    parsed: Final = DeploymentQueryResponse.model_validate(response.json())
    return tuple(r for r in parsed.resources if r.status == _RUNNING)


async def alist_deployments(
    *,
    base_url: str,
    headers: dict[str, str],  # mutable-ok: consumed as HTTPHandler.get headers dict
    http_client: AsyncHTTPHandler,
    scenario_id: str = FOUNDATION_MODELS_SCENARIO,
) -> tuple[DeploymentResource, ...]:
    response: Final = await http_client.get(_query_url(base_url), params=_query_params(scenario_id), headers=headers)
    parsed: Final = DeploymentQueryResponse.model_validate(response.json())
    return tuple(r for r in parsed.resources if r.status == _RUNNING)
