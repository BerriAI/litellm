from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.llms.vertex_ai.common_utils import get_vertex_interaction_id_from_url
from litellm.llms.vertex_ai.interactions_passthrough.id_codec import decode, encode

InteractionBody: TypeAlias = dict[str, object]  # mutable-ok: pass-through request state only accepts dict payloads


class _PassThroughDeploymentRouter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)
    get_available_deployment_for_pass_through: Callable[..., object]


class InteractionCreateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    previous_interaction_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    project: str | None
    location: str | None
    body: InteractionBody


class _DeploymentLiteLLMParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    vertex_project: str | None = None
    vertex_location: str | None = None


class _Deployment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    litellm_params: _DeploymentLiteLLMParams | None = None


def _deployment_project_location(model: str, llm_router: object) -> tuple[str | None, str | None]:
    try:
        router: Final = _PassThroughDeploymentRouter.model_validate(llm_router)
    except ValidationError:
        return None, None
    try:
        deployment: Final = router.get_available_deployment_for_pass_through(model=model)
    except Exception as error:  # noqa: BLE001 - router lookup is best-effort; any failure falls back to URL values
        verbose_proxy_logger.debug("vertex interactions: deployment lookup failed for model %s: %s", model, error)
        return None, None
    try:
        parsed: Final = _Deployment.model_validate(deployment)
    except ValidationError:
        return None, None
    litellm_params: Final = parsed.litellm_params
    if litellm_params is None:
        return None, None
    return litellm_params.vertex_project, litellm_params.vertex_location


def resolve_create_project_location(
    body: InteractionBody,
    url_project: str | None,
    url_location: str | None,
    llm_router: object,
) -> ResolvedRoute:
    parsed: Final = InteractionCreateBody.model_validate(body)
    decoded_prev: Final = decode(parsed.previous_interaction_id) if parsed.previous_interaction_id is not None else None
    forwarded_body: Final[InteractionBody] = (
        {**body, "previous_interaction_id": decoded_prev.raw_id}  # mutable-ok: downstream state requires a dict
        if decoded_prev is not None
        else body
    )
    prev_project: Final = decoded_prev.project if decoded_prev is not None else None
    prev_location: Final = decoded_prev.location if decoded_prev is not None else None
    model_project, model_location = (
        _deployment_project_location(parsed.model, llm_router) if parsed.model else (None, None)
    )
    project: Final = model_project or prev_project or url_project
    location: Final = model_location or prev_location or url_location
    return ResolvedRoute(project=project, location=location, body=forwarded_body)


@dataclass(frozen=True, slots=True)
class InputRewrite:
    project: str | None
    location: str | None
    endpoint: str


def rewrite_interaction_input(
    endpoint: str,
    url_project: str | None,
    url_location: str | None,
) -> InputRewrite:
    interaction_id: Final = get_vertex_interaction_id_from_url(endpoint)
    if interaction_id is None:
        return InputRewrite(project=url_project, location=url_location, endpoint=endpoint)
    decoded: Final = decode(interaction_id)
    if decoded is None:
        return InputRewrite(project=url_project, location=url_location, endpoint=endpoint)
    new_endpoint: Final = endpoint.replace(interaction_id, decoded.raw_id, 1)
    return InputRewrite(project=decoded.project, location=decoded.location, endpoint=new_endpoint)


def encode_interaction_response_id(
    response_body: InteractionBody,
    project: str | None,
    location: str | None,
) -> InteractionBody:
    if project is None or location is None:
        return response_body
    raw_id: Final = response_body.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        return response_body
    return {**response_body, "id": encode(project, location, raw_id)}  # mutable-ok: JSON response requires a dict
