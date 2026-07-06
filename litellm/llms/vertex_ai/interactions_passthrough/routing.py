from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.llms.vertex_ai.common_utils import get_vertex_interaction_id_from_url
from litellm.llms.vertex_ai.interactions_passthrough.id_codec import decode, encode


class InteractionCreateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None
    previous_interaction_id: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    project: Optional[str]
    location: Optional[str]
    body: dict[str, object]


class _DeploymentLiteLLMParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None


class _Deployment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    litellm_params: Optional[_DeploymentLiteLLMParams] = None


def _deployment_project_location(model: str, llm_router: object) -> tuple[Optional[str], Optional[str]]:
    getter = getattr(llm_router, "get_available_deployment_for_pass_through", None)
    if getter is None:
        return None, None
    try:
        deployment = cast(object, getter(model=model))  # cast-ok: router returns Any; narrowed by pydantic below
    except Exception as e:  # noqa: BLE001 - router lookup is best-effort; any failure falls back to URL values
        verbose_proxy_logger.debug("vertex interactions: deployment lookup failed for model %s: %s", model, e)
        return None, None
    try:
        parsed = _Deployment.model_validate(deployment)
    except ValidationError:
        return None, None
    if parsed.litellm_params is None:
        return None, None
    return parsed.litellm_params.vertex_project, parsed.litellm_params.vertex_location


def resolve_create_project_location(
    body: dict[str, object],
    url_project: Optional[str],
    url_location: Optional[str],
    llm_router: object,
) -> ResolvedRoute:
    parsed = InteractionCreateBody.model_validate(body)

    forwarded_body = body
    prev_project: Optional[str] = None
    prev_location: Optional[str] = None
    if parsed.previous_interaction_id is not None:
        decoded_prev = decode(parsed.previous_interaction_id)
        if decoded_prev is not None:
            prev_project = decoded_prev.project
            prev_location = decoded_prev.location
            forwarded_body = {
                **body,
                "previous_interaction_id": decoded_prev.raw_id,
            }

    model_project: Optional[str] = None
    model_location: Optional[str] = None
    if parsed.model:
        model_project, model_location = _deployment_project_location(parsed.model, llm_router)

    project = model_project or prev_project or url_project
    location = model_location or prev_location or url_location
    return ResolvedRoute(project=project, location=location, body=forwarded_body)


@dataclass(frozen=True, slots=True)
class InputRewrite:
    project: Optional[str]
    location: Optional[str]
    endpoint: str


def rewrite_interaction_input(
    endpoint: str,
    url_project: Optional[str],
    url_location: Optional[str],
) -> InputRewrite:
    interaction_id = get_vertex_interaction_id_from_url(endpoint)
    if interaction_id is None:
        return InputRewrite(project=url_project, location=url_location, endpoint=endpoint)
    decoded = decode(interaction_id)
    if decoded is None:
        return InputRewrite(project=url_project, location=url_location, endpoint=endpoint)
    new_endpoint = endpoint.replace(interaction_id, decoded.raw_id, 1)
    return InputRewrite(project=decoded.project, location=decoded.location, endpoint=new_endpoint)


def encode_interaction_response_id(
    response_body: dict[str, object],
    project: Optional[str],
    location: Optional[str],
) -> dict[str, object]:
    if project is None or location is None:
        return response_body
    raw_id = response_body.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        return response_body
    return {**response_body, "id": encode(project, location, raw_id)}
