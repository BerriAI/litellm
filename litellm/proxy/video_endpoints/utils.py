from typing import TYPE_CHECKING, Any, Final

import orjson
from pydantic import BaseModel, ConfigDict

from litellm.types.videos.main import DecodedVideoId, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    encode_video_id_with_provider,
    id_safe_name,
)

if TYPE_CHECKING:
    from litellm.router import Router


def extract_model_from_target_model_names(target_model_names: object) -> str | None:
    if isinstance(target_model_names, str):
        model_names: Final = [m.strip() for m in target_model_names.split(",") if m.strip()]
        return model_names[0] if model_names else None
    if not isinstance(target_model_names, list):
        return None
    return target_model_names[0] if target_model_names else None


def video_reference_to_id(video_ref: object) -> str:
    if isinstance(video_ref, dict):
        return video_ref.get("id", "")
    if not isinstance(video_ref, str):
        return ""
    try:
        parsed_ref: Final = orjson.loads(video_ref)
    except orjson.JSONDecodeError:
        return video_ref
    return parsed_ref.get("id", "") if isinstance(parsed_ref, dict) else video_ref


def get_custom_provider_from_data(data: dict[str, Any]) -> str | None:
    custom_llm_provider: Final = data.get("custom_llm_provider")
    if custom_llm_provider:
        return custom_llm_provider

    extra_body = data.get("extra_body")
    if isinstance(extra_body, str):
        try:
            parsed_extra_body: Final = orjson.loads(extra_body)
            if isinstance(parsed_extra_body, dict):
                extra_body = parsed_extra_body
        except Exception:
            extra_body = None

    if isinstance(extra_body, dict):
        extra_body_custom_llm_provider: Final = extra_body.get("custom_llm_provider")
        if isinstance(extra_body_custom_llm_provider, str):
            return extra_body_custom_llm_provider

    return None


def encode_character_id_in_response(response: Any, custom_llm_provider: str, model_id: str | None) -> object:
    if isinstance(response, dict) and response.get("id"):
        response["id"] = encode_character_id_with_provider(
            character_id=response["id"],
            provider=custom_llm_provider,
            model_id=model_id,
        )
        return response

    character_id: Final = getattr(response, "id", None)
    if isinstance(character_id, str) and character_id:
        response.id = encode_character_id_with_provider(
            character_id=character_id,
            provider=custom_llm_provider,
            model_id=model_id,
        )
    return response


class VideoRequestRoute(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: object = None
    custom_llm_provider: object = None


def routable_model_name(model: object, llm_router: "Router | None", custom_llm_provider: object) -> str | None:
    name: Final = id_safe_name(model)
    if name is None or llm_router is None:
        return name
    provider: Final = custom_llm_provider if isinstance(custom_llm_provider, str) else None
    return name if llm_router.resolve_model_name_from_model_id(name, provider) else None


def video_model_group(decoded: DecodedVideoId, llm_router: "Router | None") -> str | None:
    """
    The group a video id names. A provider model that several groups share resolves to the group of the
    deployment that created the video, under its team's public name for a team model, while a name the router
    routes by itself (the id's model, bare or provider-prefixed), or one that routes to that deployment's group
    (an alias of it), is kept as written.
    """
    if llm_router is None:
        return None
    model_id: Final = decoded.get("model_id")
    provider: Final = decoded.get("custom_llm_provider")
    resolved: Final = llm_router.resolve_model_name_from_model_id(model_id, provider) if model_id else None
    deployment_id: Final = decoded.get("deployment_id")
    if not deployment_id or resolved in (model_id, f"{provider}/{model_id}"):
        return resolved
    deployment: Final = llm_router.get_deployment(model_id=deployment_id)
    if deployment is None:
        return resolved
    creating_group: Final = deployment.model_info.team_public_model_name or deployment.model_name
    if resolved is not None and llm_router.resolve_served_model_name(resolved, creating_group) == creating_group:
        return resolved
    return creating_group


def fallback_requested_model(requested_model: str | None, response: object, llm_router: "Router | None") -> str | None:
    """The group a create asked for when one of that group's router fallbacks served it."""
    from litellm.proxy.auth.auth_utils import router_fallback_model_names

    if requested_model is None or llm_router is None or not isinstance(response, VideoObject):
        return None
    deployment_id: Final = id_safe_name(response.hidden_params.get("model_id"))
    deployment: Final = llm_router.get_deployment(model_id=deployment_id) if deployment_id else None
    if deployment is None:
        return None
    creating_group: Final = deployment.model_info.team_public_model_name or deployment.model_name
    return requested_model if creating_group in router_fallback_model_names(requested_model, llm_router) else None


def encode_video_id_in_response(
    response: object,
    fallback_provider: object,
    fallback_model_id: object,
    polled_video: DecodedVideoId | None = None,
    requested_model: str | None = None,
) -> object:
    """
    Stamps the provider, model and the deployment that answered into a returned video id. A poll of an existing
    video (`polled_video`) answered without a routed deployment keeps the deployment its id already named, and
    every returned id keeps the group a router fallback was serving (`requested_model`).
    """
    if not isinstance(response, VideoObject) or not response.id:
        return response

    decoded: Final = decode_video_id_with_provider(response.id)
    hidden_params: Final = response.hidden_params
    routed_deployment_id: Final = id_safe_name(hidden_params.get("model_id"))
    deployment_id: Final = routed_deployment_id or (polled_video or decoded).get("deployment_id")
    kept_requested_model: Final = (
        id_safe_name(requested_model) or decoded.get("requested_model") or (polled_video or {}).get("requested_model")
    )
    model_id: Final = decoded.get("model_id") or id_safe_name(fallback_model_id) or routed_deployment_id
    provider: Final = (
        id_safe_name(decoded.get("custom_llm_provider"))
        or id_safe_name(hidden_params.get("custom_llm_provider"))
        or id_safe_name(fallback_provider)
    )
    if not model_id or not provider:
        return response
    if (
        model_id == decoded.get("model_id")
        and deployment_id == decoded.get("deployment_id")
        and kept_requested_model == decoded.get("requested_model")
    ):
        return response

    restamped_id: Final = encode_video_id_with_provider(
        video_id=decoded.get("video_id") or response.id,
        provider=provider,
        model_id=model_id,
        served_model=decoded.get("served_model"),
        deployment_id=deployment_id,
        requested_model=kept_requested_model,
    )
    return response.model_copy(update={"id": restamped_id})
