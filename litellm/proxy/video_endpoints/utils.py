from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

import orjson
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping, has_attributes
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import video_access_group
from litellm.proxy.openai_files_endpoints.common_utils import (
    authorize_model_for_key,
    first_deployment_id_in_group,
    video_deployment_the_caller_may_use,
)
from litellm.types.videos.main import DecodedVideoId, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    encode_video_id_with_provider,
    id_safe_name,
)

if TYPE_CHECKING:
    from litellm.router import Router

_DECODED_JSON: Final = TypeAdapter(object)


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
        parsed_ref: Final = _DECODED_JSON.validate_python(orjson.loads(video_ref))
    except orjson.JSONDecodeError:
        return video_ref
    return parsed_ref.get("id", "") if isinstance(parsed_ref, dict) else video_ref


def get_custom_provider_from_data(data: Mapping[str, object]) -> object:
    custom_llm_provider: Final = data.get("custom_llm_provider")
    if custom_llm_provider:
        return custom_llm_provider

    extra_body = data.get("extra_body")
    if isinstance(extra_body, str):
        try:
            parsed_extra_body: Final = _DECODED_JSON.validate_python(orjson.loads(extra_body))
            if isinstance(parsed_extra_body, dict):
                extra_body = parsed_extra_body
        except Exception:
            extra_body = None

    if isinstance(extra_body, dict):
        extra_body_custom_llm_provider: Final = extra_body.get("custom_llm_provider")
        if isinstance(extra_body_custom_llm_provider, str):
            return extra_body_custom_llm_provider

    return None


class _CarriesId(Protocol):
    id: object


def encode_character_id_in_response(response: object, custom_llm_provider: object, model_id: str | None) -> object:
    if isinstance(response, dict) and response.get("id"):
        response["id"] = encode_character_id_with_provider(
            character_id=response["id"],
            provider=custom_llm_provider,
            model_id=model_id,
        )
        return response

    if not has_attributes(response, _CarriesId, "id"):
        return response
    character_id: Final = response.id
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
    return name if llm_router.registry().resolve(name, kind="from_id", custom_llm_provider=provider) else None


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
    registry: Final = llm_router.registry()
    resolved: Final = registry.resolve(model_id, kind="from_id", custom_llm_provider=provider) if model_id else None
    deployment_id: Final = decoded.get("deployment_id")
    if not deployment_id or resolved in (model_id, f"{provider}/{model_id}"):
        return resolved
    deployment: Final = registry.deployment(deployment_id)
    if deployment is None:
        return resolved
    creating_group: Final = deployment.model_info.team_public_model_name or deployment.model_name
    if (
        resolved is not None
        and registry.resolve(resolved, kind="served", served_model=creating_group) == creating_group
    ):
        return resolved
    return creating_group


async def authorized_video_model(
    decoded: DecodedVideoId, llm_router: "Router | None", user_api_key_dict: UserAPIKeyAuth
) -> str | None:
    model: Final = video_model_group(decoded, llm_router)
    if model is None:
        return None
    await authorize_model_for_key(
        model_id=video_access_group(decoded, llm_router, resolved_model=model) or model,
        llm_router=llm_router,
        user_api_key_dict=user_api_key_dict,
    )
    return model


@dataclass(frozen=True, slots=True)
class VideoListTarget:
    model: str | None = None
    deployment_id: str | None = None


async def _served_list_group(
    requested_model: str,
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: "Router | None",
    router_settings: Mapping[str, object] | None,
) -> str:
    from litellm.proxy.common_request_processing import resolve_per_request_model_group_alias
    from litellm.proxy.litellm_pre_call_utils import request_served_model

    prepared: Final = request_served_model(requested_model, user_api_key_dict, None, llm_router)
    if llm_router is None:
        return prepared
    per_request: Final = await resolve_per_request_model_group_alias(
        prepared, router_settings or MappingProxyType({}), user_api_key_dict, llm_router
    )
    served: Final = per_request or prepared
    return llm_router.registry().resolve(served, kind="alias") or served


async def video_list_target(
    requested_model: str | None,
    custom_llm_provider: object,
    llm_router: "Router | None",
    user_api_key_dict: UserAPIKeyAuth,
    router_settings: Mapping[str, object] | None = None,
) -> VideoListTarget:
    provider: Final = custom_llm_provider if isinstance(custom_llm_provider, str) else None
    if requested_model:
        await authorize_model_for_key(
            model_id=requested_model, llm_router=llm_router, user_api_key_dict=user_api_key_dict
        )
        group: Final = await _served_list_group(requested_model, user_api_key_dict, llm_router, router_settings)
        pinned: Final = first_deployment_id_in_group(llm_router, group, user_api_key_dict, provider)
        if pinned is not None:
            return VideoListTarget(model=pinned, deployment_id=pinned)
        if provider is None:
            return VideoListTarget(model=requested_model)
    if provider is None:
        return VideoListTarget()
    deployment: Final = await video_deployment_the_caller_may_use(llm_router, user_api_key_dict, provider)
    return (
        VideoListTarget() if deployment is None else VideoListTarget(deployment.deployment_id, deployment.deployment_id)
    )


def _video_id_with_deployment(video_id: object, deployment_id: str) -> object:
    if not isinstance(video_id, str):
        return video_id
    decoded: Final = decode_video_id_with_provider(video_id)
    provider: Final = decoded.get("custom_llm_provider")
    original_video_id: Final = decoded.get("video_id")
    if not provider or not original_video_id or decoded.get("deployment_id"):
        return video_id
    return encode_video_id_with_provider(
        video_id=original_video_id,
        provider=provider,
        model_id=decoded.get("model_id"),
        served_model=decoded.get("served_model"),
        deployment_id=deployment_id,
        requested_model=decoded.get("requested_model"),
    )


def _listed_video_with_deployment(video: object, deployment_id: str) -> object:
    fields: Final = as_str_mapping(video)
    if fields is None or "id" not in fields:
        return video
    return {**fields, "id": _video_id_with_deployment(fields["id"], deployment_id)}


def stamp_listed_video_ids(listed: object, deployment_id: str | None) -> object:
    fields: Final = as_str_mapping(listed)
    if deployment_id is None or fields is None:
        return listed
    videos: Final = as_object_list(fields.get("data"))
    stamped_videos: Final[Mapping[str, object]] = (
        {} if videos is None else {"data": [_listed_video_with_deployment(video, deployment_id) for video in videos]}
    )
    stamped_cursors: Final = {
        cursor: _video_id_with_deployment(fields[cursor], deployment_id)
        for cursor in ("first_id", "last_id")
        if cursor in fields
    }
    return {**fields, **stamped_videos, **stamped_cursors}


def fallback_requested_model(requested_model: str | None, response: object, llm_router: "Router | None") -> str | None:
    """The group a create asked for when one of that group's router fallbacks served it."""
    from litellm.proxy.auth.auth_utils import router_fallback_model_names

    if requested_model is None or llm_router is None or not isinstance(response, VideoObject):
        return None
    deployment_id: Final = id_safe_name(response.hidden_params.get("model_id"))
    deployment: Final = llm_router.registry().deployment(deployment_id) if deployment_id else None
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
