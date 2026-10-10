from collections.abc import Mapping
from typing import Any, Final

import orjson
from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.router import Router
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    encode_video_id_with_provider,
)

_STR_MAPPING: Final = TypeAdapter(Mapping[str, object])


def extract_model_from_target_model_names(target_model_names: Any) -> str | None:
    if isinstance(target_model_names, str):
        target_model_names = [m.strip() for m in target_model_names.split(",") if m.strip()]
    elif not isinstance(target_model_names, list):
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


def _as_str_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _STR_MAPPING.validate_python(value)
    except ValidationError:
        return None


def _str_value(container: object, key: str) -> str | None:
    value: Final = (_as_str_mapping(container) or {}).get(key)
    return value if isinstance(value, str) else None


def _hidden_param(response: object, key: str) -> str | None:
    return _str_value(getattr(response, "_hidden_params", None), key)


def deployment_id_for_encoding(
    response: object, data: Mapping[str, Any], pinned_deployment_id: str | None = None
) -> str | None:
    """The deployment the request actually routed to, not the public model group.

    The router leaves ``model_id`` out of ``_hidden_params`` on the generic path and
    records the selected deployment under ``litellm_metadata.model_info.id`` instead.
    Encoding ``data["model"]`` there would embed the group, so a later status or
    content call could load-balance to a different deployment and fail to resolve.
    """
    litellm_metadata: Final = _as_str_mapping(data.get("litellm_metadata")) or {}
    routed_id: Final = _str_value(litellm_metadata.get("model_info"), "id")
    return _hidden_param(response, "model_id") or routed_id or pinned_deployment_id or data.get("model")


def route_to_encoded_deployment(
    llm_router: Router,
    model_id: str,
    data: dict[str, Any],  # mutable-ok: sets the model group in the request body in place
) -> str | None:
    """Route by the model group and return the deployment id to pin, if the id encodes one.

    ``data["model"]`` stays the group so guardrails, limits and budgets keyed on it apply.
    The router's own access-group and team filters run first, then the pin narrows the result.
    """
    resolved_model: Final = llm_router.resolve_model_name_from_model_id(model_id)
    if resolved_model:
        data["model"] = resolved_model  # rebind-ok: the router routes by the model group in the request body
    if model_id in llm_router.model_names or not llm_router.has_model_id(model_id):
        return None
    return model_id


def video_id_for_provider(llm_router: Router, video_id: str) -> str:
    decoded: Final = decode_video_id_with_provider(video_id)
    provider: Final = decoded.get("custom_llm_provider")
    deployment: Final = llm_router.get_deployment(model_id=decoded.get("model_id") or "")
    if provider is None or deployment is None:
        return video_id
    provider_model: Final = get_llm_provider(
        model=deployment.litellm_params.model,
        custom_llm_provider=deployment.litellm_params.custom_llm_provider,
    )[0]
    return encode_video_id_with_provider(
        video_id=decoded.get("video_id", ""), provider=provider, model_id=provider_model
    )


def encode_video_id_in_response(response: object, fallback_model: str | None) -> object:
    hidden_provider: Final = _hidden_param(response, "custom_llm_provider")
    hidden_model_id: Final = _hidden_param(response, "model_id")
    video_id: Final = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
    if not isinstance(video_id, str) or not video_id:
        return response

    decoded: Final = decode_video_id_with_provider(video_id)
    encoded: Final = encode_video_id_with_provider(
        video_id=decoded.get("video_id", ""),
        provider=hidden_provider or decoded.get("custom_llm_provider") or "openai",
        model_id=hidden_model_id or fallback_model or decoded.get("model_id"),
    )
    if isinstance(response, dict):
        response["id"] = encoded  # rebind-ok: in-place id rewrite, matching encode_character_id_in_response
        return response

    response.id = encoded  # pyright: ignore[reportAttributeAccessIssue]  # rebind-ok: in-place id rewrite on a non-dict response, matching encode_character_id_in_response
    return response


def encode_character_id_in_response(response: Any, custom_llm_provider: str, model_id: str | None) -> Any:
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
