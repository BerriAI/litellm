from typing import TYPE_CHECKING, Any, Final

import orjson

from litellm.types.videos.main import DecodedVideoId
from litellm.types.videos.utils import encode_character_id_with_provider

if TYPE_CHECKING:
    from litellm.router import Router


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


def resolve_video_model_name(llm_router: "Router", decoded: DecodedVideoId) -> str | None:
    model_name: Final = llm_router.resolve_model_name_from_model_id(decoded.get("model_id"))
    served_model: Final = decoded.get("served_model")
    if model_name is None or served_model is None:
        return model_name
    return llm_router.resolve_served_model_name(model_name, served_model)
