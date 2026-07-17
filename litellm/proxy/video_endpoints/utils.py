from typing import Any, Dict, Optional

import orjson

from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    encode_video_id_with_provider,
)


def extract_model_from_target_model_names(target_model_names: Any) -> Optional[str]:
    if isinstance(target_model_names, str):
        target_model_names = [m.strip() for m in target_model_names.split(",") if m.strip()]
    elif not isinstance(target_model_names, list):
        return None
    return target_model_names[0] if target_model_names else None


def get_custom_provider_from_data(data: Dict[str, Any]) -> Optional[str]:
    custom_llm_provider = data.get("custom_llm_provider")
    if custom_llm_provider:
        return custom_llm_provider

    extra_body = data.get("extra_body")
    if isinstance(extra_body, str):
        try:
            parsed_extra_body = orjson.loads(extra_body)
            if isinstance(parsed_extra_body, dict):
                extra_body = parsed_extra_body
        except Exception:
            extra_body = None

    if isinstance(extra_body, dict):
        extra_body_custom_llm_provider = extra_body.get("custom_llm_provider")
        if isinstance(extra_body_custom_llm_provider, str):
            return extra_body_custom_llm_provider

    return None


def encode_character_id_in_response(response: Any, custom_llm_provider: str, model_id: Optional[str]) -> Any:
    if isinstance(response, dict) and response.get("id"):
        response["id"] = encode_character_id_with_provider(
            character_id=response["id"],
            provider=custom_llm_provider,
            model_id=model_id,
        )
        return response

    character_id = getattr(response, "id", None)
    if isinstance(character_id, str) and character_id:
        response.id = encode_character_id_with_provider(
            character_id=character_id,
            provider=custom_llm_provider,
            model_id=model_id,
        )
    return response


def reencode_video_id_with_model_id(response: Any, custom_llm_provider: str | None, model_id: str | None) -> Any:
    """
    Re-encode a returned video id so it carries the router-selected deployment id.

    The provider transformation layer encodes the client-facing model/group name
    into the video id because it runs before the router attaches the deployment id
    to ``_hidden_params``. That is enough for single-deployment groups, but for a
    group backed by several deployments the status/content round-trip decodes the
    group name and re-routes through load balancing instead of pinning to the
    deployment that created the job. Preferring ``model_id`` (the deployment id)
    here keeps the follow-up calls on the deployment that owns the video
    """
    if not model_id:
        return response

    if isinstance(response, dict):
        current_id = response.get("id")
    else:
        current_id = getattr(response, "id", None)

    if not isinstance(current_id, str) or not current_id:
        return response

    decoded = decode_video_id_with_provider(current_id)
    if decoded.get("model_id") == model_id:
        return response

    provider = decoded.get("custom_llm_provider") or custom_llm_provider
    if not provider:
        return response

    raw_video_id = decoded.get("video_id") or current_id
    new_id = encode_video_id_with_provider(
        video_id=raw_video_id,
        provider=provider,
        model_id=model_id,
    )

    if isinstance(response, dict):
        response["id"] = new_id
    else:
        response.id = new_id
    return response
