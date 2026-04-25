"""
Utils used for litellm.ahealth_check()
"""

from typing import Iterable, Optional


def _filter_model_params(
    model_params: dict, additional_keys_to_remove: Optional[Iterable[str]] = None
) -> dict:
    """Remove 'messages' (and any additional caller-supplied keys) from model params.

    `_update_litellm_params_for_health_check` injects chat-completion-only fields
    (`messages`, `max_tokens`) into every deployment's params before dispatch. Most
    handlers only need `messages` stripped, but non-chat handlers (image/video
    generation, embedding, transcription, etc.) need `max_tokens` stripped too —
    OpenAI's image endpoints reject it with 400, breaking health checks for
    `dall-e-*` / `gpt-image-1`.
    """
    keys_to_remove = {"messages"}
    if additional_keys_to_remove:
        keys_to_remove.update(additional_keys_to_remove)
    return {k: v for k, v in model_params.items() if k not in keys_to_remove}


def _create_health_check_response(response_headers: dict) -> dict:
    response = {}

    if (
        response_headers.get("x-ratelimit-remaining-requests", None) is not None
    ):  # not provided for dall-e requests
        response["x-ratelimit-remaining-requests"] = response_headers[
            "x-ratelimit-remaining-requests"
        ]

    if response_headers.get("x-ratelimit-remaining-tokens", None) is not None:
        response["x-ratelimit-remaining-tokens"] = response_headers[
            "x-ratelimit-remaining-tokens"
        ]

    if response_headers.get("x-ms-region", None) is not None:
        response["x-ms-region"] = response_headers["x-ms-region"]
    return response
