"""
Utils used for litellm.ahealth_check()
"""

#: litellm params that ``_update_litellm_params_for_health_check`` injects for
#: chat-completion health checks but that are invalid (or rejected) on every
#: other handler in :class:`HealthCheckHelpers.get_mode_handlers` — image and
#: video generation, embeddings, audio speech / transcription, rerank, ocr,
#: responses, batch, etc. ``messages`` is always chat-only; ``max_tokens`` is
#: chat/completion-only and is rejected by strict providers (e.g. OpenAI's
#: image-generation endpoints return 400 ``Unknown parameter: 'max_tokens'``).
_NON_CHAT_HEALTH_CHECK_STRIP_KEYS = {"messages", "max_tokens"}


def _filter_model_params(model_params: dict) -> dict:
    """Strip chat-only params before invoking a non-chat health check handler.

    ``litellm.acompletion`` is the only mode handler that consumes
    ``model_params`` unfiltered; every other handler routes through this
    helper, so removing chat-completion-only keys here keeps strict providers
    (OpenAI image generation, etc.) from rejecting the request.
    """
    return {
        k: v
        for k, v in model_params.items()
        if k not in _NON_CHAT_HEALTH_CHECK_STRIP_KEYS
    }


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
