"""
Utils used for litellm.ahealth_check()
"""

#: ``messages`` is injected by ``_update_litellm_params_for_health_check`` for
#: chat-style health checks; every non-chat handler in
#: :class:`HealthCheckHelpers.get_mode_handlers` rejects it, and
#: ``litellm.atext_completion`` takes ``prompt`` instead.
_COMPLETION_HEALTH_CHECK_STRIP_KEYS = {"messages"}

#: Strict non-chat handlers (OpenAI image generation, embeddings, rerank, etc.)
#: also reject ``max_tokens`` — e.g. ``dall-e-3`` returns
#: ``400 Unknown parameter: 'max_tokens'``. ``atext_completion`` does accept
#: ``max_tokens``, so the ``completion`` mode keeps it (preserves the
#: ``BACKGROUND_HEALTH_CHECK_MAX_TOKENS`` cost cap).
_NON_CHAT_HEALTH_CHECK_STRIP_KEYS = _COMPLETION_HEALTH_CHECK_STRIP_KEYS | {"max_tokens"}


def _filter_model_params(model_params: dict, *, keep_max_tokens: bool = False) -> dict:
    """Strip chat-only params before invoking a non-chat health check handler.

    ``litellm.acompletion`` is the only mode handler that consumes
    ``model_params`` unfiltered; every other handler routes through this
    helper. ``litellm.atext_completion`` (the ``completion`` mode) accepts
    ``max_tokens`` and should pass ``keep_max_tokens=True`` to preserve the
    cost-control cap.
    """
    strip_keys = (
        _COMPLETION_HEALTH_CHECK_STRIP_KEYS
        if keep_max_tokens
        else _NON_CHAT_HEALTH_CHECK_STRIP_KEYS
    )
    return {k: v for k, v in model_params.items() if k not in strip_keys}


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
