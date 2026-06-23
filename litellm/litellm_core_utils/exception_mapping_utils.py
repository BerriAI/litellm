import json
import re
import traceback
from typing import Any, Optional, Protocol, cast

import httpx

import litellm
from litellm._logging import _ENABLE_SECRET_REDACTION, _redact_string, verbose_logger
from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.types.utils import LlmProviders

from ..exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
)


class ExceptionCheckers:
    """
    Helper class for checking various error conditions in exception strings.
    """

    @staticmethod
    def is_error_str_rate_limit(error_str: str) -> bool:
        """
        Check if an error string indicates a rate limit error.

        Args:
            error_str: The error string to check

        Returns:
            True if the error indicates a rate limit, False otherwise
        """
        if not isinstance(error_str, str):
            return False

        # Only treat 429 as a rate limit signal when it appears as a standalone token
        if re.search(r"\b429\b", error_str):
            return True

        _error_str_lower = error_str.lower()

        # Match "rate limit" (including variations like rate-limit / rate_limit)
        if re.search(r"rate[\s_\-]*limit", _error_str_lower):
            return True

        #######################################
        # Mistral API returns this error string
        #########################################
        if "service tier capacity exceeded" in _error_str_lower:
            return True

        return False

    @staticmethod
    def is_error_str_context_window_exceeded(error_str: str) -> bool:
        """
        Check if an error string indicates a context window exceeded error.
        """
        _error_str_lowercase = error_str.lower()
        # Exclude param validation errors (e.g. OpenAI "user" param max 64 chars)
        if "string_above_max_length" in _error_str_lowercase:
            return False
        if (
            "invalid 'user'" in _error_str_lowercase
            and "string too long" in _error_str_lowercase
        ):
            return False
        known_exception_substrings = [
            "exceed context limit",
            "this model's maximum context length is",
            "string too long. expected a string with maximum length",
            "model's maximum context limit",
            "is longer than the model's context length",
            "input tokens exceed the configured limit",
            "`inputs` tokens + `max_new_tokens` must be",
            "exceeds the available context size",  # llama.cpp/Lemonade
            "exceeds the maximum number of tokens allowed",  # Gemini
        ]
        for substring in known_exception_substrings:
            if substring in _error_str_lowercase:
                return True

        # Cerebras pattern: "Current length is X while limit is Y"
        if (
            "current length is" in _error_str_lowercase
            and "while limit is" in _error_str_lowercase
        ):
            return True

        return False

    @staticmethod
    def is_azure_content_policy_violation_error(error_str: str) -> bool:
        """
        Check if an error string indicates a content policy violation error.
        """
        _lower = error_str.lower()
        known_exception_substrings = [
            "content_policy_violation",
            "responsibleaipolicyviolation",
            "the response was filtered due to the prompt triggering azure openai's content management",
            "your task failed as a result of our safety system",
            "the model produced invalid content",
            "content_filter_policy",
            "your request was rejected as a result of our safety system",
        ]
        for substring in known_exception_substrings:
            if substring in _lower:
                return True
        return False


def get_error_message(error_obj) -> Optional[str]:
    """
    OpenAI Returns Error message that is nested, this extract the message

    Example:
    {
        'request': "<Request('POST', 'https://api.openai.com/v1/chat/completions')>",
        'message': "Error code: 400 - {\'error\': {\'message\': \"Invalid 'temperature': decimal above maximum value. Expected a value <= 2, but got 200 instead.\", 'type': 'invalid_request_error', 'param': 'temperature', 'code': 'decimal_above_max_value'}}",
        'body': {
            'message': "Invalid 'temperature': decimal above maximum value. Expected a value <= 2, but got 200 instead.",
            'type': 'invalid_request_error',
            'param': 'temperature',
            'code': 'decimal_above_max_value'
        },
        'code': 'decimal_above_max_value',
        'param': 'temperature',
        'type': 'invalid_request_error',
        'response': "<Response [400 Bad Request]>",
        'status_code': 400,
        'request_id': 'req_f287898caa6364cd42bc01355f74dd2a'
    }
    """
    try:
        # First, try to access the message directly from the 'body' key
        if error_obj is None:
            return None

        if hasattr(error_obj, "body"):
            _error_obj_body = getattr(error_obj, "body")
            if isinstance(_error_obj_body, dict):
                # OpenAI-style: {"message": "...", "type": "...", ...}
                if _error_obj_body.get("message"):
                    return _error_obj_body.get("message")

                # Azure-style: {"error": {"message": "...", ...}}
                nested_error = _error_obj_body.get("error")
                if isinstance(nested_error, dict):
                    return nested_error.get("message")

        # If all else fails, return None
        return None
    except Exception:
        return None


####### EXCEPTION MAPPING ################
def _get_body_error_code(error_str: str) -> int | None:
    """Return error.code from a JSON error body, or None if not parseable."""
    try:
        body = json.loads(error_str)
        code = body.get("error", {}).get("code")
        return int(code) if code is not None else None
    except Exception:
        return None


def _get_response_headers(original_exception: Exception) -> Optional[httpx.Headers]:
    """
    Extract and return the response headers from an exception, if present.

    Used for accurate retry logic.
    """
    _response_headers: Optional[httpx.Headers] = None
    try:
        _response_headers = getattr(original_exception, "headers", None)
        error_response = getattr(original_exception, "response", None)
        if not _response_headers and error_response:
            _response_headers = getattr(error_response, "headers", None)
        if not _response_headers:
            _response_headers = getattr(
                original_exception, "litellm_response_headers", None
            )
    except Exception:
        return None

    return _response_headers


def extract_and_raise_litellm_exception(
    response: Optional[Any],
    error_str: str,
    model: str,
    custom_llm_provider: str,
):
    """
    Covers scenario where litellm sdk calling proxy.

    Enables raising the special errors raised by litellm, eg. ContextWindowExceededError.

    Relevant Issue: https://github.com/BerriAI/litellm/issues/7259
    """
    pattern = r"litellm\.\w+Error"

    # Search for the exception in the error string
    match = re.search(pattern, error_str)

    # Extract the exception if found
    if match:
        exception_name = match.group(0)
        exception_name = exception_name.strip().replace("litellm.", "")
        raised_exception_obj = getattr(litellm, exception_name, None)
        if raised_exception_obj:
            # Try with response parameter first, fall back to without it
            # Some exceptions (e.g., APIConnectionError) don't accept response param
            try:
                raise raised_exception_obj(
                    message=error_str,
                    llm_provider=custom_llm_provider,
                    model=model,
                    response=response,
                )
            except TypeError:
                # Exception doesn't accept response parameter
                raise raised_exception_obj(
                    message=error_str,
                    llm_provider=custom_llm_provider,
                    model=model,
                )


def exception_type(  # type: ignore
    model,
    original_exception,
    custom_llm_provider,
    completion_kwargs={},
    extra_kwargs={},
):
    """Maps an LLM Provider Exception to OpenAI Exception Format"""
    if any(
        isinstance(original_exception, exc_type)
        for exc_type in litellm.LITELLM_EXCEPTION_TYPES
    ):
        return original_exception
    exception_mapping_worked = False
    exception_provider = custom_llm_provider
    mappable_exception: _ProviderHTTPException = cast(
        "_ProviderHTTPException", original_exception
    )
    if litellm.suppress_debug_info is False:
        print()  # noqa: T201
        print(  # noqa: T201
            "\033[1;31mGive Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new\033[0m"
        )
        print(  # noqa: T201
            "LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'."
        )
        print()  # noqa: T201

    litellm_response_headers = _get_response_headers(
        original_exception=original_exception
    )
    try:
        error_str = (
            redact_string(str(original_exception))
            if _ENABLE_SECRET_REDACTION
            else str(original_exception)
        )
        if model:
            if hasattr(original_exception, "message"):
                error_str = (
                    redact_string(str(original_exception.message))
                    if _ENABLE_SECRET_REDACTION
                    else str(original_exception.message)
                )
            if isinstance(original_exception, BaseException):
                exception_type = type(original_exception).__name__
            else:
                exception_type = ""

            ################################################################################
            # Common Extra information needed for all providers
            # We pass num retries, api_base, vertex_deployment etc to the exception here
            ################################################################################
            extra_information = ""
            try:
                _api_base = litellm.get_api_base(
                    model=model, optional_params=extra_kwargs
                )
                messages = litellm.get_first_chars_messages(kwargs=completion_kwargs)
                _vertex_project = extra_kwargs.get("vertex_project")
                _vertex_location = extra_kwargs.get("vertex_location")
                _metadata = extra_kwargs.get("metadata", {}) or {}
                _model_group = _metadata.get("model_group")
                _deployment = _metadata.get("deployment")
                extra_information = f"\nModel: {model}"

                if (
                    isinstance(custom_llm_provider, str)
                    and len(custom_llm_provider) > 0
                ):
                    exception_provider = (
                        custom_llm_provider[0].upper()
                        + custom_llm_provider[1:]
                        + "Exception"
                    )

                if _api_base:
                    extra_information += f"\nAPI Base: `{_api_base}`"
                if (
                    messages
                    and len(messages) > 0
                    and litellm.redact_messages_in_exceptions is False
                ):
                    extra_information += f"\nMessages: `{messages}`"

                if _model_group is not None:
                    extra_information += f"\nmodel_group: `{_model_group}`\n"
                if _deployment is not None:
                    extra_information += f"\ndeployment: `{_deployment}`\n"
                if _vertex_project is not None:
                    extra_information += f"\nvertex_project: `{_vertex_project}`\n"
                if _vertex_location is not None:
                    extra_information += f"\nvertex_location: `{_vertex_location}`\n"

                # on litellm proxy add key name + team to exceptions
                extra_information = _add_key_name_and_team_to_alert(
                    request_info=extra_information, metadata=_metadata
                )
            except Exception:
                # DO NOT LET this Block raising the original exception
                pass

            ################################################################################
            # End of Common Extra information Needed for all providers
            ################################################################################

            ################################################################################
            #################### Start of Provider Exception mapping ####################
            ################################################################################

            if (
                "Request Timeout Error" in error_str
                or "Request timed out" in error_str
                or "Timed out generating response" in error_str
                or "The read operation timed out" in error_str
            ):
                exception_mapping_worked = True

                raise Timeout(
                    message=f"APITimeoutError - Request timed out. Error_str: {error_str}",
                    model=model,
                    llm_provider=custom_llm_provider,
                    litellm_debug_info=extra_information,
                )

            if (
                custom_llm_provider == "litellm_proxy"
            ):  # handle special case where calling litellm proxy + exception str contains error message
                extract_and_raise_litellm_exception(
                    response=getattr(original_exception, "response", None),
                    error_str=error_str,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )
            if (
                custom_llm_provider == "openai"
                or custom_llm_provider == "text-completion-openai"
                or custom_llm_provider == "custom_openai"
                or custom_llm_provider in litellm.openai_compatible_providers
                or custom_llm_provider == "mistral"
            ):
                _map_openai_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif (
                custom_llm_provider == "anthropic"
                or custom_llm_provider == "anthropic_text"
            ):  # one of the anthropics
                _map_anthropic_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif custom_llm_provider == "replicate":
                _map_replicate_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif custom_llm_provider in litellm._openai_like_providers:
                _map_openai_like_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif custom_llm_provider == "bedrock":
                _map_bedrock_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif (
                custom_llm_provider == "sagemaker"
                or custom_llm_provider == "sagemaker_chat"
            ):
                _map_sagemaker_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif (
                custom_llm_provider == LlmProviders.VERTEX_AI
                or custom_llm_provider == LlmProviders.VERTEX_AI_BETA
                or custom_llm_provider == LlmProviders.GEMINI
            ):
                if (
                    "Vertex AI API has not been used in project" in error_str
                    or "Unable to find your project" in error_str
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"litellm.BadRequestError: {custom_llm_provider}Exception - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        response=httpx.Response(
                            status_code=400,
                            request=httpx.Request(
                                method="POST",
                                url=" https://cloud.google.com/vertex-ai/",
                            ),
                        ),
                        litellm_debug_info=extra_information,
                    )
                if "400 Request payload size exceeds" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"{custom_llm_provider.capitalize()}Exception - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                    )
                elif ExceptionCheckers.is_error_str_context_window_exceeded(error_str):
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"ContextWindowExceededError: {custom_llm_provider.capitalize()}Exception - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        litellm_debug_info=extra_information,
                    )
                elif (
                    "None Unknown Error." in error_str
                    or "Content has no parts." in error_str
                ):
                    exception_mapping_worked = True
                    raise litellm.InternalServerError(
                        message=f"litellm.InternalServerError: {custom_llm_provider}Exception - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        response=httpx.Response(
                            status_code=500,
                            content=str(original_exception),
                            request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                        ),
                        litellm_debug_info=extra_information,
                    )
                elif "API key not valid." in error_str:
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"{custom_llm_provider.capitalize()}Exception - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        litellm_debug_info=extra_information,
                    )
                elif "403" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"{custom_llm_provider.capitalize()}Exception BadRequestError - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        response=httpx.Response(
                            status_code=403,
                            request=httpx.Request(
                                method="POST",
                                url=" https://cloud.google.com/vertex-ai/",
                            ),
                        ),
                        litellm_debug_info=extra_information,
                    )
                elif (
                    "The response was blocked." in error_str
                    or "Output blocked by content filtering policy"
                    in error_str  # anthropic on vertex ai
                ):
                    exception_mapping_worked = True
                    raise ContentPolicyViolationError(
                        message=f"{custom_llm_provider.capitalize()}Exception ContentPolicyViolationError - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        litellm_debug_info=extra_information,
                        response=httpx.Response(
                            status_code=400,
                            request=httpx.Request(
                                method="POST",
                                url=" https://cloud.google.com/vertex-ai/",
                            ),
                        ),
                    )
                elif (
                    "429 Quota exceeded" in error_str
                    or "Quota exceeded for" in error_str
                    or "Resource exhausted" in error_str
                    or "IndexError: list index out of range" in error_str
                    or "429 Unable to submit request because the service is temporarily out of capacity."
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"litellm.RateLimitError: {custom_llm_provider}Exception - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        litellm_debug_info=extra_information,
                        response=httpx.Response(
                            status_code=429,
                            request=httpx.Request(
                                method="POST",
                                url=" https://cloud.google.com/vertex-ai/",
                            ),
                        ),
                    )
                elif (
                    isinstance(getattr(original_exception, "status_code", None), int)
                    and 500 <= original_exception.status_code < 600
                    and _get_body_error_code(error_str) == 429
                ):
                    # upstream gateway wraps a 429 inside a 5xx envelope
                    # e.g. HTTP 500/503 with {"error":{"code":429,...}}.
                    # Scoped to 5xx so HTTP 400/401 with body code:429
                    # still maps to BadRequestError / AuthenticationError.
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"litellm.RateLimitError: {custom_llm_provider}Exception - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        litellm_debug_info=extra_information,
                        response=httpx.Response(
                            status_code=429,
                            request=httpx.Request(
                                method="POST",
                                url=" https://cloud.google.com/vertex-ai/",
                            ),
                        ),
                    )
                elif (
                    "500 Internal Server Error" in error_str
                    or "The model is overloaded." in error_str
                ):
                    exception_mapping_worked = True
                    raise litellm.InternalServerError(
                        message=f"litellm.InternalServerError: {custom_llm_provider}Exception - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        litellm_debug_info=extra_information,
                    )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"{custom_llm_provider.capitalize()}Exception BadRequestError - {error_str}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                            response=httpx.Response(
                                status_code=400,
                                request=httpx.Request(
                                    method="POST",
                                    url="https://cloud.google.com/vertex-ai/",
                                ),
                            ),
                        )
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"{custom_llm_provider.capitalize()}Exception - {error_str}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )
                    if original_exception.status_code == 403:
                        exception_mapping_worked = True
                        raise PermissionDeniedError(
                            message=f"{custom_llm_provider.capitalize()}Exception - {error_str}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            response=httpx.Response(
                                status_code=403,
                                request=httpx.Request(
                                    method="POST",
                                    url="https://cloud.google.com/vertex-ai/",
                                ),
                            ),
                        )
                    if original_exception.status_code == 404:
                        exception_mapping_worked = True
                        raise NotFoundError(
                            message=f"{custom_llm_provider.capitalize()}Exception - {error_str}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )
                    if original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"{custom_llm_provider.capitalize()}Exception - {error_str}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )

                    if original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"litellm.RateLimitError: {custom_llm_provider.capitalize()}Exception - {error_str}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                            response=httpx.Response(
                                status_code=429,
                                request=httpx.Request(
                                    method="POST",
                                    url=" https://cloud.google.com/vertex-ai/",
                                ),
                            ),
                        )
                    if original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise litellm.InternalServerError(
                            message=f"{custom_llm_provider.capitalize()}Exception InternalServerError - {error_str}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                            response=httpx.Response(
                                status_code=500,
                                content=str(original_exception),
                                request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                            ),
                        )
                    if original_exception.status_code == 502:
                        exception_mapping_worked = True
                        raise APIConnectionError(
                            message=f"{custom_llm_provider.capitalize()}Exception - {error_str}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )
                    if original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"{custom_llm_provider.capitalize()}Exception - {error_str}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )
            elif custom_llm_provider == "cloudflare":
                _map_cloudflare_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif (
                custom_llm_provider == "cohere" or custom_llm_provider == "cohere_chat"
            ):  # Cohere
                _map_cohere_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif custom_llm_provider == "huggingface":
                _map_huggingface_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif custom_llm_provider == "ai21":
                _map_ai21_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif custom_llm_provider == "nlp_cloud":
                _map_nlp_cloud_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif custom_llm_provider == "together_ai":
                _map_together_ai_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif custom_llm_provider == "aleph_alpha":
                _map_aleph_alpha_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif (
                custom_llm_provider == "ollama" or custom_llm_provider == "ollama_chat"
            ):
                _map_ollama_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif custom_llm_provider == "vllm":
                _map_vllm_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            elif custom_llm_provider == "azure" or custom_llm_provider == "azure_text":
                _map_azure_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
            if custom_llm_provider == "openrouter":
                _map_openrouter_exception(
                    model=model,
                    original_exception=mappable_exception,
                    custom_llm_provider=custom_llm_provider,
                    error_str=error_str,
                    exception_type=exception_type,
                    exception_provider=exception_provider,
                    extra_information=extra_information,
                )
        if (
            "BadRequestError.__init__() missing 1 required positional argument: 'param'"
            in str(original_exception)
        ):  # deal with edge-case invalid request error bug in openai-python sdk
            exception_mapping_worked = True
            raise BadRequestError(
                message=f"{exception_provider} BadRequestError : This can happen due to missing AZURE_API_VERSION: {str(original_exception)}",
                model=model,
                llm_provider=custom_llm_provider,
                response=getattr(original_exception, "response", None),
            )
        else:  # ensure generic errors always return APIConnectionError=
            """
            For unmapped exceptions - raise the exception with traceback - https://github.com/BerriAI/litellm/issues/4201
            """
            exception_mapping_worked = True
            if hasattr(original_exception, "request"):
                raise APIConnectionError(
                    message="{} - {}".format(exception_provider, error_str),
                    llm_provider=custom_llm_provider,
                    model=model,
                    request=getattr(original_exception, "request", None),
                )
            else:
                raise APIConnectionError(
                    message="{}\n{}".format(
                        str(original_exception),
                        _redact_string(traceback.format_exc()),
                    ),
                    llm_provider=custom_llm_provider,
                    model=model,
                    request=httpx.Request(
                        method="POST", url="https://api.openai.com/v1/"
                    ),  # stub the request
                )
    except Exception as e:
        # LOGGING
        exception_logging(
            logger_fn=None,
            additional_args={
                "exception_mapping_worked": exception_mapping_worked,
                "original_exception": original_exception,
            },
            exception=e,
        )

        # don't let an error with mapping interrupt the user from receiving an error from the llm api calls
        if exception_mapping_worked:
            setattr(e, "litellm_response_headers", litellm_response_headers)
            raise e
        else:
            for error_type in litellm.LITELLM_EXCEPTION_TYPES:
                if isinstance(e, error_type):
                    setattr(e, "litellm_response_headers", litellm_response_headers)
                    raise e  # it's already mapped
            raised_exc = APIConnectionError(
                message="{}\n{}".format(
                    original_exception,
                    _redact_string(traceback.format_exc()),
                ),
                llm_provider="",
                model="",
            )
            setattr(raised_exc, "litellm_response_headers", litellm_response_headers)
            raise raised_exc


####### LOGGING ###################


def exception_logging(
    additional_args={},
    logger_fn=None,
    exception=None,
):
    try:
        model_call_details = {}
        if exception:
            model_call_details["exception"] = exception
        model_call_details["additional_args"] = additional_args
        # User Logging -> if you pass in a custom logging function or want to use sentry breadcrumbs
        verbose_logger.debug(
            f"Logging Details: logger_fn - {logger_fn} | callable(logger_fn) - {callable(logger_fn)}"
        )
        if logger_fn and callable(logger_fn):
            try:
                logger_fn(
                    model_call_details
                )  # Expectation: any logger function passed in by the user should accept a dict object
            except Exception:
                verbose_logger.debug(
                    f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
                )
    except Exception:
        verbose_logger.debug(
            f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
        )
        pass


def _add_key_name_and_team_to_alert(request_info: str, metadata: dict) -> str:
    """
    Internal helper function for litellm proxy
    Add the Key Name + Team Name to the error
    Only gets added if the metadata contains the user_api_key_alias and user_api_key_team_alias

    [Non-Blocking helper function]
    """
    try:
        _api_key_name = metadata.get("user_api_key_alias", None)
        _user_api_key_team_alias = metadata.get("user_api_key_team_alias", None)
        if _api_key_name is not None:
            request_info = (
                f"\n\nKey Name: `{_api_key_name}`\nTeam: `{_user_api_key_team_alias}`"
                + request_info
            )

        return request_info
    except Exception:
        return request_info
