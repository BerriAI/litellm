import json
import traceback
from typing import Any, Optional

import httpx

import litellm
from litellm._logging import verbose_logger

from ..exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
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
        
        if "429" in error_str or "rate limit" in error_str.lower():
            return True
        
        #######################################
        # Mistral API returns this error string
        #########################################
        if "service tier capacity exceeded" in error_str.lower():
            return True
        
        return False

    @staticmethod
    def is_error_str_context_window_exceeded(error_str: str) -> bool:
        """
        Check if an error string indicates a context window exceeded error.
        """
        _error_str_lowercase = error_str.lower()
        known_exception_substrings = [
            "exceed context limit",
            "this model's maximum context length is",
            "string too long. expected a string with maximum length",
            "model's maximum context limit",
            "is longer than the model's context length",
        ]
        for substring in known_exception_substrings:
            if substring in _error_str_lowercase:
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
                return _error_obj_body.get("message")

        # If all else fails, return None
        return None
    except Exception:
        return None


####### EXCEPTION MAPPING ################
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


import re


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
            raise raised_exception_obj(
                message=error_str,
                llm_provider=custom_llm_provider,
                model=model,
                response=response,
            )


def exception_type(  # type: ignore  # noqa: PLR0915
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
    if litellm.suppress_debug_info is False:
        print()  # noqa
        print(  # noqa
            "\033[1;31mGive Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new\033[0m"  # noqa
        )  # noqa
        print(  # noqa
            "LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'."  # noqa
        )  # noqa
        print()  # noqa

    litellm_response_headers = _get_response_headers(
        original_exception=original_exception
    )
    try:
        error_str = str(original_exception)
        if model:
            if hasattr(original_exception, "message"):
                error_str = str(original_exception.message)
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
                # custom_llm_provider is openai, make it OpenAI
                message = get_error_message(error_obj=original_exception)
                if message is None:
                    if hasattr(original_exception, "message"):
                        message = original_exception.message
                    else:
                        message = str(original_exception)

                if message is not None and isinstance(
                    message, str
                ):  # done to prevent user-confusion. Relevant issue - https://github.com/BerriAI/litellm/issues/1414
                    message = message.replace("OPENAI", custom_llm_provider.upper())
                    message = message.replace(
                        "openai.OpenAIError",
                        "{}.{}Error".format(custom_llm_provider, custom_llm_provider),
                    )
                if custom_llm_provider == "openai":
                    exception_provider = "OpenAI" + "Exception"
                else:
                    exception_provider = (
                        custom_llm_provider[0].upper()
                        + custom_llm_provider[1:]
                        + "Exception"
                    )

                if ExceptionCheckers.is_error_str_rate_limit(error_str):
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"RateLimitError: {exception_provider} - {message}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        response=getattr(original_exception, "response", None),
                    )
                elif ExceptionCheckers.is_error_str_context_window_exceeded(error_str):
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"ContextWindowExceededError: {exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=getattr(original_exception, "response", None),
                        litellm_debug_info=extra_information,
                    )
                elif (
                    "invalid_request_error" in error_str
                    and "model_not_found" in error_str
                ):
                    exception_mapping_worked = True
                    raise NotFoundError(
                        message=f"{exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=getattr(original_exception, "response", None),
                        litellm_debug_info=extra_information,
                    )
                elif "A timeout occurred" in error_str:
                    exception_mapping_worked = True
                    raise Timeout(
                        message=f"{exception_provider} - {message}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        litellm_debug_info=extra_information,
                    )
                elif (
                    (
                        "invalid_request_error" in error_str
                        and "content_policy_violation" in error_str
                    )
                    or (
                        "Invalid prompt" in error_str
                        and "violating our usage policy" in error_str
                    )
                    or (
                        "request was rejected as a result of the safety system"
                        in error_str.lower()
                    )
                ):
                    exception_mapping_worked = True
                    raise ContentPolicyViolationError(
                        message=f"ContentPolicyViolationError: {exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=getattr(original_exception, "response", None),
                        litellm_debug_info=extra_information,
                    )
                elif (
                    "invalid_request_error" in error_str
                    and "Incorrect API key provided" not in error_str
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"{exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=getattr(original_exception, "response", None),
                        litellm_debug_info=extra_information,
                        body=getattr(original_exception, "body", None),
                    )
                elif (
                    "Web server is returning an unknown error" in error_str
                    or "The server had an error processing your request." in error_str
                ):
                    exception_mapping_worked = True
                    raise litellm.InternalServerError(
                        message=f"{exception_provider} - {message}",
                        model=model,
                        llm_provider=custom_llm_provider,
                    )
                elif "Request too large" in error_str:
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"RateLimitError: {exception_provider} - {message}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        response=getattr(original_exception, "response", None),
                        litellm_debug_info=extra_information,
                    )
                elif (
                    "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"AuthenticationError: {exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=getattr(original_exception, "response", None),
                        litellm_debug_info=extra_information,
                    )
                elif "Mistral API raised a streaming error" in error_str:
                    exception_mapping_worked = True
                    _request = httpx.Request(
                        method="POST", url="https://api.openai.com/v1"
                    )
                    raise APIError(
                        status_code=500,
                        message=f"{exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        request=_request,
                        litellm_debug_info=extra_information,
                    )
                elif hasattr(original_exception, "status_code"):
                    exception_mapping_worked = True
                    if original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"{exception_provider} - {message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AuthenticationError: {exception_provider} - {message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 404:
                        exception_mapping_worked = True
                        raise NotFoundError(
                            message=f"NotFoundError: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"Timeout Error: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"{exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                            body=getattr(original_exception, "body", None),
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"RateLimitError: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise InternalServerError(
                            message=f"InternalServerError: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"ServiceUnavailableError: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 504:  # gateway timeout error
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"Timeout Error: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                            exception_status_code=original_exception.status_code,
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"APIError: {exception_provider} - {message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            request=getattr(original_exception, "request", None),
                            litellm_debug_info=extra_information,
                        )
                else:
                    # if no status code then it is an APIConnectionError: https://github.com/openai/openai-python#handling-errors
                    # exception_mapping_worked = True
                    raise APIConnectionError(
                        message=f"APIConnectionError: {exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        litellm_debug_info=extra_information,
                        request=httpx.Request(
                            method="POST", url="https://api.openai.com/v1/"
                        ),
                    )
            elif (
                custom_llm_provider == "anthropic"
                or custom_llm_provider == "anthropic_text"
            ):  # one of the anthropics
                if "prompt is too long" in error_str or "prompt: length" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message="AnthropicError - {}".format(error_str),
                        model=model,
                        llm_provider="anthropic",
                    )
                elif "overloaded_error" in error_str:
                    exception_mapping_worked = True
                    raise InternalServerError(
                        message="AnthropicError - {}".format(error_str),
                        model=model,
                        llm_provider="anthropic",
                    )
                if "Invalid API Key" in error_str:
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message="AnthropicError - {}".format(error_str),
                        model=model,
                        llm_provider="anthropic",
                    )
                if "content filtering policy" in error_str:
                    exception_mapping_worked = True
                    raise ContentPolicyViolationError(
                        message="AnthropicError - {}".format(error_str),
                        model=model,
                        llm_provider="anthropic",
                    )
                if "Client error '400 Bad Request'" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message="AnthropicError - {}".format(error_str),
                        model=model,
                        llm_provider="anthropic",
                    )
                if hasattr(original_exception, "status_code"):
                    verbose_logger.debug(
                        f"status_code: {original_exception.status_code}"
                    )
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AnthropicException - {error_str}",
                            llm_provider="anthropic",
                            model=model,
                        )
                    elif (
                        original_exception.status_code == 400
                        or original_exception.status_code == 413
                    ):
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AnthropicException - {error_str}",
                            model=model,
                            llm_provider="anthropic",
                        )
                    elif original_exception.status_code == 404:
                        exception_mapping_worked = True
                        raise NotFoundError(
                            message=f"AnthropicException - {error_str}",
                            model=model,
                            llm_provider="anthropic",
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"AnthropicException - {error_str}",
                            model=model,
                            llm_provider="anthropic",
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"AnthropicException - {error_str}",
                            llm_provider="anthropic",
                            model=model,
                        )
                    elif (
                        original_exception.status_code == 500
                        or original_exception.status_code == 529
                    ):
                        exception_mapping_worked = True
                        raise litellm.InternalServerError(
                            message=f"AnthropicException - {error_str}. Handle with `litellm.InternalServerError`.",
                            llm_provider="anthropic",
                            model=model,
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise litellm.ServiceUnavailableError(
                            message=f"AnthropicException - {error_str}. Handle with `litellm.ServiceUnavailableError`.",
                            llm_provider="anthropic",
                            model=model,
                        )
            elif custom_llm_provider == "replicate":
                if "Incorrect authentication token" in error_str:
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"ReplicateException - {error_str}",
                        llm_provider="replicate",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif "input is too long" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"ReplicateException - {error_str}",
                        model=model,
                        llm_provider="replicate",
                        response=getattr(original_exception, "response", None),
                    )
                elif exception_type == "ModelError":
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"ReplicateException - {error_str}",
                        model=model,
                        llm_provider="replicate",
                        response=getattr(original_exception, "response", None),
                    )
                elif "Request was throttled" in error_str:
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"ReplicateException - {error_str}",
                        llm_provider="replicate",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"ReplicateException - {original_exception.message}",
                            llm_provider="replicate",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif (
                        original_exception.status_code == 400
                        or original_exception.status_code == 413
                    ):
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"ReplicateException - {original_exception.message}",
                            model=model,
                            llm_provider="replicate",
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise UnprocessableEntityError(
                            message=f"ReplicateException - {original_exception.message}",
                            model=model,
                            llm_provider="replicate",
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"ReplicateException - {original_exception.message}",
                            model=model,
                            llm_provider="replicate",
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise UnprocessableEntityError(
                            message=f"ReplicateException - {original_exception.message}",
                            llm_provider="replicate",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"ReplicateException - {original_exception.message}",
                            llm_provider="replicate",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"ReplicateException - {original_exception.message}",
                            llm_provider="replicate",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                exception_mapping_worked = True
                raise APIError(
                    status_code=500,
                    message=f"ReplicateException - {str(original_exception)}",
                    llm_provider="replicate",
                    model=model,
                    request=httpx.Request(
                        method="POST",
                        url="https://api.replicate.com/v1/deployments",
                    ),
                )
            elif custom_llm_provider in litellm._openai_like_providers:
                if "authorization denied for" in error_str:
                    exception_mapping_worked = True

                    # Predibase returns the raw API Key in the response - this block ensures it's not returned in the exception
                    if (
                        error_str is not None
                        and isinstance(error_str, str)
                        and "bearer" in error_str.lower()
                    ):
                        # only keep the first 10 chars after the occurnence of "bearer"
                        _bearer_token_start_index = error_str.lower().find("bearer")
                        error_str = error_str[: _bearer_token_start_index + 14]
                        error_str += "XXXXXXX" + '"'

                    raise AuthenticationError(
                        message=f"{custom_llm_provider}Exception: Authentication Error - {error_str}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=getattr(original_exception, "response", None),
                        litellm_debug_info=extra_information,
                    )
                elif "model's maximum context limit" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"{custom_llm_provider}Exception: Context Window Error - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                    )
                elif "token_quota_reached" in error_str:
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"{custom_llm_provider}Exception: Rate Limit Errror - {error_str}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif (
                    "The server received an invalid response from an upstream server."
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise litellm.InternalServerError(
                        message=f"{custom_llm_provider}Exception - {original_exception.message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                    )
                elif "model_no_support_for_function" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"{custom_llm_provider}Exception - Use 'watsonx_text' route instead. IBM WatsonX does not support `/text/chat` endpoint. - {error_str}",
                        llm_provider=custom_llm_provider,
                        model=model,
                    )
                elif hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise litellm.InternalServerError(
                            message=f"{custom_llm_provider}Exception - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )
                    elif (
                        original_exception.status_code == 401
                        or original_exception.status_code == 403
                    ):
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"{custom_llm_provider}Exception - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"{custom_llm_provider}Exception - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )
                    elif original_exception.status_code == 404:
                        exception_mapping_worked = True
                        raise NotFoundError(
                            message=f"{custom_llm_provider}Exception - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"{custom_llm_provider}Exception - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif (
                        original_exception.status_code == 422
                        or original_exception.status_code == 424
                    ):
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"{custom_llm_provider}Exception - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"{custom_llm_provider}Exception - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"{custom_llm_provider}Exception - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 504:  # gateway timeout error
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"{custom_llm_provider}Exception - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                            exception_status_code=original_exception.status_code,
                        )
            elif custom_llm_provider == "bedrock":
                if (
                    "too many tokens" in error_str
                    or "expected maxLength:" in error_str
                    or "Input is too long" in error_str
                    or "prompt is too long" in error_str
                    or "prompt: length: 1.." in error_str
                    or "Too many input tokens" in error_str
                ):
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"BedrockException: Context Window Error - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                    )
                elif (
                    "Conversation blocks and tool result blocks cannot be provided in the same turn."
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"BedrockException - {error_str}\n. Enable 'litellm.modify_params=True' (for PROXY do: `litellm_settings::modify_params: True`) to insert a dummy assistant message and fix this error.",
                        model=model,
                        llm_provider="bedrock",
                        response=getattr(original_exception, "response", None),
                    )
                elif "Malformed input request" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"BedrockException - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                        response=getattr(original_exception, "response", None),
                    )
                elif "A conversation must start with a user message." in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"BedrockException - {error_str}\n. Pass in default user message via `completion(..,user_continue_message=)` or enable `litellm.modify_params=True`.\nFor Proxy: do via `litellm_settings::modify_params: True` or user_continue_message under `litellm_params`",
                        model=model,
                        llm_provider="bedrock",
                        response=getattr(original_exception, "response", None),
                    )
                elif (
                    "Unable to locate credentials" in error_str
                    or "The security token included in the request is invalid"
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"BedrockException Invalid Authentication - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                        response=getattr(original_exception, "response", None),
                    )
                elif "AccessDeniedException" in error_str:
                    exception_mapping_worked = True
                    raise PermissionDeniedError(
                        message=f"BedrockException PermissionDeniedError - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                        response=getattr(original_exception, "response", None),
                    )
                elif (
                    "throttlingException" in error_str
                    or "ThrottlingException" in error_str
                ):
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"BedrockException: Rate Limit Error - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                        response=getattr(original_exception, "response", None),
                    )
                elif (
                    "Connect timeout on endpoint URL" in error_str
                    or "timed out" in error_str
                ):
                    exception_mapping_worked = True
                    raise Timeout(
                        message=f"BedrockException: Timeout Error - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                    )
                elif "Could not process image" in error_str:
                    exception_mapping_worked = True
                    raise litellm.InternalServerError(
                        message=f"BedrockException - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                    )
                elif hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"BedrockException - {original_exception.message}",
                            llm_provider="bedrock",
                            model=model,
                            response=httpx.Response(
                                status_code=500,
                                request=httpx.Request(
                                    method="POST", url="https://api.openai.com/v1/"
                                ),
                            ),
                        )
                    elif original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"BedrockException - {original_exception.message}",
                            llm_provider="bedrock",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"BedrockException - {original_exception.message}",
                            llm_provider="bedrock",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 404:
                        exception_mapping_worked = True
                        raise NotFoundError(
                            message=f"BedrockException - {original_exception.message}",
                            llm_provider="bedrock",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"BedrockException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"BedrockException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"BedrockException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"BedrockException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 504:  # gateway timeout error
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"BedrockException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                            exception_status_code=original_exception.status_code,
                        )
            elif (
                custom_llm_provider == "sagemaker"
                or custom_llm_provider == "sagemaker_chat"
            ):
                if "Unable to locate credentials" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"litellm.BadRequestError: SagemakerException - {error_str}",
                        model=model,
                        llm_provider="sagemaker",
                        response=getattr(original_exception, "response", None),
                    )
                elif (
                    "Input validation error: `best_of` must be > 0 and <= 2"
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message="SagemakerException - the value of 'n' must be > 0 and <= 2 for sagemaker endpoints",
                        model=model,
                        llm_provider="sagemaker",
                        response=getattr(original_exception, "response", None),
                    )
                elif (
                    "`inputs` tokens + `max_new_tokens` must be <=" in error_str
                    or "instance type with more CPU capacity or memory" in error_str
                ):
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"SagemakerException - {error_str}",
                        model=model,
                        llm_provider="sagemaker",
                        response=getattr(original_exception, "response", None),
                    )
                elif hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"SagemakerException - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            response=httpx.Response(
                                status_code=500,
                                request=httpx.Request(
                                    method="POST", url="https://api.openai.com/v1/"
                                ),
                            ),
                        )
                    elif original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"SagemakerException - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"SagemakerException - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 404:
                        exception_mapping_worked = True
                        raise NotFoundError(
                            message=f"SagemakerException - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"SagemakerException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif (
                        original_exception.status_code == 422
                        or original_exception.status_code == 424
                    ):
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"SagemakerException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"SagemakerException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"SagemakerException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 504:  # gateway timeout error
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"SagemakerException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                            exception_status_code=original_exception.status_code,
                        )
            elif (
                custom_llm_provider == "vertex_ai"
                or custom_llm_provider == "vertex_ai_beta"
                or custom_llm_provider == "gemini"
            ):
                if (
                    "Vertex AI API has not been used in project" in error_str
                    or "Unable to find your project" in error_str
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"litellm.BadRequestError: VertexAIException - {error_str}",
                        model=model,
                        llm_provider="vertex_ai",
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
                        message=f"VertexException - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                    )
                elif (
                    "None Unknown Error." in error_str
                    or "Content has no parts." in error_str
                ):
                    exception_mapping_worked = True
                    raise litellm.InternalServerError(
                        message=f"litellm.InternalServerError: VertexAIException - {error_str}",
                        model=model,
                        llm_provider="vertex_ai",
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
                        message=f"{custom_llm_provider}Exception - {error_str}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        litellm_debug_info=extra_information,
                    )
                elif "403" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"VertexAIException BadRequestError - {error_str}",
                        model=model,
                        llm_provider="vertex_ai",
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
                        message=f"VertexAIException ContentPolicyViolationError - {error_str}",
                        model=model,
                        llm_provider="vertex_ai",
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
                    or "IndexError: list index out of range" in error_str
                    or "429 Unable to submit request because the service is temporarily out of capacity."
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"litellm.RateLimitError: VertexAIException - {error_str}",
                        model=model,
                        llm_provider="vertex_ai",
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
                        message=f"litellm.InternalServerError: VertexAIException - {error_str}",
                        model=model,
                        llm_provider="vertex_ai",
                        litellm_debug_info=extra_information,
                    )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"VertexAIException BadRequestError - {error_str}",
                            model=model,
                            llm_provider="vertex_ai",
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
                            message=f"VertexAIException - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )
                    if original_exception.status_code == 404:
                        exception_mapping_worked = True
                        raise NotFoundError(
                            message=f"VertexAIException - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )
                    if original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"VertexAIException - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )

                    if original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"litellm.RateLimitError: VertexAIException - {error_str}",
                            model=model,
                            llm_provider="vertex_ai",
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
                            message=f"VertexAIException InternalServerError - {error_str}",
                            model=model,
                            llm_provider="vertex_ai",
                            litellm_debug_info=extra_information,
                            response=httpx.Response(
                                status_code=500,
                                content=str(original_exception),
                                request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                            ),
                        )
                    if original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"VertexAIException - {original_exception.message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                        )
            elif custom_llm_provider == "palm" or custom_llm_provider == "gemini":
                if "503 Getting metadata" in error_str:
                    # auth errors look like this
                    # 503 Getting metadata from plugin failed with error: Reauthentication is needed. Please run `gcloud auth application-default login` to reauthenticate.
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message="GeminiException - Invalid api key",
                        model=model,
                        llm_provider="palm",
                        response=getattr(original_exception, "response", None),
                    )
                if (
                    "504 Deadline expired before operation could complete." in error_str
                    or "504 Deadline Exceeded" in error_str
                ):
                    exception_mapping_worked = True
                    raise Timeout(
                        message=f"GeminiException - {original_exception.message}",
                        model=model,
                        llm_provider="palm",
                        exception_status_code=original_exception.status_code,
                    )
                if "400 Request payload size exceeds" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"GeminiException - {error_str}",
                        model=model,
                        llm_provider="palm",
                        response=getattr(original_exception, "response", None),
                    )
                if (
                    "500 An internal error has occurred." in error_str
                    or "list index out of range" in error_str
                ):
                    exception_mapping_worked = True
                    raise APIError(
                        status_code=getattr(original_exception, "status_code", 500),
                        message=f"GeminiException - {original_exception.message}",
                        llm_provider="palm",
                        model=model,
                        request=httpx.Response(
                            status_code=429,
                            request=httpx.Request(
                                method="POST",
                                url=" https://cloud.google.com/vertex-ai/",
                            ),
                        ),
                    )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"GeminiException - {error_str}",
                            model=model,
                            llm_provider="palm",
                            response=getattr(original_exception, "response", None),
                        )
                # Dailed: Error occurred: 400 Request payload size exceeds the limit: 20000 bytes
            elif custom_llm_provider == "cloudflare":
                if "Authentication error" in error_str:
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"Cloudflare Exception - {original_exception.message}",
                        llm_provider="cloudflare",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                if "must have required property" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"Cloudflare Exception - {original_exception.message}",
                        llm_provider="cloudflare",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
            elif (
                custom_llm_provider == "cohere" or custom_llm_provider == "cohere_chat"
            ):  # Cohere
                if (
                    "invalid api token" in error_str
                    or "No API key provided." in error_str
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif "too many tokens" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"CohereException - {original_exception.message}",
                        model=model,
                        llm_provider="cohere",
                        response=getattr(original_exception, "response", None),
                    )
                elif hasattr(original_exception, "status_code"):
                    if (
                        original_exception.status_code == 400
                        or original_exception.status_code == 498
                    ):
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"CohereException - {original_exception.message}",
                            llm_provider="cohere",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"CohereException - {original_exception.message}",
                            llm_provider="cohere",
                            model=model,
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"CohereException - {original_exception.message}",
                            llm_provider="cohere",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                elif (
                    "CohereConnectionError" in exception_type
                ):  # cohere seems to fire these errors when we load test it (1k+ messages / min)
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif "invalid type:" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif "Unexpected server error" in error_str:
                    exception_mapping_worked = True
                    raise ServiceUnavailableError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                else:
                    if hasattr(original_exception, "status_code"):
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"CohereException - {original_exception.message}",
                            llm_provider="cohere",
                            model=model,
                            request=original_exception.request,
                        )
                    raise original_exception
            elif custom_llm_provider == "huggingface":
                if "length limit exceeded" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=error_str,
                        model=model,
                        llm_provider="huggingface",
                        response=getattr(original_exception, "response", None),
                    )
                elif "A valid user token is required" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=error_str,
                        llm_provider="huggingface",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif "Rate limit reached" in error_str:
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=error_str,
                        llm_provider="huggingface",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"HuggingfaceException - {original_exception.message}",
                            llm_provider="huggingface",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"HuggingfaceException - {original_exception.message}",
                            model=model,
                            llm_provider="huggingface",
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"HuggingfaceException - {original_exception.message}",
                            model=model,
                            llm_provider="huggingface",
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"HuggingfaceException - {original_exception.message}",
                            llm_provider="huggingface",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"HuggingfaceException - {original_exception.message}",
                            llm_provider="huggingface",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"HuggingfaceException - {original_exception.message}",
                            llm_provider="huggingface",
                            model=model,
                            request=original_exception.request,
                        )
            elif custom_llm_provider == "ai21":
                if hasattr(original_exception, "message"):
                    if "Prompt has too many tokens" in original_exception.message:
                        exception_mapping_worked = True
                        raise ContextWindowExceededError(
                            message=f"AI21Exception - {original_exception.message}",
                            model=model,
                            llm_provider="ai21",
                            response=getattr(original_exception, "response", None),
                        )
                    if "Bad or missing API token." in original_exception.message:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AI21Exception - {original_exception.message}",
                            model=model,
                            llm_provider="ai21",
                            response=getattr(original_exception, "response", None),
                        )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AI21Exception - {original_exception.message}",
                            llm_provider="ai21",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"AI21Exception - {original_exception.message}",
                            model=model,
                            llm_provider="ai21",
                        )
                    if original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AI21Exception - {original_exception.message}",
                            model=model,
                            llm_provider="ai21",
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"AI21Exception - {original_exception.message}",
                            llm_provider="ai21",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"AI21Exception - {original_exception.message}",
                            llm_provider="ai21",
                            model=model,
                            request=original_exception.request,
                        )
            elif custom_llm_provider == "nlp_cloud":
                if "detail" in error_str:
                    if "Input text length should not exceed" in error_str:
                        exception_mapping_worked = True
                        raise ContextWindowExceededError(
                            message=f"NLPCloudException - {error_str}",
                            model=model,
                            llm_provider="nlp_cloud",
                            response=getattr(original_exception, "response", None),
                        )
                    elif "value is not a valid" in error_str:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"NLPCloudException - {error_str}",
                            model=model,
                            llm_provider="nlp_cloud",
                            response=getattr(original_exception, "response", None),
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=500,
                            message=f"NLPCloudException - {error_str}",
                            model=model,
                            llm_provider="nlp_cloud",
                            request=original_exception.request,
                        )
                if hasattr(
                    original_exception, "status_code"
                ):  # https://docs.nlpcloud.com/?shell#errors
                    if (
                        original_exception.status_code == 400
                        or original_exception.status_code == 406
                        or original_exception.status_code == 413
                        or original_exception.status_code == 422
                    ):
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif (
                        original_exception.status_code == 401
                        or original_exception.status_code == 403
                    ):
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif (
                        original_exception.status_code == 522
                        or original_exception.status_code == 524
                    ):
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"NLPCloudException - {original_exception.message}",
                            model=model,
                            llm_provider="nlp_cloud",
                        )
                    elif (
                        original_exception.status_code == 429
                        or original_exception.status_code == 402
                    ):
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif (
                        original_exception.status_code == 500
                        or original_exception.status_code == 503
                    ):
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model,
                            request=original_exception.request,
                        )
                    elif (
                        original_exception.status_code == 504
                        or original_exception.status_code == 520
                    ):
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"NLPCloudException - {original_exception.message}",
                            model=model,
                            llm_provider="nlp_cloud",
                            response=getattr(original_exception, "response", None),
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model,
                            request=original_exception.request,
                        )
            elif custom_llm_provider == "together_ai":
                try:
                    error_response = json.loads(error_str)
                except Exception:
                    error_response = {"error": error_str}
                if (
                    "error" in error_response
                    and "`inputs` tokens + `max_new_tokens` must be <="
                    in error_response["error"]
                ):
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"TogetherAIException - {error_response['error']}",
                        model=model,
                        llm_provider="together_ai",
                        response=getattr(original_exception, "response", None),
                    )
                elif (
                    "error" in error_response
                    and "invalid private key" in error_response["error"]
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"TogetherAIException - {error_response['error']}",
                        llm_provider="together_ai",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif (
                    "error" in error_response
                    and "INVALID_ARGUMENT" in error_response["error"]
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"TogetherAIException - {error_response['error']}",
                        model=model,
                        llm_provider="together_ai",
                        response=getattr(original_exception, "response", None),
                    )
                elif "A timeout occurred" in error_str:
                    exception_mapping_worked = True
                    raise Timeout(
                        message=f"TogetherAIException - {error_str}",
                        model=model,
                        llm_provider="together_ai",
                    )
                elif (
                    "error" in error_response
                    and "API key doesn't match expected format."
                    in error_response["error"]
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"TogetherAIException - {error_response['error']}",
                        model=model,
                        llm_provider="together_ai",
                        response=getattr(original_exception, "response", None),
                    )
                elif (
                    "error_type" in error_response
                    and error_response["error_type"] == "validation"
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"TogetherAIException - {error_response['error']}",
                        model=model,
                        llm_provider="together_ai",
                        response=getattr(original_exception, "response", None),
                    )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"TogetherAIException - {original_exception.message}",
                            model=model,
                            llm_provider="together_ai",
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"TogetherAIException - {error_response['error']}",
                            model=model,
                            llm_provider="together_ai",
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"TogetherAIException - {original_exception.message}",
                            llm_provider="together_ai",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 524:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"TogetherAIException - {original_exception.message}",
                            llm_provider="together_ai",
                            model=model,
                        )
                else:
                    exception_mapping_worked = True
                    raise APIError(
                        status_code=original_exception.status_code,
                        message=f"TogetherAIException - {original_exception.message}",
                        llm_provider="together_ai",
                        model=model,
                        request=original_exception.request,
                    )
            elif custom_llm_provider == "aleph_alpha":
                if (
                    "This is longer than the model's maximum context length"
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"AlephAlphaException - {original_exception.message}",
                        llm_provider="aleph_alpha",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif "InvalidToken" in error_str or "No token provided" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"AlephAlphaException - {original_exception.message}",
                        llm_provider="aleph_alpha",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif hasattr(original_exception, "status_code"):
                    verbose_logger.debug(
                        f"status code: {original_exception.status_code}"
                    )
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model,
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model,
                            response=getattr(original_exception, "response", None),
                        )
                    raise original_exception
                raise original_exception
            elif (
                custom_llm_provider == "ollama" or custom_llm_provider == "ollama_chat"
            ):
                if isinstance(original_exception, dict):
                    error_str = original_exception.get("error", "")
                else:
                    error_str = str(original_exception)
                if "no such file or directory" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"OllamaException: Invalid Model/Model not loaded - {original_exception}",
                        model=model,
                        llm_provider="ollama",
                        response=getattr(original_exception, "response", None),
                    )
                elif "Failed to establish a new connection" in error_str:
                    exception_mapping_worked = True
                    raise ServiceUnavailableError(
                        message=f"OllamaException: {original_exception}",
                        llm_provider="ollama",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif "Invalid response object from API" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"OllamaException: {original_exception}",
                        llm_provider="ollama",
                        model=model,
                        response=getattr(original_exception, "response", None),
                    )
                elif "Read timed out" in error_str:
                    exception_mapping_worked = True
                    raise Timeout(
                        message=f"OllamaException: {original_exception}",
                        llm_provider="ollama",
                        model=model,
                    )
            elif custom_llm_provider == "vllm":
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 0:
                        exception_mapping_worked = True
                        raise APIConnectionError(
                            message=f"VLLMException - {original_exception.message}",
                            llm_provider="vllm",
                            model=model,
                            request=original_exception.request,
                        )
            elif custom_llm_provider == "azure" or custom_llm_provider == "azure_text":
                message = get_error_message(error_obj=original_exception)
                if message is None:
                    if hasattr(original_exception, "message"):
                        message = original_exception.message
                    else:
                        message = str(original_exception)

                if "Internal server error" in error_str:
                    exception_mapping_worked = True
                    raise litellm.InternalServerError(
                        message=f"AzureException Internal server error - {message}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        response=getattr(original_exception, "response", None),
                    )
                elif "This model's maximum context length is" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"AzureException ContextWindowExceededError - {message}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        response=getattr(original_exception, "response", None),
                    )
                elif "DeploymentNotFound" in error_str:
                    exception_mapping_worked = True
                    raise NotFoundError(
                        message=f"AzureException NotFoundError - {message}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        response=getattr(original_exception, "response", None),
                    )
                elif (
                    (
                        "invalid_request_error" in error_str
                        and "content_policy_violation" in error_str
                    )
                    or (
                        "The response was filtered due to the prompt triggering Azure OpenAI's content management"
                        in error_str
                    )
                    or "Your task failed as a result of our safety system" in error_str
                    or "The model produced invalid content" in error_str
                    or "content_filter_policy" in error_str
                ):
                    exception_mapping_worked = True
                    raise ContentPolicyViolationError(
                        message=f"litellm.ContentPolicyViolationError: AzureException - {message}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        response=getattr(original_exception, "response", None),
                    )
                elif "invalid_request_error" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"AzureException BadRequestError - {message}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        response=getattr(original_exception, "response", None),
                        body=getattr(original_exception, "body", None),
                    )
                elif (
                    "The api_key client option must be set either by passing api_key to the client or by setting"
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"{exception_provider} AuthenticationError - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        litellm_debug_info=extra_information,
                        response=getattr(original_exception, "response", None),
                    )
                elif "Connection error" in error_str:
                    exception_mapping_worked = True
                    raise APIConnectionError(
                        message=f"{exception_provider} APIConnectionError - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        litellm_debug_info=extra_information,
                    )
                elif hasattr(original_exception, "status_code"):
                    exception_mapping_worked = True
                    if original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AzureException - {message}",
                            llm_provider="azure",
                            model=model,
                            litellm_debug_info=extra_information,
                            response=getattr(original_exception, "response", None),
                            body=getattr(original_exception, "body", None),
                        )
                    elif original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AzureException AuthenticationError - {message}",
                            llm_provider="azure",
                            model=model,
                            litellm_debug_info=extra_information,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"AzureException Timeout - {message}",
                            model=model,
                            litellm_debug_info=extra_information,
                            llm_provider="azure",
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AzureException BadRequestError - {message}",
                            model=model,
                            llm_provider="azure",
                            litellm_debug_info=extra_information,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"AzureException RateLimitError - {message}",
                            model=model,
                            llm_provider="azure",
                            litellm_debug_info=extra_information,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"AzureException ServiceUnavailableError - {message}",
                            model=model,
                            llm_provider="azure",
                            litellm_debug_info=extra_information,
                            response=getattr(original_exception, "response", None),
                        )
                    elif original_exception.status_code == 504:  # gateway timeout error
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"AzureException Timeout - {message}",
                            model=model,
                            litellm_debug_info=extra_information,
                            llm_provider="azure",
                            exception_status_code=original_exception.status_code,
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"AzureException APIError - {message}",
                            llm_provider="azure",
                            litellm_debug_info=extra_information,
                            model=model,
                            request=httpx.Request(
                                method="POST", url="https://openai.com/"
                            ),
                        )
                else:
                    # if no status code then it is an APIConnectionError: https://github.com/openai/openai-python#handling-errors
                    raise APIConnectionError(
                        message=f"{exception_provider} APIConnectionError - {message}\n{traceback.format_exc()}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        request=httpx.Request(method="POST", url="https://openai.com/"),
                    )
            if custom_llm_provider == "openrouter":
                if hasattr(original_exception, "status_code"):
                    exception_mapping_worked = True
                    if original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"{exception_provider} - {error_str}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AuthenticationError: {exception_provider} - {error_str}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 404:
                        exception_mapping_worked = True
                        raise NotFoundError(
                            message=f"NotFoundError: {exception_provider} - {error_str}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"Timeout Error: {exception_provider} - {error_str}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"BadRequestError: {exception_provider} - {error_str}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"RateLimitError: {exception_provider} - {error_str}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"ServiceUnavailableError: {exception_provider} - {error_str}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=getattr(original_exception, "response", None),
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 504:  # gateway timeout error
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"Timeout Error: {exception_provider} - {error_str}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                            exception_status_code=original_exception.status_code,
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"APIError: {exception_provider} - {error_str}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            request=original_exception.request,
                            litellm_debug_info=extra_information,
                        )
                else:
                    # if no status code then it is an APIConnectionError: https://github.com/openai/openai-python#handling-errors
                    raise APIConnectionError(
                        message=f"APIConnectionError: {exception_provider} - {error_str}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        litellm_debug_info=extra_information,
                        request=httpx.Request(
                            method="POST", url="https://api.openai.com/v1/"
                        ),
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
                    request=original_exception.request,
                )
            else:
                raise APIConnectionError(
                    message="{}\n{}".format(
                        str(original_exception), traceback.format_exc()
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
                message="{}\n{}".format(original_exception, traceback.format_exc()),
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
