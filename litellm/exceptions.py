# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

## LiteLLM versions of the OpenAI Exception Types

from typing import Optional

import httpx
import openai

from litellm.types.utils import LiteLLMCommonStrings

_MINIMAL_ERROR_RESPONSE: Optional[httpx.Response] = None


def _get_minimal_error_response() -> httpx.Response:
    """Get a cached minimal httpx.Response object for error cases."""
    global _MINIMAL_ERROR_RESPONSE
    if _MINIMAL_ERROR_RESPONSE is None:
        _MINIMAL_ERROR_RESPONSE = httpx.Response(
            status_code=400,
            request=httpx.Request(
                method="GET", url="https://litellm.ai"
            ),
        )
    return _MINIMAL_ERROR_RESPONSE


class AuthenticationError(openai.AuthenticationError):  # type: ignore
    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 401
        self.message = "litellm.AuthenticationError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        self.response = response or httpx.Response(
            status_code=self.status_code,
            request=httpx.Request(
                method="GET", url="https://litellm.ai"
            ),  # mock request object
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# raise when invalid models passed, example gpt-8
class NotFoundError(openai.NotFoundError):  # type: ignore
    def __init__(
        self,
        message,
        model,
        llm_provider,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 404
        self.message = "litellm.NotFoundError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        self.response = response or httpx.Response(
            status_code=self.status_code,
            request=httpx.Request(
                method="GET", url="https://litellm.ai"
            ),  # mock request object
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class BadRequestError(openai.BadRequestError):  # type: ignore
    def __init__(
        self,
        message,
        model,
        llm_provider,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
        body: Optional[dict] = None,
    ):
        self.status_code = 400
        self.message = "litellm.BadRequestError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        # Use response if it's a valid httpx.Response with a request, otherwise use minimal error response
        # Note: We check _request (not .request property) to avoid RuntimeError when _request is None
        if (
            response is not None
            and isinstance(response, httpx.Response)
            and hasattr(response, "_request")
            and getattr(response, "_request", None) is not None
        ):
            self.response = response
        else:
            self.response = _get_minimal_error_response()
        super().__init__(
            self.message, response=self.response, body=body
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class ImageFetchError(BadRequestError):
    def __init__(
        self,
        message,
        model=None,
        llm_provider=None,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
        body: Optional[dict] = None,
    ):
        super().__init__(
            message=message,
            model=model,
            llm_provider=llm_provider,
            response=response,
            litellm_debug_info=litellm_debug_info,
            max_retries=max_retries,
            num_retries=num_retries,
            body=body,
        )


class UnprocessableEntityError(openai.UnprocessableEntityError):  # type: ignore
    def __init__(
        self,
        message,
        model,
        llm_provider,
        response: httpx.Response,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 422
        self.message = "litellm.UnprocessableEntityError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        super().__init__(
            self.message, response=response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class Timeout(openai.APITimeoutError):  # type: ignore
    def __init__(
        self,
        message,
        model,
        llm_provider,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
        headers: Optional[dict] = None,
        exception_status_code: Optional[int] = None,
    ):
        request = httpx.Request(
            method="POST",
            url="https://api.openai.com/v1",
        )
        super().__init__(
            request=request
        )  # Call the base class constructor with the parameters it needs
        self.status_code = exception_status_code or 408
        self.message = "litellm.Timeout: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        self.headers = headers

    # custom function to convert to str
    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class PermissionDeniedError(openai.PermissionDeniedError):  # type:ignore
    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: httpx.Response,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 403
        self.message = "litellm.PermissionDeniedError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        super().__init__(
            self.message, response=response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class RateLimitError(openai.RateLimitError):  # type: ignore
    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 429
        self.message = "litellm.RateLimitError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        _response_headers = (
            getattr(response, "headers", None) if response is not None else None
        )
        self.response = httpx.Response(
            status_code=429,
            headers=_response_headers,
            request=httpx.Request(
                method="POST",
                url=" https://cloud.google.com/vertex-ai/",
            ),
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs
        self.code = "429"
        self.type = "throttling_error"

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# sub class of rate limit error - meant to give more granularity for error handling context window exceeded errors
class ContextWindowExceededError(BadRequestError):  # type: ignore
    def __init__(
        self,
        message,
        model,
        llm_provider,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
    ):
        self.status_code = 400
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        super().__init__(
            message=message,
            model=self.model,  # type: ignore
            llm_provider=self.llm_provider,  # type: ignore
            response=response,
            litellm_debug_info=self.litellm_debug_info,
        )  # Call the base class constructor with the parameters it needs

        # set after, to make it clear the raised error is a context window exceeded error
        self.message = "litellm.ContextWindowExceededError: {}".format(self.message)

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# sub class of bad request error - meant to help us catch guardrails-related errors on proxy.
class RejectedRequestError(BadRequestError):  # type: ignore
    def __init__(
        self,
        message,
        model,
        llm_provider,
        request_data: dict,
        litellm_debug_info: Optional[str] = None,
    ):
        self.status_code = 400
        self.message = "litellm.RejectedRequestError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.request_data = request_data
        request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        response = httpx.Response(status_code=400, request=request)
        super().__init__(
            message=self.message,
            model=self.model,  # type: ignore
            llm_provider=self.llm_provider,  # type: ignore
            response=response,
            litellm_debug_info=self.litellm_debug_info,
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class ContentPolicyViolationError(BadRequestError):  # type: ignore
    #  Error code: 400 - {'error': {'code': 'content_policy_violation', 'message': 'Your request was rejected as a result of our safety system. Image descriptions generated from your prompt may contain text that is not allowed by our safety system. If you believe this was done in error, your request may succeed if retried, or by adjusting your prompt.', 'param': None, 'type': 'invalid_request_error'}}
    def __init__(
        self,
        message,
        model,
        llm_provider,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        provider_specific_fields: Optional[dict] = None,
        body: Optional[dict] = None,
    ):
        self.status_code = 400
        self.message = "litellm.ContentPolicyViolationError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        self.provider_specific_fields = provider_specific_fields
        super().__init__(
            message=self.message,
            model=self.model,  # type: ignore
            llm_provider=self.llm_provider,  # type: ignore
            response=response,
            litellm_debug_info=self.litellm_debug_info,
            body=body,
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        return self._transform_error_to_string()

    def __repr__(self):
        return self._transform_error_to_string()

    def _transform_error_to_string(self) -> str:
        """
        Transform the error to a string
        """
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class ServiceUnavailableError(openai.APIStatusError):  # type: ignore
    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 503
        self.message = "litellm.ServiceUnavailableError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        _response_headers = (
            getattr(response, "headers", None) if response is not None else None
        )
        self.response = httpx.Response(
            status_code=self.status_code,
            headers=_response_headers,
            request=httpx.Request(
                method="POST",
                url=" https://cloud.google.com/vertex-ai/",
            ),
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class BadGatewayError(openai.APIStatusError):  # type: ignore
    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 502
        self.message = "litellm.BadGatewayError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        _response_headers = (
            getattr(response, "headers", None) if response is not None else None
        )
        self.response = httpx.Response(
            status_code=self.status_code,
            headers=_response_headers,
            request=httpx.Request(
                method="POST",
                url=" https://cloud.google.com/vertex-ai/",
            ),
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class InternalServerError(openai.InternalServerError):  # type: ignore
    def __init__(
        self,
        message,
        llm_provider,
        model,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 500
        self.message = "litellm.InternalServerError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        _response_headers = (
            getattr(response, "headers", None) if response is not None else None
        )
        self.response = httpx.Response(
            status_code=self.status_code,
            headers=_response_headers,
            request=httpx.Request(
                method="POST",
                url=" https://cloud.google.com/vertex-ai/",
            ),
        )
        super().__init__(
            self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# raise this when the API returns an invalid response object - https://github.com/openai/openai-python/blob/1be14ee34a0f8e42d3f9aa5451aa4cb161f1781f/openai/api_requestor.py#L401
class APIError(openai.APIError):  # type: ignore
    def __init__(
        self,
        status_code: int,
        message,
        llm_provider,
        model,
        request: Optional[httpx.Request] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = status_code
        self.message = "litellm.APIError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        if request is None:
            request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        super().__init__(self.message, request=request, body=None)  # type: ignore

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# raised if an invalid request (not get, delete, put, post) is made
class APIConnectionError(openai.APIConnectionError):  # type: ignore
    def __init__(
        self,
        message,
        llm_provider,
        model,
        request: Optional[httpx.Request] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.message = "litellm.APIConnectionError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.status_code = 500
        self.litellm_debug_info = litellm_debug_info
        self.request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        self.max_retries = max_retries
        self.num_retries = num_retries
        super().__init__(message=self.message, request=self.request)

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


# raised if an invalid request (not get, delete, put, post) is made
class APIResponseValidationError(openai.APIResponseValidationError):  # type: ignore
    def __init__(
        self,
        message,
        llm_provider,
        model,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.message = "litellm.APIResponseValidationError: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        response = httpx.Response(status_code=500, request=request)
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        super().__init__(response=response, body=None, message=message)

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message


class JSONSchemaValidationError(APIResponseValidationError):
    def __init__(
        self, model: str, llm_provider: str, raw_response: str, schema: str
    ) -> None:
        self.raw_response = raw_response
        self.schema = schema
        self.model = model
        message = "litellm.JSONSchemaValidationError: model={}, returned an invalid response={}, for schema={}.\nAccess raw response with `e.raw_response`".format(
            model, raw_response, schema
        )
        self.message = message
        super().__init__(model=model, message=message, llm_provider=llm_provider)


class OpenAIError(openai.OpenAIError):  # type: ignore
    def __init__(self, original_exception=None):
        super().__init__()
        self.llm_provider = "openai"


class UnsupportedParamsError(BadRequestError):
    def __init__(
        self,
        message,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: int = 400,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = 400
        self.message = "litellm.UnsupportedParamsError: {}".format(message)
        self.model = model
        self.llm_provider = llm_provider
        self.litellm_debug_info = litellm_debug_info
        response = response or httpx.Response(
            status_code=self.status_code,
            request=httpx.Request(
                method="GET", url="https://litellm.ai"
            ),  # mock request object
        )
        self.max_retries = max_retries
        self.num_retries = num_retries


LITELLM_EXCEPTION_TYPES = [
    AuthenticationError,
    NotFoundError,
    BadRequestError,
    UnprocessableEntityError,
    UnsupportedParamsError,
    Timeout,
    PermissionDeniedError,
    RateLimitError,
    ContextWindowExceededError,
    RejectedRequestError,
    ContentPolicyViolationError,
    InternalServerError,
    ServiceUnavailableError,
    BadGatewayError,
    APIError,
    APIConnectionError,
    APIResponseValidationError,
    OpenAIError,
    InternalServerError,
    JSONSchemaValidationError,
]


class BudgetExceededError(Exception):
    def __init__(
        self, current_cost: float, max_budget: float, message: Optional[str] = None
    ):
        self.current_cost = current_cost
        self.max_budget = max_budget
        message = (
            message
            or f"Budget has been exceeded! Current cost: {current_cost}, Max budget: {max_budget}"
        )
        self.message = message
        super().__init__(message)


## DEPRECATED ##
class InvalidRequestError(openai.BadRequestError):  # type: ignore
    def __init__(self, message, model, llm_provider):
        self.status_code = 400
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        self.response = httpx.Response(
            status_code=400,
            request=httpx.Request(
                method="GET", url="https://litellm.ai"
            ),  # mock request object
        )
        super().__init__(
            message=self.message, response=self.response, body=None
        )  # Call the base class constructor with the parameters it needs


class MockException(openai.APIError):
    # used for testing
    def __init__(
        self,
        status_code: int,
        message,
        llm_provider,
        model,
        request: Optional[httpx.Request] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = status_code
        self.message = "litellm.MockException: {}".format(message)
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        if request is None:
            request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        super().__init__(self.message, request=request, body=None)  # type: ignore


class LiteLLMUnknownProvider(BadRequestError):
    def __init__(self, model: str, custom_llm_provider: Optional[str] = None):
        self.message = LiteLLMCommonStrings.llm_provider_not_provided.value.format(
            model=model, custom_llm_provider=custom_llm_provider
        )
        super().__init__(
            self.message, model=model, llm_provider=custom_llm_provider, response=None
        )

    def __str__(self):
        return self.message


class GuardrailRaisedException(Exception):
    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        message: str = "",
        should_wrap_with_default_message: bool = True,
    ):
        default_message = f"Guardrail raised an exception, Guardrail: {guardrail_name}, Message: {message}"
        self.guardrail_name = guardrail_name
        self.message = default_message if should_wrap_with_default_message else message
        super().__init__(self.message)


class BlockedPiiEntityError(Exception):
    def __init__(
        self,
        entity_type: str,
        guardrail_name: Optional[str] = None,
    ):
        """
        Raised when a blocked entity is detected by a guardrail.
        """
        self.entity_type = entity_type
        self.guardrail_name = guardrail_name
        self.message = f"Blocked entity detected: {entity_type} by Guardrail: {guardrail_name}. This entity is not allowed to be used in this request."
        super().__init__(self.message)


class MidStreamFallbackError(ServiceUnavailableError):  # type: ignore
    def __init__(
        self,
        message: str,
        model: str,
        llm_provider: str,
        original_exception: Optional[Exception] = None,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
        generated_content: str = "",
        is_pre_first_chunk: bool = False,
    ):
        self.status_code = 503  # Service Unavailable
        self.message = f"litellm.MidStreamFallbackError: {message}"
        self.model = model
        self.llm_provider = llm_provider
        self.original_exception = original_exception
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        self.generated_content = generated_content
        self.is_pre_first_chunk = is_pre_first_chunk

        # Create a response if one wasn't provided
        if response is None:
            self.response = httpx.Response(
                status_code=self.status_code,
                request=httpx.Request(
                    method="POST",
                    url=f"https://{llm_provider}.com/v1/",
                ),
            )
        else:
            self.response = response

        # Call the parent constructor
        super().__init__(
            message=self.message,
            llm_provider=llm_provider,
            model=model,
            response=self.response,
            litellm_debug_info=self.litellm_debug_info,
            max_retries=self.max_retries,
            num_retries=self.num_retries,
        )

    def __str__(self):
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        if self.original_exception:
            _message += f" Original exception: {type(self.original_exception).__name__}: {str(self.original_exception)}"
        return _message

    def __repr__(self):
        return self.__str__()


class GuardrailInterventionNormalStringError(
    Exception
):  # custom exception to raise when a guardrail intervenes, but we want to return a normal string to the user
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.__str__()
