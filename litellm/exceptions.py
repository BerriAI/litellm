# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

## LiteLLM versions of the OpenAI Exception Types

from openai import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    APIStatusError,
    OpenAIError,
    APIError,
    APITimeoutError,
    APIConnectionError,
    APIResponseValidationError,
    UnprocessableEntityError,
    PermissionDeniedError,
)
import httpx
from typing import Optional


class AuthenticationError(AuthenticationError):  # type: ignore
    def __init__(self, message, llm_provider, model, response: httpx.Response):
        self.status_code = 401
        self.message = message
        self.llm_provider = llm_provider
        self.model = model
        super().__init__(
            self.message, response=response, body=None
        )  # Call the base class constructor with the parameters it needs


# raise when invalid models passed, example gpt-8
class NotFoundError(NotFoundError):  # type: ignore
    def __init__(self, message, model, llm_provider, response: httpx.Response):
        self.status_code = 404
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        super().__init__(
            self.message, response=response, body=None
        )  # Call the base class constructor with the parameters it needs


class BadRequestError(BadRequestError):  # type: ignore
    def __init__(
        self, message, model, llm_provider, response: Optional[httpx.Response] = None
    ):
        self.status_code = 400
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        response = response or httpx.Response(
            status_code=self.status_code,
            request=httpx.Request(
                method="GET", url="https://litellm.ai"
            ),  # mock request object
        )
        super().__init__(
            self.message, response=response, body=None
        )  # Call the base class constructor with the parameters it needs


class UnprocessableEntityError(UnprocessableEntityError):  # type: ignore
    def __init__(self, message, model, llm_provider, response: httpx.Response):
        self.status_code = 422
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        super().__init__(
            self.message, response=response, body=None
        )  # Call the base class constructor with the parameters it needs


class Timeout(APITimeoutError):  # type: ignore
    def __init__(self, message, model, llm_provider):
        self.status_code = 408
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        super().__init__(
            request=request
        )  # Call the base class constructor with the parameters it needs


class PermissionDeniedError(PermissionDeniedError):  # type:ignore
    def __init__(self, message, llm_provider, model, response: httpx.Response):
        self.status_code = 403
        self.message = message
        self.llm_provider = llm_provider
        self.model = model
        super().__init__(
            self.message, response=response, body=None
        )  # Call the base class constructor with the parameters it needs


class RateLimitError(RateLimitError):  # type: ignore
    def __init__(self, message, llm_provider, model, response: httpx.Response):
        self.status_code = 429
        self.message = message
        self.llm_provider = llm_provider
        self.modle = model
        super().__init__(
            self.message, response=response, body=None
        )  # Call the base class constructor with the parameters it needs


# sub class of rate limit error - meant to give more granularity for error handling context window exceeded errors
class ContextWindowExceededError(BadRequestError):  # type: ignore
    def __init__(self, message, model, llm_provider, response: httpx.Response):
        self.status_code = 400
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        super().__init__(
            message=self.message,
            model=self.model,  # type: ignore
            llm_provider=self.llm_provider,  # type: ignore
            response=response,
        )  # Call the base class constructor with the parameters it needs


class ContentPolicyViolationError(BadRequestError):  # type: ignore
    #  Error code: 400 - {'error': {'code': 'content_policy_violation', 'message': 'Your request was rejected as a result of our safety system. Image descriptions generated from your prompt may contain text that is not allowed by our safety system. If you believe this was done in error, your request may succeed if retried, or by adjusting your prompt.', 'param': None, 'type': 'invalid_request_error'}}
    def __init__(self, message, model, llm_provider, response: httpx.Response):
        self.status_code = 400
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        super().__init__(
            message=self.message,
            model=self.model,  # type: ignore
            llm_provider=self.llm_provider,  # type: ignore
            response=response,
        )  # Call the base class constructor with the parameters it needs


class ServiceUnavailableError(APIStatusError):  # type: ignore
    def __init__(self, message, llm_provider, model, response: httpx.Response):
        self.status_code = 503
        self.message = message
        self.llm_provider = llm_provider
        self.model = model
        super().__init__(
            self.message, response=response, body=None
        )  # Call the base class constructor with the parameters it needs


# raise this when the API returns an invalid response object - https://github.com/openai/openai-python/blob/1be14ee34a0f8e42d3f9aa5451aa4cb161f1781f/openai/api_requestor.py#L401
class APIError(APIError):  # type: ignore
    def __init__(
        self, status_code, message, llm_provider, model, request: httpx.Request
    ):
        self.status_code = status_code
        self.message = message
        self.llm_provider = llm_provider
        self.model = model
        super().__init__(self.message, request=request, body=None)  # type: ignore


# raised if an invalid request (not get, delete, put, post) is made
class APIConnectionError(APIConnectionError):  # type: ignore
    def __init__(self, message, llm_provider, model, request: httpx.Request):
        self.message = message
        self.llm_provider = llm_provider
        self.model = model
        self.status_code = 500
        super().__init__(message=self.message, request=request)


# raised if an invalid request (not get, delete, put, post) is made
class APIResponseValidationError(APIResponseValidationError):  # type: ignore
    def __init__(self, message, llm_provider, model):
        self.message = message
        self.llm_provider = llm_provider
        self.model = model
        request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        response = httpx.Response(status_code=500, request=request)
        super().__init__(response=response, body=None, message=message)


class OpenAIError(OpenAIError):  # type: ignore
    def __init__(self, original_exception):
        self.status_code = original_exception.http_status
        super().__init__(
            http_body=original_exception.http_body,
            http_status=original_exception.http_status,
            json_body=original_exception.json_body,
            headers=original_exception.headers,
            code=original_exception.code,
        )
        self.llm_provider = "openai"


class BudgetExceededError(Exception):
    def __init__(self, current_cost, max_budget):
        self.current_cost = current_cost
        self.max_budget = max_budget
        message = f"Budget has been exceeded! Current cost: {current_cost}, Max budget: {max_budget}"
        super().__init__(message)


## DEPRECATED ##
class InvalidRequestError(BadRequestError):  # type: ignore
    def __init__(self, message, model, llm_provider):
        self.status_code = 400
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        super().__init__(
            self.message, f"{self.model}"
        )  # Call the base class constructor with the parameters it needs
