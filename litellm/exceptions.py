## LiteLLM versions of the OpenAI Exception Types
from openai.error import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
    OpenAIError,
)


class AuthenticationError(AuthenticationError):  # type: ignore
    def __init__(self, message, llm_provider, model):
        self.status_code = 401
        self.message = message
        self.llm_provider = llm_provider
        self.model = model
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class InvalidRequestError(InvalidRequestError):  # type: ignore
    def __init__(self, message, model, llm_provider):
        self.status_code = 400
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        super().__init__(
            self.message, f"{self.model}"
        )  # Call the base class constructor with the parameters it needs

# sub class of invalid request error - meant to give more granularity for error handling context window exceeded errors
class ContextWindowExceededError(InvalidRequestError):  # type: ignore
    def __init__(self, message, model, llm_provider):
        self.status_code = 400
        self.message = message
        self.model = model
        self.llm_provider = llm_provider
        super().__init__(
            self.message, self.model, self.llm_provider
        )  # Call the base class constructor with the parameters it needs


class RateLimitError(RateLimitError):  # type: ignore
    def __init__(self, message, llm_provider, model):
        self.status_code = 429
        self.message = message
        self.llm_provider = llm_provider
        self.modle = model
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class ServiceUnavailableError(ServiceUnavailableError):  # type: ignore
    def __init__(self, message, llm_provider, model):
        self.status_code = 500
        self.message = message
        self.llm_provider = llm_provider
        self.model = model
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


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
