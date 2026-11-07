import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class InfinityError(BaseLLMException):
    def __init__(self, status_code: int, message: str, headers: dict | httpx.Headers | None = None):
        resolved_headers = {} if headers is None else headers
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://github.com/michaelfeil/infinity")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            request=self.request,
            response=self.response,
            headers=resolved_headers,
        )  # Call the base class constructor with the parameters it needs
