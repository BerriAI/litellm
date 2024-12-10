from typing import Optional, Union

import httpx

from litellm.llms.base_llm.transformation import BaseLLMException


class AzureOpenAIError(BaseLLMException):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[httpx.Headers, dict]] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
        )


def process_azure_headers(headers: Union[httpx.Headers, dict]) -> dict:
    openai_headers = {}
    if "x-ratelimit-limit-requests" in headers:
        openai_headers["x-ratelimit-limit-requests"] = headers[
            "x-ratelimit-limit-requests"
        ]
    if "x-ratelimit-remaining-requests" in headers:
        openai_headers["x-ratelimit-remaining-requests"] = headers[
            "x-ratelimit-remaining-requests"
        ]
    if "x-ratelimit-limit-tokens" in headers:
        openai_headers["x-ratelimit-limit-tokens"] = headers["x-ratelimit-limit-tokens"]
    if "x-ratelimit-remaining-tokens" in headers:
        openai_headers["x-ratelimit-remaining-tokens"] = headers[
            "x-ratelimit-remaining-tokens"
        ]
    llm_response_headers = {
        "{}-{}".format("llm_provider", k): v for k, v in headers.items()
    }

    return {**llm_response_headers, **openai_headers}
