import asyncio
import time
from typing import (
    Final,
    cast,  # noqa: TID251  # casts moved verbatim from litellm/main.py
)

import httpx
import openai

import litellm
from litellm.constants import DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT, DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT
from litellm.llms.base_llm.base_model_iterator import convert_model_response_to_streaming
from litellm.native.request_support import admitted_prompt_token_counter
from litellm.types.utils import ChatCompletionMessageToolCall, ModelResponseStream
from litellm.utils import (
    CustomStreamWrapper,
    ModelResponse,
    Usage,
    async_mock_completion_streaming_obj,
    mock_completion_streaming_obj,
)

MOCK_RESPONSE_TYPE = str | Exception | dict | ModelResponse | ModelResponseStream


def handle_mock_potential_exceptions(
    mock_response: str | Exception,
    model: str,
    custom_llm_provider: str | None = None,
) -> None:
    if isinstance(mock_response, Exception):
        if isinstance(mock_response, openai.APIError):
            raise mock_response
        raise litellm.MockException(
            status_code=getattr(mock_response, "status_code", 500),
            message=getattr(mock_response, "text", str(mock_response)),
            llm_provider=getattr(mock_response, "llm_provider", custom_llm_provider or "openai"),
            model=model,
            request=httpx.Request(method="POST", url="https://api.openai.com/v1/"),
        )
    elif isinstance(mock_response, str) and mock_response == "litellm.RateLimitError":
        raise litellm.RateLimitError(
            message="this is a mock rate limit error",
            llm_provider=getattr(mock_response, "llm_provider", custom_llm_provider or "openai"),
            model=model,
        )
    elif isinstance(mock_response, str) and mock_response == "litellm.ContextWindowExceededError":
        raise litellm.ContextWindowExceededError(
            message="this is a mock context window exceeded error",
            llm_provider=getattr(mock_response, "llm_provider", custom_llm_provider or "openai"),
            model=model,
        )
    elif isinstance(mock_response, str) and mock_response == "litellm.InternalServerError":
        raise litellm.InternalServerError(
            message="this is a mock internal server error",
            llm_provider=getattr(mock_response, "llm_provider", custom_llm_provider or "openai"),
            model=model,
        )
    elif isinstance(mock_response, str) and mock_response.startswith("Exception: content_filter_policy"):
        raise litellm.MockException(
            status_code=400,
            message=mock_response,
            llm_provider="azure",
            model=model,
            request=httpx.Request(method="POST", url="https://api.openai.com/v1/"),
        )


def handle_mock_timeout(
    mock_timeout: bool | None,
    timeout: float | str | httpx.Timeout | None,
    model: str,
) -> None:
    if mock_timeout is True and timeout is not None:
        _sleep_for_timeout(timeout)
        raise litellm.Timeout(
            message="This is a mock timeout error",
            llm_provider="openai",
            model=model,
        )


async def handle_mock_timeout_async(
    mock_timeout: bool | None,
    timeout: float | str | httpx.Timeout | None,
    model: str,
) -> None:
    if mock_timeout is True and timeout is not None:
        await _sleep_for_timeout_async(timeout)
        raise litellm.Timeout(
            message="This is a mock timeout error",
            llm_provider="openai",
            model=model,
        )


def _sleep_for_timeout(timeout: float | str | httpx.Timeout):
    if isinstance(timeout, float):
        time.sleep(timeout)
    elif isinstance(timeout, str):
        time.sleep(float(timeout))
    elif isinstance(timeout, httpx.Timeout) and timeout.connect is not None:
        time.sleep(timeout.connect)


async def _sleep_for_timeout_async(timeout: float | str | httpx.Timeout):
    if isinstance(timeout, float):
        await asyncio.sleep(timeout)
    elif isinstance(timeout, str):
        await asyncio.sleep(float(timeout))
    elif isinstance(timeout, httpx.Timeout) and timeout.connect is not None:
        await asyncio.sleep(timeout.connect)


def mock_completion(
    model: str,
    messages: list,
    stream: bool | None = False,
    n: int | None = None,
    mock_response: MOCK_RESPONSE_TYPE | None = "This is a mock request",
    mock_tool_calls: list | None = None,
    mock_timeout: bool | None = False,
    logging=None,
    custom_llm_provider=None,
    timeout: float | str | httpx.Timeout | None = None,
    prompt_tokens: int | None = None,
    **kwargs,
):
    """
    Generate a mock completion response for testing or debugging purposes.

    This is a helper function that simulates the response structure of the OpenAI completion API.

    Parameters:
        model (str): The name of the language model for which the mock response is generated.
        messages (List): A list of message objects representing the conversation context.
        stream (bool, optional): If True, returns a mock streaming response (default is False).
        mock_response (str, optional): The content of the mock response (default is "This is a mock request").
        mock_timeout (bool, optional): If True, the mock response will be a timeout error (default is False).
        timeout (float, optional): The timeout value to use for the mock response (default is None).
        **kwargs: Additional keyword arguments that can be used but are not required.

    Returns:
        litellm.ModelResponse: A ModelResponse simulating a completion response with the specified model, messages, and mock response.

    Raises:
        Exception: If an error occurs during the generation of the mock completion response.
    Note:
        - This function is intended for testing or debugging purposes to generate mock completion responses.
        - If 'stream' is True, it returns a response that mimics the behavior of a streaming completion.
    """
    try:
        is_acompletion: Final = kwargs.get("acompletion") or False
        if mock_response is None:
            mock_response = "This is a mock request"

        handle_mock_timeout(mock_timeout=mock_timeout, timeout=timeout, model=model)

        ## LOGGING
        if logging is not None:
            logging.pre_call(
                input=messages,
                api_key="mock-key",
            )

        if isinstance(mock_response, str) or isinstance(mock_response, Exception):
            handle_mock_potential_exceptions(
                mock_response=mock_response,
                model=model,
                custom_llm_provider=custom_llm_provider,
            )

        mock_response = cast(
            str | dict | ModelResponse | ModelResponseStream, mock_response
        )  # after this point, mock_response is a string, dict, ModelResponse, or ModelResponseStream
        if isinstance(mock_response, str) and mock_response.startswith("Exception: mock_streaming_error"):
            mock_response = litellm.MockException(
                message="This is a mock error raised mid-stream",
                llm_provider="anthropic",
                model=model,
                status_code=529,
            )
        time_delay: Final = kwargs.get("mock_delay", None)
        if time_delay is not None and not is_acompletion:
            time.sleep(time_delay)

        if isinstance(mock_response, dict):
            return ModelResponse(**mock_response)

        if isinstance(mock_response, ModelResponse):
            if not stream:
                return mock_response
            # convert to ModelResponseStream
            mock_response = convert_model_response_to_streaming(mock_response)

        model_response: ModelResponse | ModelResponseStream = ModelResponse()

        if stream is True:
            model_response = ModelResponseStream()
            count_prompt_tokens: Final = admitted_prompt_token_counter(prompt_tokens)
            # don't try to access stream object,
            if kwargs.get("acompletion", False) is True:
                return CustomStreamWrapper(
                    completion_stream=async_mock_completion_streaming_obj(
                        model_response, mock_response=mock_response, model=model, n=n, prompt_tokens=prompt_tokens
                    ),
                    model=model,
                    custom_llm_provider="openai",
                    logging_obj=logging,
                    count_prompt_tokens=count_prompt_tokens,
                )
            return CustomStreamWrapper(
                completion_stream=mock_completion_streaming_obj(
                    model_response, mock_response=mock_response, model=model, n=n, prompt_tokens=prompt_tokens
                ),
                model=model,
                custom_llm_provider="openai",
                logging_obj=logging,
                count_prompt_tokens=count_prompt_tokens,
            )
        if isinstance(mock_response, litellm.MockException):
            raise mock_response
        # At this point, mock_response must be a string (all other types have been handled or returned early)
        mock_response = cast(str, mock_response)

        if n is None:
            model_response.choices[0].message.content = mock_response
        else:
            _all_choices: Final = []
            for i in range(n):
                _choice = litellm.utils.Choices(
                    index=i,
                    message=litellm.utils.Message(content=mock_response, role="assistant"),
                )
                _all_choices.append(_choice)
            model_response.choices = _all_choices
        model_response.created = int(time.time())
        model_response.model = model

        if mock_tool_calls:
            model_response.choices[0].message.tool_calls = [
                ChatCompletionMessageToolCall(**tool_call) for tool_call in mock_tool_calls
            ]

        usage_prompt_tokens: Final = (
            prompt_tokens if prompt_tokens is not None else DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT
        )
        setattr(
            model_response,
            "usage",
            Usage(
                prompt_tokens=usage_prompt_tokens,
                completion_tokens=DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT,
                total_tokens=usage_prompt_tokens + DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT,
            ),
        )

        if custom_llm_provider is not None:
            model_response.hidden_params["custom_llm_provider"] = custom_llm_provider
        else:
            try:
                _, inferred_provider, _, _ = litellm.utils.get_llm_provider(model=model)
                model_response.hidden_params["custom_llm_provider"] = inferred_provider
            except Exception:
                # dont let setting a hidden param block a mock_respose
                pass

        if logging is not None:
            logging.post_call(
                input=messages,
                api_key="my-secret-key",
                original_response="my-original-response",
            )

        return model_response

    except Exception as e:
        if isinstance(e, openai.APIError):
            raise e
        raise Exception(f"Mock completion response failed - {e}")
