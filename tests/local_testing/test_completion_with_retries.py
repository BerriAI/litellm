import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import openai
import litellm
from litellm import completion_with_retries, completion, acompletion
from litellm import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    OpenAIError,
    RetryPolicy,
    Timeout,
)
from litellm.integrations.custom_logger import CustomLogger


class CallCounterHandler(CustomLogger):
    success: bool = False
    failure: bool = False
    api_call_count: int = 0

    def log_pre_api_call(self, model, messages, kwargs):
        print(f"Pre-API Call")
        self.api_call_count += 1

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        print(
            f"Post-API Call - response object: {response_obj}; model: {kwargs['model']}"
        )

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")

    def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Success")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Failure")


# completion with num retries + impact on exception mapping
def test_completion_exception_mapping_with_num_retries():
    try:
        response = completion(
            model="j2-ultra",
            messages=[{"messages": "vibe", "bad": "message"}],
            num_retries=2,
        )
        pytest.fail(f"Unmapped exception occurred")
    except Exception as e:
        pass


@pytest.mark.parametrize("max_retries", [0, 3])
def test_completion_max_retries(max_retries):
    call_counter_handler = CallCounterHandler()
    litellm.callbacks = [call_counter_handler]

    with pytest.raises(Exception, match="Invalid Request"):
        completion(
            model="gpt-3.5-turbo",
            messages=[{"gm": "vibe", "role": "user"}],
            mock_response=(Exception("Invalid Request")),
            max_retries=max_retries,
        )

    assert (
        call_counter_handler.api_call_count == max_retries + 1
    )  # 1 initial call + retries


@pytest.mark.parametrize(
    ("Error", "expected_num_retries"),
    [
        (RateLimitError, 3),
        (Timeout, 1),
    ],
)
def test_completion_retry_policy(Error, expected_num_retries):
    call_counter_handler = CallCounterHandler()
    litellm.callbacks = [call_counter_handler]
    retry_policy = RetryPolicy(
        RateLimitErrorRetries=3,
        TimeoutErrorRetries=1,
    )

    with pytest.raises(Error):
        completion(
            model="gpt-3.5-turbo",
            messages=[{"gm": "vibe", "role": "user"}],
            mock_response=(
                Error(message="Bad!", llm_provider="openai", model="gpt-3.5-turbo")
            ),
            # Verify that the retry policy is used instead of the num_retries parameter
            # when both are provided
            max_retries=100,
            retry_policy=retry_policy,
        )

    assert (
        call_counter_handler.api_call_count == expected_num_retries + 1
    )  # 1 initial call + retries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("Error", "expected_num_retries"),
    [
        (RateLimitError, 3),
        (Timeout, 1),
    ],
)
async def test_async_completion_retry_policy(Error, expected_num_retries):
    call_counter_handler = CallCounterHandler()
    litellm.callbacks = [call_counter_handler]
    retry_policy = RetryPolicy(
        RateLimitErrorRetries=3,
        TimeoutErrorRetries=1,
    )

    with pytest.raises(Error):
        await completion(
            model="gpt-3.5-turbo",
            messages=[{"gm": "vibe", "role": "user"}],
            mock_response=(
                Error(message="Bad!", llm_provider="openai", model="gpt-3.5-turbo")
            ),
            # Verify that the retry policy is used instead of the num_retries parameter
            # when both are provided
            max_retries=100,
            retry_policy=retry_policy,
        )

    assert (
        call_counter_handler.api_call_count == expected_num_retries + 1
    )  # 1 initial call + retries
