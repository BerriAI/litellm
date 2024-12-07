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
from litellm import completion_with_retries, completion
from litellm import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    OpenAIError,
)

user_message = "Hello, whats the weather in San Francisco??"
messages = [{"content": user_message, "role": "user"}]


def logger_fn(user_model_dict):
    # print(f"user_model_dict: {user_model_dict}")
    pass


# completion with num retries + impact on exception mapping
def test_completion_with_num_retries():
    try:
        response = completion(
            model="j2-ultra",
            messages=[{"messages": "vibe", "bad": "message"}],
            num_retries=2,
        )
        pytest.fail(f"Unmapped exception occurred")
    except Exception as e:
        pass


# test_completion_with_num_retries()
def test_completion_with_0_num_retries():
    try:
        litellm.set_verbose = False
        print("making request")

        # Use the completion function
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"gm": "vibe", "role": "user"}],
            max_retries=4,
        )

        print(response)

        # print(response)
    except Exception as e:
        print("exception", e)
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_completion_with_retry_policy(sync_mode):
    from unittest.mock import patch, MagicMock, AsyncMock
    from litellm.types.router import RetryPolicy

    retry_number = 1
    retry_policy = RetryPolicy(
        ContentPolicyViolationErrorRetries=retry_number,  # run 3 retries for ContentPolicyViolationErrors
        AuthenticationErrorRetries=0,  # run 0 retries for AuthenticationErrorRetries
    )

    target_function = "completion_with_retries"

    with patch.object(litellm, target_function) as mock_completion_with_retries:
        data = {
            "model": "azure/gpt-3.5-turbo",
            "messages": [{"gm": "vibe", "role": "user"}],
            "retry_policy": retry_policy,
            "mock_response": "Exception: content_filter_policy",
        }
        try:
            if sync_mode:
                completion(**data)
            else:
                await completion(**data)
        except Exception as e:
            print(e)

        mock_completion_with_retries.assert_called_once()
        assert (
            mock_completion_with_retries.call_args.kwargs["num_retries"] == retry_number
        )
        assert retry_policy.ContentPolicyViolationErrorRetries == retry_number


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_completion_with_retry_policy_no_error(sync_mode):
    """
    Test that the completion function does not throw an error when the retry policy is set
    """
    from unittest.mock import patch, MagicMock, AsyncMock
    from litellm.types.router import RetryPolicy

    retry_number = 1
    retry_policy = RetryPolicy(
        ContentPolicyViolationErrorRetries=retry_number,  # run 3 retries for ContentPolicyViolationErrors
        AuthenticationErrorRetries=0,  # run 0 retries for AuthenticationErrorRetries
    )

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"gm": "vibe", "role": "user"}],
        "retry_policy": retry_policy,
    }
    try:
        if sync_mode:
            completion(**data)
        else:
            await completion(**data)
    except Exception as e:
        print(e)
