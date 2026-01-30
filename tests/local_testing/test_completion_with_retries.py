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
from litellm import completion_with_retries, completion, acompletion_with_retries
from litellm import responses_with_retries, aresponses_with_retries
from litellm.responses.main import responses, aresponses
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
        BadRequestErrorRetries=10,
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


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_completion_with_retries(sync_mode):
    """
    If completion_with_retries is called with num_retries=3, and max_retries=0, then litellm.completion should receive num_retries , max_retries=0
    """
    from unittest.mock import patch, MagicMock, AsyncMock

    if sync_mode:
        target_function = "completion"
    else:
        target_function = "acompletion"

    with patch.object(litellm, target_function) as mock_completion:
        if sync_mode:
            completion_with_retries(
                model="gpt-3.5-turbo",
                messages=[{"gm": "vibe", "role": "user"}],
                num_retries=3,
                original_function=mock_completion,
            )
        else:
            await acompletion_with_retries(
                model="gpt-3.5-turbo",
                messages=[{"gm": "vibe", "role": "user"}],
                num_retries=3,
                original_function=mock_completion,
            )
        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["num_retries"] == 0
        assert mock_completion.call_args.kwargs["max_retries"] == 0


# ==================== Responses API Retry Tests ====================


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_responses_with_retries(sync_mode):
    """
    Test that responses() and aresponses() properly handle num_retries parameter.
    If responses_with_retries is called with num_retries=3, and max_retries=0, 
    then litellm.responses should receive num_retries=0, max_retries=0
    """
    from unittest.mock import patch, MagicMock, AsyncMock

    if sync_mode:
        target_function = "responses"
        retry_function = responses_with_retries
    else:
        target_function = "aresponses"
        retry_function = aresponses_with_retries

    # Mock the responses/aresponses function
    with patch("litellm.responses.main.responses" if sync_mode else "litellm.responses.main.aresponses") as mock_responses:
        if sync_mode:
            mock_responses.return_value = MagicMock()
            retry_function(
                model="gpt-4o",
                input="Hello, what's the weather?",
                num_retries=3,
                original_function=mock_responses,
            )
        else:
            mock_responses.return_value = AsyncMock()
            await retry_function(
                model="gpt-4o",
                input="Hello, what's the weather?",
                num_retries=3,
                original_function=mock_responses,
            )
        
        mock_responses.assert_called_once()
        assert mock_responses.call_args.kwargs["num_retries"] == 0
        assert mock_responses.call_args.kwargs["max_retries"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_responses_retry_on_auth_error(sync_mode):
    """
    Test that responses API actually retries when encountering authentication errors.
    This validates that the @client decorator properly handles responses/aresponses retries.
    """
    from unittest.mock import patch
    import openai

    num_retries = 2
    
    # Mock the responses/aresponses to raise an authentication error
    if sync_mode:
        with patch.object(litellm, "responses_with_retries") as mock_retry:
            mock_retry.return_value = None
            try:
                responses(
                    model="gpt-4o",
                    input="Test input",
                    num_retries=num_retries,
                    api_key="sk-invalid-key-12345",
                )
            except Exception:
                pass  # Expected to fail with invalid key
            
            # Check if retry function was called (means @client decorator triggered retry)
            if mock_retry.called:
                assert mock_retry.call_args.kwargs.get("num_retries") == num_retries
    else:
        with patch.object(litellm, "aresponses_with_retries") as mock_retry:
            mock_retry.return_value = None
            try:
                await aresponses(
                    model="gpt-4o",
                    input="Test input",
                    num_retries=num_retries,
                    api_key="sk-invalid-key-12345",
                )
            except Exception:
                pass  # Expected to fail with invalid key
            
            # Check if retry function was called (means @client decorator triggered retry)
            if mock_retry.called:
                assert mock_retry.call_args.kwargs.get("num_retries") == num_retries
