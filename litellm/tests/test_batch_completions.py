#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os
import traceback
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from openai import APITimeoutError as Timeout
import litellm

litellm.num_retries = 0
from litellm import (
    batch_completion,
    batch_completion_models,
    completion,
    batch_completion_models_all_responses,
)

# litellm.set_verbose=True


def test_batch_completions():
    messages = [[{"role": "user", "content": "write a short poem"}] for _ in range(3)]
    model = "gpt-3.5-turbo"
    litellm.set_verbose = True
    try:
        result = batch_completion(
            model=model,
            messages=messages,
            max_tokens=10,
            temperature=0.2,
            request_timeout=1,
        )
        print(result)
        print(len(result))
        assert len(result) == 3
    except Timeout as e:
        print(f"IN TIMEOUT")
        pass
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


# test_batch_completions()


def test_batch_completions_models():
    try:
        result = batch_completion_models(
            models=["gpt-3.5-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo"],
            messages=[{"role": "user", "content": "Hey, how's it going"}],
        )
        print(result)
    except Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


# test_batch_completions_models()


def test_batch_completion_models_all_responses():
    try:
        responses = batch_completion_models_all_responses(
            models=["j2-light", "claude-instant-1.2"],
            messages=[{"role": "user", "content": "write a poem"}],
            max_tokens=10,
        )
        print(responses)
        assert len(responses) == 2
    except Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


# test_batch_completion_models_all_responses()
