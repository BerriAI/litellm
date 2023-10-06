#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os
import traceback
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from openai.error import Timeout
import litellm
from litellm import batch_completion, batch_completion_models, completion, batch_completion_models_all_responses
# litellm.set_verbose=True

def test_batch_completions():
    messages = [[{"role": "user", "content": "write a short poem"}] for _ in range(3)]
    model = "gpt-3.5-turbo"
    try:
        result = batch_completion(
            model=model, 
            messages=messages,
            max_tokens=10,
            temperature=0.2
        )
        print(result)
        print(len(result))
        assert(len(result)==3)
    except Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")
# test_batch_completions()

def test_batch_completions_models():
    try:
        result = batch_completion_models(
            models=["gpt-3.5-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo"], 
            messages=[{"role": "user", "content": "Hey, how's it going"}]
        )
        print(result)
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")
# test_batch_completions_models()

def test_batch_completion_models_all_responses():
    responses = batch_completion_models_all_responses(
        models=["j2-light", "claude-instant-1.2", "command-nightly"], 
        messages=[{"role": "user", "content": "write a poem"}],
        max_tokens=500
    )
    print(responses)
    assert(len(responses) == 3)
# test_batch_completion_models_all_responses()

# def test_batch_completions():
#     try:
#         result = completion(
#             model=["gpt-3.5-turbo", "claude-instant-1.2", "command-nightly"], 
#             messages=[{"role": "user", "content": "Hey, how's it going"}]
#         )
#         print(result)
#     except Exception as e:
#         pytest.fail(f"An error occurred: {e}")
# test_batch_completions()



def test_rate_limit_handler():
    import asyncio
    ##### USAGE ################

    jobs = [
        {"model": "gpt-3.5-turbo-16k", "messages": [{"content": "Please provide a summary of the latest scientific discoveries.", "role": "user"}]},
        {"model": "gpt-3.5-turbo-16k", "messages": [{"content": "Please provide a summary of the latest scientific discoveries.", "role": "user"}]},
    ]

    from litellm import RateLimitManager

    handler = RateLimitManager(
        max_requests_per_minute = 60,
        max_tokens_per_minute = 20000
    )

    try:
        asyncio.run(
        handler.batch_completion(
            jobs = jobs,
            api_key=os.environ['OPENAI_API_KEY'],
        )
    )
    except Exception as e:
        print(e)
