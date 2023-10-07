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

@pytest.mark.asyncio
async def test_rate_limit_handler():
    import asyncio
    ##### USAGE ################

    from litellm import RateLimitManager

    handler = RateLimitManager(
        max_requests_per_minute = 60,
        max_tokens_per_minute = 200
    )


    async def send_request():
        response =  await handler.acompletion(
            model="gpt-3.5-turbo", 
            messages=[{
                "content": "Please provide a summary of the latest scientific discoveries."*10, 
                "role": "user"
            }]
        )
        print("got a response", response)
        return response


    tasks = []

    for _ in range(4):
        tasks.append(send_request())

    responses = await asyncio.gather(*tasks)

    for response in responses:
        print(response)

# import asyncio
# asyncio.run(
#     test_rate_limit_handler()
# )


@pytest.mark.asyncio
async def test_rate_limit_handler_batch():
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
        handler.batch_completion(
            jobs = jobs,
            api_key=os.environ['OPENAI_API_KEY'],
        )
    except Exception as e:
        print(e)


test_rate_limit_handler_batch()