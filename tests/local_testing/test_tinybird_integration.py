#### What this tests ####
#    This tests if logging to the tinybird integration actually works
import sys, os
import traceback
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion

litellm.success_callback = ["tinybird"]
litellm.failure_callback = ["tinybird"]


def test_tinybird_logging():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello tell me hi"}],
            user="ishaanRegular",
            max_tokens=10,
        )
        print(response)
    except Exception as e:
        print(e)


# test_tinybird_logging()


def test_acompletion_sync():
    import asyncio
    import time

    async def completion_call():
        try:
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "write a poem"}],
                max_tokens=10,
                stream=False,
                user="ishaanStreamingUser",
                timeout=5,
            )
            print(response)
        except litellm.Timeout as e:
            print(f"Timeout occurred: {e}")
        except Exception as e:
            print(f"error occurred: {traceback.format_exc()}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        task = loop.create_task(completion_call())
        loop.run_until_complete(task)
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending))
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

test_acompletion_sync()

litellm.success_callback = []
litellm.failure_callback = []


