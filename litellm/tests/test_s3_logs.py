import sys
import os
import io, asyncio

# import logging
# logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))

from litellm import completion
import litellm

litellm.num_retries = 3

import time, random
import pytest


def test_s3_logging():
    # all s3 requests need to be in one test function
    # since we are modifying stdout, and pytests runs tests in parallel
    # on circle ci - we only test litellm.acompletion()
    try:
        # pre
        # redirect stdout to log_file

        litellm.success_callback = ["s3"]
        litellm.set_verbose = True

        print("Testing async s3 logging")

        expected_keys = []

        async def _test():
            return await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "This is a test"}],
                max_tokens=10,
                temperature=0.7,
                user="ishaan-2",
            )

        response = asyncio.run(_test())
        print(f"response: {response}")
        expected_keys.append(response.id)

        # # streaming + async
        # async def _test2():
        #     response = await litellm.acompletion(
        #         model="gpt-3.5-turbo",
        #         messages=[{"role": "user", "content": "what llm are u"}],
        #         max_tokens=10,
        #         temperature=0.7,
        #         user="ishaan-2",
        #         stream=True,
        #     )
        #     async for chunk in response:
        #         pass

        # asyncio.run(_test2())

        # aembedding()
        # async def _test3():
        #     return await litellm.aembedding(
        #         model="text-embedding-ada-002", input=["hi"], user="ishaan-2"
        #     )

        # response = asyncio.run(_test3())
        # expected_keys.append(response.id)
        # time.sleep(1)

        import boto3

        s3 = boto3.client("s3")
        bucket_name = "litellm-logs"
        # List objects in the bucket
        response = s3.list_objects(Bucket=bucket_name)

        # Sort the objects based on the LastModified timestamp
        objects = sorted(
            response["Contents"], key=lambda x: x["LastModified"], reverse=True
        )
        # Get the keys of the most recent objects
        most_recent_keys = [obj["Key"] for obj in objects]
        print("\n most recent keys", most_recent_keys)
        print("\n Expected keys: ", expected_keys)
        for key in expected_keys:
            assert key in most_recent_keys
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")
    finally:
        # post, close log file and verify
        # Reset stdout to the original value
        print("Passed! Testing async s3 logging")


test_s3_logging()
