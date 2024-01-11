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
        # redirect stdout to log_file
        litellm.cache = litellm.Cache(
            type="s3", s3_bucket_name="cache-bucket-litellm", s3_region_name="us-west-2"
        )

        litellm.success_callback = ["s3"]
        litellm.s3_callback_params = {
            "s3_bucket_name": "litellm-logs",
            "s3_aws_secret_access_key": "os.environ/AWS_SECRET_ACCESS_KEY",
            "s3_aws_access_key_id": "os.environ/AWS_ACCESS_KEY_ID",
        }
        litellm.set_verbose = True

        print("Testing async s3 logging")

        expected_keys = []

        import time

        curr_time = str(time.time())

        async def _test():
            return await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"This is a test {curr_time}"}],
                max_tokens=10,
                temperature=0.7,
                user="ishaan-2",
            )

        response = asyncio.run(_test())
        print(f"response: {response}")
        expected_keys.append(response.id)

        async def _test():
            return await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"This is a test {curr_time}"}],
                max_tokens=10,
                temperature=0.7,
                user="ishaan-2",
            )

        response = asyncio.run(_test())
        expected_keys.append(response.id)
        print(f"response: {response}")

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
        print(most_recent_keys)
        # for each key, get the part before "-" as the key. Do it safely
        cleaned_keys = []
        for key in most_recent_keys:
            split_key = key.split("-time=")
            cleaned_keys.append(split_key[0])
        print("\n most recent keys", most_recent_keys)
        print("\n cleaned keys", cleaned_keys)
        print("\n Expected keys: ", expected_keys)
        matches = 0
        for key in expected_keys:
            assert key in cleaned_keys

            if key in cleaned_keys:
                matches += 1
                # remove the match key
                cleaned_keys.remove(key)
        # this asserts we log, the first request + the 2nd cached request
        print("we had two matches ! passed ", matches)
        assert matches == 2
        try:
            # cleanup s3 bucket in test
            for key in most_recent_keys:
                s3.delete_object(Bucket=bucket_name, Key=key)
        except:
            # don't let cleanup fail a test
            pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")
    finally:
        # post, close log file and verify
        # Reset stdout to the original value
        print("Passed! Testing async s3 logging")


test_s3_logging()
